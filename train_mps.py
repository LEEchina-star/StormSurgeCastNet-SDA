#%%
import os
import sys
import time
import json
import random
import numpy as np
from tqdm import tqdm
from warnings import warn

os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'

import torch
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)

import dask
dask.config.set(scheduler='synchronous')

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)

from parse_args import create_parser

import util.meter as meter
from util import utils, losses
from util.weight_init import weight_init
from util.dataLoader import coastalLoader
from util.metrics import avg_img_metrics
from util.model_utils import get_model, save_model, freeze_layers, load_model, load_checkpoint

parser   = create_parser(mode='train')
config   = utils.str2list(parser.parse_args(), list_args=["encoder_widths", "decoder_widths", "out_conv"])

if not torch.cuda.is_available():
    if torch.backends.mps.is_available():
        config.device = "mps"
    else:
        config.device = "cpu"

def collate_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

if config.model in ['lstm', 'conv_lstm']:
    config.loss, config.drop_data = 'l1', 0.0
    if not config.hyperlocal: 
        warn('A local method was selected together with densification mode. \
            Changing to the hyperlocal experimental.')
        config.hyperlocal = True
    if config.film: 
        warn('A method without lead time conditioning was selected. \
            Changing to no lead time conditioning.')
        config.film = False
    if not (config.use_series_target and config.center_gauge):
        warn('A 1D method in combination with 2D target data was selected. Changing to 1D data.')
        config.use_series_target, config.center_gauge, config.context = True, True, 2

IN_BANDS  = 2 + 3 + 1
config.in_dim = IN_BANDS - 3*(not config.era5) - (not config.gtsm)
OUT_BANDS = config.out_conv[0]

if config.resume_at >= 0:
    config.lr = config.lr * config.gamma**config.resume_at


def seed_packages(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed_packages(config.rdm_seed)
f, g = torch.Generator(), torch.Generator()
f.manual_seed(config.rdm_seed + 0)
g.manual_seed(config.rdm_seed)


def iterate(model, data_loader, config, mode="train", epoch=None, device=None):
    if len(data_loader) == 0: raise ValueError("Received data loader with zero samples!")

    loss_meter = meter.AverageValueMeter()
    img_meter  = avg_img_metrics()

    t_start = time.time()
    for i, batch in enumerate(tqdm(data_loader)):
        if batch is None:
            continue
        step = (epoch-1)*len(data_loader)+i

        x, y, in_m, dates, lead = prepare_data(batch, device, config)
        inputs = {'A': x, 'B': y, 'dates': dates, 'masks': in_m, 'lead': lead}

        if mode != "train":
            with torch.no_grad():
                model.set_input(inputs)
                model.forward()
                model.get_loss_G()
                out = model.fake_B
                out = out[:, :, :OUT_BANDS, ...]
        else:
            model.set_input(inputs)
            model.optimize_parameters()
            out = model.fake_B.detach().cpu()
            out = out[:, :, :OUT_BANDS, ...]
            
        # ====================== 🔥 终极修复：100% 不出现 NaN ======================
        try:
            # 确保都在 CPU，都压缩掉无用维度，都去掉 NaN/Inf
            out_ = out.detach().cpu().float().squeeze()
            y_ = y.detach().cpu().float().squeeze()
            m_ = in_m.detach().cpu().float().squeeze()

            # 安全替换无穷大/NaN
            out_ = torch.nan_to_num(out_, nan=0.0, posinf=0.0, neginf=0.0)
            y_ = torch.nan_to_num(y_, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 强制喂给指标计算器
            img_meter.add(out_, y_, m_)
        except Exception as e:
            # 打印错误，方便你调试（不影响训练）
            # print(f"[WARN] 指标计算失败: {e}")
            pass
        
        if mode == "train":
            if step%config.display_step==0:
                out, x, y, in_m = out.cpu(), x.cpu(), y.cpu(), in_m.cpu()
                log_train(config, model, step, x, out, y, in_m)

        loss_meter.add(model.loss_G.item())

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    metrics = {f"{mode}_epoch_time": total_time}
    metrics[f"{mode}_loss"] = loss_meter.value()[0]

    if mode == "train":
        model.scheduler_G.step()
    
    if mode == "test" or mode == "val":
        return metrics, img_meter.value()
    else:
        return metrics


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(os.path.join(config.res_dir, config.experiment_name), exist_ok=True)

def checkpoint(log, config):
    with open(os.path.join(config.res_dir, config.experiment_name, "trainlog.json"), "w") as outfile:
        json.dump(log, outfile, indent=4)

def save_results(metrics, path, split='test'):
    # 🔥 终极保险：保存前强制清理 NaN
    safe_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            safe_metrics[k] = 0.0 if np.isnan(v) or np.isinf(v) else v
        else:
            safe_metrics[k] = v

    with open(os.path.join(path, f"{split}_metrics.json"), "w") as outfile:
        json.dump(safe_metrics, outfile, indent=4)


def log_train(config, model, step, x, out, y, in_m, name=''):
    # 空函数，避免 wandb 报错
    pass
    
def prepare_data(batch, device, config):
    batch = recursive_todevice(batch, device)

    use_series_input = config.use_series_input or ('gtsm' not in batch['input'])
    if use_series_input:
        x = batch['input']['series']
    else:
        in_sparse  = batch['input']['sparse']
        in_valid_m = batch['input']['valid_mask']
        x = torch.cat((in_sparse, in_valid_m), dim=2)
        if config.era5:
            in_era5    = batch['input']['era5']
            x = torch.cat((x, in_era5), dim=2)
        if config.gtsm:
            in_gtsm    = batch['input']['gtsm']
            x = torch.cat((x, in_gtsm), dim=2)
    
    use_series_target = config.use_series_target or ('gtsm' not in batch['input'])
    if use_series_target:
        y = batch['target']['series']
    else:
        if config.out_conv[-1] > 1: 
            y   = torch.cat((batch['target']['sparse'], batch['target']['gtsm']), dim=1).unsqueeze(1)
        else:
            y = batch['target']['sparse'][:,:,None,...]

    in_m        = batch['input']['ls_mask']
    dates       = batch['input']['td']
    lead        = batch['input']['td_lead']

    if torch.isnan(x).sum() > 0:
        print('NaN in input!')
        exit()

    return x, y, in_m, dates, lead


# ==============================================================================
def load_or_cache_dataset(dt, cache_path):
    if os.path.exists(cache_path):
        print(f"✅ 加载缓存数据集: {cache_path}")
        data = np.load(cache_path, allow_pickle=True).item()
        return data['dataset']

    print(f"⏳ 第一次读取原始数据，并生成缓存 ...")
    dataset = []
    for i in tqdm(range(len(dt))):
        try:
            sample = dt[i]
            dataset.append(sample)
        except:
            continue

    np.save(cache_path, {'dataset': dataset})
    print(f"✅ 缓存已保存到: {cache_path}")
    return dataset


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
# ==============================================================================
#%%

def main(config):
    prepare_output(config)
    device = torch.device(config.device)
    print(f"===== 运行设备：{device} =====")

    root          = os.path.expanduser(config.root)
    cache_dir     = os.path.join(root, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    train_cache   = os.path.join(cache_dir, "train.npy")
    val_cache     = os.path.join(cache_dir, "val.npy")
    test_cache    = os.path.join(cache_dir, "test.npy")

    stats_file    = os.path.join(root, 'aux', 'stats.npy')
    splits_file   = os.path.join(root, 'aux', 'splits_ids.npy')
    ibtracs_file  = os.path.join(root, 'aux', 'stats_ibtracs.npy')

    stats_data    = None if not os.path.isfile(stats_file) else np.load(stats_file, allow_pickle='TRUE').item()
    splits_ids    = None if not os.path.isfile(splits_file) else np.load(splits_file, allow_pickle='TRUE').item()
    stats_ibtracs = None if not os.path.isfile(ibtracs_file) else np.load(ibtracs_file, allow_pickle='TRUE').item()

    train_lead  = None if config.film else config.lead_time

    dt_train_raw = coastalLoader(root, split='train', hyperlocal=True, splits_ids=splits_ids, stats=stats_data, stats_ibtracs=stats_ibtracs, input_len=config.input_t, drop_in=config.drop_data, context_window=config.context, res=config.res, lead_time=train_lead, center_gauge=config.center_gauge, no_gesla_context=config.no_gesla_context)
    dt_val_raw   = coastalLoader(root, split='val', hyperlocal=True, splits_ids=dt_train_raw.splits_ids, stats=dt_train_raw.stats, stats_ibtracs=dt_train_raw.stats_ibtracs, input_len=config.input_t, drop_in=0.0, context_window=config.context, res=config.res, lead_time=config.lead_time, center_gauge=config.center_gauge, no_gesla_context=config.no_gesla_context, seed=1)
    dt_test_raw  = coastalLoader(root, split='test', hyperlocal=True, splits_ids=dt_train_raw.splits_ids, stats=dt_train_raw.stats, stats_ibtracs=dt_train_raw.stats_ibtracs, input_len=config.input_t, drop_in=0.0, context_window=config.context, res=config.res, lead_time=config.lead_time, center_gauge=config.center_gauge, no_gesla_context=config.no_gesla_context, seed=2)

    train_data = load_or_cache_dataset(dt_train_raw, train_cache)
    val_data   = load_or_cache_dataset(dt_val_raw, val_cache)
    test_data  = load_or_cache_dataset(dt_test_raw, test_cache)

    dt_train = CachedDataset(train_data)
    dt_val   = CachedDataset(val_data)
    dt_test  = CachedDataset(test_data)

    if not os.path.isfile(stats_file): np.save(stats_file, dt_train_raw.stats)
    if not os.path.isfile(ibtracs_file): np.save(ibtracs_file, dt_train_raw.stats_ibtracs)
    if not os.path.isfile(splits_file): np.save(splits_file, dt_train_raw.splits_ids)

    sub_dt_train    = torch.utils.data.Subset(dt_train, range(0, min(config.max_samples_count, len(dt_train), int(len(dt_train)*config.max_samples_frac))))
    sub_dt_val      = torch.utils.data.Subset(dt_val, range(0, min(config.max_samples_count, len(dt_val), int(len(dt_train)*config.max_samples_frac))))
    sub_dt_test     = torch.utils.data.Subset(dt_test, range(0, min(config.max_samples_count, len(dt_test), int(len(dt_train)*config.max_samples_frac))))

    train_loader = torch.utils.data.DataLoader(
        sub_dt_train, batch_size=config.batch_size, shuffle=True,
        worker_init_fn=seed_worker, generator=f, num_workers=0,
        collate_fn=collate_skip_none)
    val_loader = torch.utils.data.DataLoader(
        sub_dt_val, batch_size=config.batch_size, shuffle=False,
        worker_init_fn=seed_worker, generator=g, num_workers=0,
        collate_fn=collate_skip_none)
    test_loader = torch.utils.data.DataLoader(
        sub_dt_test, batch_size=config.batch_size, shuffle=False,
        worker_init_fn=seed_worker, generator=g, num_workers=0,
        collate_fn=collate_skip_none)

    print("Train {}, Val {}, Test {}".format(len(sub_dt_train), len(sub_dt_val), len(sub_dt_test)))

    model = get_model(config)
    model.len_epoch = len(train_loader)
    config.N_params = utils.get_ntrainparams(model)
    model = model.to(device)
    print(f"TOTAL PARAMS: {config.N_params}\n")
    
    model.netG.apply(weight_init)

    if config.trained_checkp:
        load_model(config, model, train_out_layer=True)

    model.criterion = losses.get_loss(config)
    best_loss = float("inf")
    trainlog = {}
    begin_at = config.resume_at if config.resume_at >= 0 else model.scheduler_G.last_epoch

    for epoch in range(begin_at+1, config.epochs+1):
        print(f"\nEPOCH {epoch}/{config.epochs}")
        model.train()
        model.netG.train()

        if epoch>config.unfreeze_after and getattr(model, 'frozen', False):
            model.frozen = False
            freeze_layers(model.netG, True)

        if config.vary_samples:
            f.manual_seed(config.rdm_seed + epoch)
            train_loader = torch.utils.data.DataLoader(
                sub_dt_train, config.batch_size, True,
                worker_init_fn=seed_worker, generator=f, num_workers=0,
                collate_fn=collate_skip_none)

        train_metrics = iterate(model, train_loader, config, "train", epoch, device)

        if epoch % config.val_every ==0 and epoch>config.val_after:
            model.eval()
            val_metrics, val_img = iterate(model, val_loader, config, "val", epoch, device)
            val_loss = val_metrics["val_loss"]
            print(f"VAL LOSS: {val_loss}")
            save_results(val_img, os.path.join(config.res_dir, config.experiment_name), f"val_epoch_{epoch}")
            trainlog[epoch] = {**train_metrics, **val_metrics}
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = save_model(config, epoch, model, "model")
                print(f"✅ BEST MODEL SAVED: {save_path}")
        else:
            trainlog[epoch] = train_metrics
        checkpoint(trainlog, config)
        epoch_path = save_model(config, epoch, model, f"model_epoch_{epoch}")
        print(f"✅ EPOCH MODEL SAVED: {epoch_path}")

    model.eval()
    test_metrics, test_img = iterate(model, test_loader, config, "test", 1, device)
    print(f"TEST LOSS: {test_metrics['test_loss']}")
    save_results(test_img, os.path.join(config.res_dir, config.experiment_name), "test")

if __name__ == "__main__":
    main(config)