#%%
import os
import sys
import json
import pprint
import argparse
import numpy as np
import xarray as xr
from warnings import warn

import shapely
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from matplotlib.colors import LogNorm

os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'

from tqdm import tqdm
from parse_args import create_parser
from scipy.interpolate import interp1d

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import dask
dask.config.set(scheduler='synchronous')

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

from util import utils, losses
from util.dataLoader import coastalLoader
from util.model_utils import get_model, load_checkpoint
from train_mps import iterate, save_results, prepare_data, prepare_output, seed_packages, seed_worker

# ===================== 强制固定路径 =====================
MODEL_PATH = "/Volumes/code_copy/科研工作2026/StormSurgeCastNet-main/results/myFirstRun/model_epoch_40.pth.tar"
TEST_CACHE = "/Volumes/code_copy/科研工作2026/StormSurgeCastNet-main/Data2/cache/train.npy"
SAVE_FIG  = "/Volumes/code_copy/科研工作2026/StormSurgeCastNet-main/results/myFirstRun/test_scatter.png"

parser = create_parser(mode='test')
test_config = parser.parse_args()

test_config.pid = os.getpid()
test_config.resume_at = 30
test_config.experiment_name = "myFirstRun"
test_config.weight_folder = "results"
test_config.root = "/Volumes/code_copy/科研工作2026/StormSurgeCastNet-main/Data2"
test_config.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

conf_path = os.path.join(dirname, test_config.weight_folder, test_config.experiment_name, "conf.json")
if os.path.isfile(conf_path):
    with open(conf_path) as file:
        model_config = json.loads(file.read())
        t_args = argparse.Namespace()
        no_overwrite = ['root', 'pid', 'device', 'resume_at', 'trained_checkp', 'res_dir', 'weight_folder',
                        'num_workers', 'max_samples_count', 'batch_size', 'input_t', 'lead_time']
        conf_dict = {key:val for key,val in model_config.items() if key not in no_overwrite}
        for key, val in vars(test_config).items():
            if key in no_overwrite: conf_dict[key] = val
        t_args.__dict__.update(conf_dict)
        config = parser.parse_args(namespace=t_args)
else:
    config = test_config

config = utils.str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])

if config.model in ['lstm', 'conv_lstm']:
    if not config.hyperlocal:
        config.hyperlocal = True

experime_dir = os.path.join(config.res_dir, config.experiment_name)
if not os.path.exists(experime_dir):
    os.makedirs(experime_dir)

seed_packages(config.rdm_seed)
f, g = torch.Generator(), torch.Generator()
f.manual_seed(config.rdm_seed + 0)
g.manual_seed(config.rdm_seed)

if __name__ == "__main__":
    pprint.pprint(config)

models = ['utae', 'metnet3', 'lstm', 'conv_lstm']

# ==============================================================================
# ✅ 修复：自定义 CachedDataset，自动过滤 None
# ==============================================================================
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = [d for d in data_list if d is not None]  # 🔥 过滤无效样本
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ==============================================================================
# ========================== 主测试函数（已修改） ===============================
# ==============================================================================
def main(config):
    device = torch.device(config.device)
    prepare_output(config)

    # 加载模型
    model = get_model(config)
    model = model.to(device)
    model.eval()
    load_checkpoint(config, config.weight_folder, model, "model_epoch_30")
    print("✅ 模型加载完成：model_epoch_30.pth.tar")

    # 直接加载 test.npy
    print("✅ 加载测试缓存：", TEST_CACHE)
    test_data = np.load(TEST_CACHE, allow_pickle=True).item()['dataset']

    # ✅ 使用修复后的数据集
    dt_test = CachedDataset(test_data)
    print(f"✅ 有效样本数量：{len(dt_test)}")

    test_loader = torch.utils.data.DataLoader(
        dt_test, batch_size=config.batch_size, shuffle=False,
        worker_init_fn=seed_worker, generator=g, num_workers=0
    )

    # 开始推理
    print("\n🚀 开始测试 ...")
    y_true = []
    y_pred = []

    for batch in tqdm(test_loader):
        if batch is None:
            continue

        x, y, in_m, dates, lead = prepare_data(batch, device, config)
        inputs = {'A': x, 'B': y, 'dates': dates, 'masks': in_m, 'lead': lead}

        with torch.no_grad():
            model.set_input(inputs)
            model.forward()
            out = model.fake_B

        # 收集真值和预测
        y_true.append(y.cpu().numpy().flatten())
        y_pred.append(out.cpu().numpy().flatten())

    # 拼接所有结果
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # 去NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # ===================== 计算指标 =====================
    bias  = np.mean(y_pred - y_true)
    rmse  = np.sqrt(np.mean((y_pred - y_true)**2))
    r     = np.corrcoef(y_true, y_pred)[0,1]
    r2    = r**2

    print("\n" + "="*60)
    print("📊 测试集指标")
    print("="*60)
    print(f"BIAS   = {bias:.4f}")
    print(f"RMSE   = {rmse:.4f}")
    print(f"R      = {r:.4f}")
    print(f"R²     = {r2:.4f}")
    print("="*60)

    # ===================== 绘制高密度散点图 =====================
    plt.figure(figsize=(7,6), dpi=150)

    # 热度散点图
    plt.hist2d(y_true, y_pred, bins=80, cmap='jet', norm=LogNorm())
    plt.colorbar(label='点密度')

    # 对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'w--', lw=2, label='y=x')

    plt.xlabel('真值')
    plt.ylabel('预测')
    plt.title(f'测试集散点图\nBIAS={bias:.3f} | RMSE={rmse:.3f} | R²={r2:.3f}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_FIG, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 图片已保存：\n{SAVE_FIG}")

if __name__ == "__main__":
    main(config)
# %%
