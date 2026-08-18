# -*- coding: utf-8 -*-
"""
SDA-Diff training: conditional score-based diffusion (EDM) with sparse masking
and optional GTSM-residual targets, plus likelihood-guided validation sampling.

Usage (smoke test example):
  python train_EDMda.py --model edm_da --root Data2 --cache_dir cache_smoke \
      --experiment_name sda_smoke --res_dir results_sda \
      --context 32 --res 0.5 --input_t 4 --lead_time 6 --gtsm_years 2010-2014 \
      --max_samples_count 8 --epochs 2 --batch_size 2 --diff_dim 32 --diff_levels 3 \
      --sample_steps 10 --ensemble 2 --num_workers 0
"""
import os
import sys
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_default_dtype(torch.float32)

import dask
dask.config.set(scheduler="synchronous")

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)

from parse_args import create_parser
import util.meter as meter
from util import utils, losses
from util.dataLoader import coastalLoader
from util.model_utils import get_model, save_model
import util.metrics as mets


# ----------------------------------------------------------------------------
# seed & device
# ----------------------------------------------------------------------------
def seed_packages(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_gtsm_years(s):
    a, b = s.split("-")
    return (int(a), int(b))


def main(config):
    # ---- device ----
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        config.device = "mps"
    device = torch.device(config.device)
    print(f"===== device: {device} =====\n")

    # ---- output dirs ----
    res_dir = os.path.join(config.res_dir, config.experiment_name)
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "conf.json"), "w") as f:
        f.write(json.dumps(vars(config), indent=4))

    root = os.path.expanduser(config.root)
    cache_dir = config.cache_dir if config.cache_dir else os.path.join(root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    gtsm_years = parse_gtsm_years(config.gtsm_years)

    # ---- stats / splits (precomputed) ----
    stats_file   = os.path.join(root, "aux", "stats.npy")
    splits_file  = os.path.join(root, "aux", "splits_ids.npy")
    ibtracs_file = os.path.join(root, "aux", "stats_ibtracs.npy")
    stats_data    = None if not os.path.isfile(stats_file) else np.load(stats_file, allow_pickle="TRUE").item()
    splits_ids    = None if not os.path.isfile(splits_file) else np.load(splits_file, allow_pickle="TRUE").item()
    stats_ibtracs = None if not os.path.isfile(ibtracs_file) else np.load(ibtracs_file, allow_pickle="TRUE").item()

    # ---- build (or load) a limited cache of real samples ----
    def build_cache(split, n_wanted, seed=None):
        cache_path = os.path.join(cache_dir, f"{split}_sda.npy")
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True).item()
            print(f"loaded cache {cache_path} ({len(data['dataset'])} samples)")
            return data["dataset"]
        print(f"building cache {cache_path}: up to {n_wanted} samples from split '{split}' ...")
        dt = coastalLoader(
            root, split=split, hyperlocal=config.hyperlocal,
            splits_ids=splits_ids, stats=stats_data, stats_ibtracs=stats_ibtracs,
            input_len=config.input_t, drop_in=0.0 if split != "train" else config.drop_data,
            context_window=config.context, res=config.res,
            lead_time=config.lead_time, center_gauge=config.center_gauge,
            no_gesla_context=config.no_gesla_context, seed=seed, gtsm_years=gtsm_years)
        data, seen, idx = [], 0, 0
        pbar = tqdm(total=n_wanted, desc=f"cache {split}")
        while len(data) < n_wanted and idx < len(dt) * 4:
            try:
                sample = dt[idx % len(dt)]
            except Exception:
                sample = None
            if sample is not None:
                data.append(sample)
                pbar.update(1)
            idx += 1
        pbar.close()
        if len(data) == 0:
            raise RuntimeError(f"no valid samples built for split '{split}'")
        np.save(cache_path, {"dataset": data})
        print(f"saved cache {cache_path} ({len(data)} samples)")
        return data

    n_train = min(config.max_samples_count, 32)
    n_val = min(max(2, config.max_samples_count // 2), 8)
    train_data = build_cache("train", n_train, seed=config.rdm_seed)
    val_data   = build_cache("val", n_val, seed=1)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i): return self.d[i]

    def collate(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.default_collate(batch)

    train_loader = torch.utils.data.DataLoader(
        _DS(train_data), batch_size=config.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(
        _DS(val_data), batch_size=config.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate)

    print(f"Train {len(train_data)} | Val {len(val_data)}")

    # ---- model ----
    config.in_dim = 6 - 3 * (not config.era5) - (not config.gtsm)
    model = get_model(config).to(device)
    ema_model = get_model(config).to(device) if config.ema_decay > 0 else None
    if ema_model is not None:
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config.gamma)
    print(f"TOTAL PARAMS: {utils.get_ntrainparams(model)}\n")

    # ---- EMA update ----
    def ema_update(decay):
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)

    def use_ema():
        return ema_model if ema_model is not None else model

    # ---- prepare a batch for the EDM objective ----
    def prep_batch(batch, device):
        for k in ("input", "target"):
            for kk, v in batch[k].items():
                if isinstance(v, torch.Tensor):
                    batch[k][kk] = v.float().to(device)
        # context c: [B,T,C,H,W]
        x = torch.cat((batch["input"]["sparse"], batch["input"]["valid_mask"]), dim=2)
        if config.era5:
            x = torch.cat((x, batch["input"]["era5"]), dim=2)
        if config.gtsm:
            x = torch.cat((x, batch["input"]["gtsm"]), dim=2)
        lead = batch["input"]["td_lead"].float().to(device)
        # target: sparse surge at lead time, [B,1,H,W] (NaN = unobserved pixels)
        y_sparse = batch["target"]["sparse"]
        mask = (~torch.isnan(y_sparse)).float()
        x0 = torch.nan_to_num(y_sparse, 0.0)
        if config.use_residual and config.gtsm:
            y_gtsm = batch["target"]["gtsm"][:, :, 0]
            x0 = x0 - torch.nan_to_num(y_gtsm, 0.0)
        return x, x0, mask, lead

    # ---- one train step ----
    def train_step(batch):
        x, x0, mask, lead = prep_batch(batch, device)
        loss = model.denoise_loss(x0, x, lead, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ema_model is not None:
            ema_update(config.ema_decay)
        return float(loss.item())

    # ---- validation: SDA posterior sampling + probabilistic metrics ----
    @torch.no_grad()
    def validate():
        m = use_ema()
        crps_all, cov_all, rmse_all, pit_all = [], [], [], []
        steps = min(config.sample_steps, 12)
        ens_n = min(config.ensemble, 3)
        sample_dev = "cpu" if device.type == "mps" else None   # dodge flaky MPS autograd
        for batch in val_loader:
            if batch is None:
                continue
            x, x0, mask, lead = prep_batch(batch, device)
            y_obs = torch.where(mask > 0, x0, torch.full_like(x0, float("nan")))
            R = max(config.obs_noise, 1e-3)   # obs noise std (m), R = R^2 I
            out = m.sample_posterior(
                x, lead, y=y_obs, mask=mask, R=R,
                steps=steps, guidance=config.sda_guidance, ensemble=ens_n,
                sigma_max=config.sigma_max, sigma_min=config.sigma_min, seed=0,
                device=sample_dev)
            mean, samples = out["mean"].cpu().numpy(), out["samples"].cpu().numpy()
            y_np, m_np = x0.cpu().numpy(), mask.cpu().numpy()
            for bi in range(mean.shape[0]):
                crps_all.append(mets.crps_ensemble(samples[bi], y_np[bi], m_np[bi]))
                cov_all.append(mets.interval_coverage(samples[bi], y_np[bi], m_np[bi], alpha=0.1))
                pit_all.extend(mets.pit_values(samples[bi], y_np[bi], m_np[bi]))
                diff = (mean[bi, 0][m_np[bi, 0] > 0] - y_np[bi, 0][m_np[bi, 0] > 0])
                rmse_all.append(float(np.sqrt((diff ** 2).mean())) if diff.size else float("nan"))
        def _nanmean(a):
            a = np.array([v for v in a if v == v], dtype=np.float64)
            return float(a.mean()) if a.size else float("nan")
        return dict(val_crps=_nanmean(crps_all), val_coverage=_nanmean(cov_all),
                    val_rmse=_nanmean(rmse_all), val_pit_mean=_nanmean(pit_all))

    # ============================ TRAIN LOOP ============================
    best = float("inf")
    trainlog = {}
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses_run = meter.AverageValueMeter()
        t0 = time.time()
        for i, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
            if batch is None:
                continue
            l = train_step(batch)
            losses_run.add(l)
        sched.step()
        msg = f"[EPOCH {epoch}/{config.epochs}] loss {losses_run.value()[0]:.5f} | {time.time()-t0:.1f}s"
        trainlog[epoch] = {"train_loss": losses_run.value()[0]}

        if epoch % config.val_every == 0:
            v = validate()
            msg += f" | CRPS {v['val_crps']:.4f} | cov {v['val_coverage']:.3f} | RMSE {v['val_rmse']:.4f} m"
            trainlog[epoch].update(v)
            if v["val_crps"] < best:
                best = v["val_crps"]
                save_model(config, epoch, use_ema(), "best_edm_da_model")
        print(msg)
        with open(os.path.join(res_dir, "trainlog.json"), "w") as f:
            json.dump(trainlog, f, indent=4)
        # periodic checkpoint (ema weights + optimizer)
        torch.save({
            "epoch": epoch, "config": vars(config),
            "model": model.state_dict(),
            "ema": ema_model.state_dict() if ema_model is not None else None,
            "optimizer": opt.state_dict(),
        }, os.path.join(res_dir, f"edm_da_epoch_{epoch}.pth.tar"))
    print(f"\ndone. best val CRPS {best:.4f} -> results in {res_dir}")


if __name__ == "__main__":
    parser = create_parser(mode="train")
    config = utils.str2list(parser.parse_args(),
                            list_args=["encoder_widths", "decoder_widths", "out_conv"])
    seed_packages(config.rdm_seed)
    main(config)
