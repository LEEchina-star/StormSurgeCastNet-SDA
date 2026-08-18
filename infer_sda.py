# -*- coding: utf-8 -*-
"""
SDA-Diff inference: likelihood-guided posterior sampling (data assimilation) on a
trained conditional diffusion model, producing an ensemble of dense surge fields
plus calibration metrics (CRPS, interval coverage, PIT, spread-skill) and an
exceedance-probability diagnostic plot.

Usage (smoke test):
  python infer_sda.py --model edm_da --root Data2 --cache_dir cache_smoke \
      --experiment_name sda_smoke --res_dir results_sda \
      --context 32 --res 0.5 --input_t 4 --lead_time 6 --gtsm_years 2010-2014 \
      --ensemble 4 --sample_steps 15 --sda_guidance 1.0 --obs_noise 0.1 \
      --max_samples_count 2 --num_workers 0
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)

from parse_args import create_parser
from util import utils
from util.dataLoader import coastalLoader
from util.model_utils import get_model
import util.metrics as mets


def seed_packages(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_gtsm_years(s):
    a, b = s.split("-")
    return (int(a), int(b))


def main(config):
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        config.device = "mps"
    device = torch.device(config.device)
    print(f"===== device: {device} =====\n")

    res_dir = os.path.join(config.res_dir, config.experiment_name)
    out_dir = os.path.join(res_dir, "sda_inference")
    os.makedirs(out_dir, exist_ok=True)

    root = os.path.expanduser(config.root)
    cache_dir = config.cache_dir if config.cache_dir else os.path.join(root, "cache")
    gtsm_years = parse_gtsm_years(config.gtsm_years)

    # ---- load checkpoint (config from train run) ----
    # only override args NOT explicitly given on the CLI (architecture must match training)
    parser_defaults = vars(parser.parse_args([]))
    given_on_cli = {k for k, v in vars(config).items()
                    if k in parser_defaults and v != parser_defaults[k]}
    if os.path.isfile(os.path.join(res_dir, "conf.json")):
        with open(os.path.join(res_dir, "conf.json")) as f:
            train_conf = json.load(f)
        for k, v in train_conf.items():
            if k not in given_on_cli:
                setattr(config, k, v)
        print(f"loaded train config from conf.json ({len(train_conf)} keys; "
              f"kept {len(given_on_cli)} CLI overrides)")
    ckpt = os.path.join(res_dir, "best_edm_da_model.pth.tar")
    if not os.path.isfile(ckpt):
        cands = sorted([f for f in os.listdir(res_dir) if f.endswith(".pth.tar")])
        if not cands:
            raise FileNotFoundError(f"no checkpoint in {res_dir}")
        ckpt = os.path.join(res_dir, cands[-1])
    chk = torch.load(ckpt, map_location=device)
    print(f"loaded checkpoint {ckpt} (epoch {chk.get('epoch', '?')})")

    # ---- model ----
    config.in_dim = 6 - 3 * (not config.era5) - (not config.gtsm)
    model = get_model(config).to(device)
    if chk.get("ema") is not None:
        model.load_state_dict(chk["ema"])
        print("using EMA weights")
    else:
        model.load_state_dict(chk["state_dict"])
    model.eval()

    # ---- small val/test sample cache ----
    stats_file   = os.path.join(root, "aux", "stats.npy")
    splits_file  = os.path.join(root, "aux", "splits_ids.npy")
    ibtracs_file = os.path.join(root, "aux", "stats_ibtracs.npy")
    stats_data    = None if not os.path.isfile(stats_file) else np.load(stats_file, allow_pickle="TRUE").item()
    splits_ids    = None if not os.path.isfile(splits_file) else np.load(splits_file, allow_pickle="TRUE").item()
    stats_ibtracs = None if not os.path.isfile(ibtracs_file) else np.load(ibtracs_file, allow_pickle="TRUE").item()

    split = "val"
    cache_path = os.path.join(cache_dir, f"{split}_sda.npy")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()["dataset"]
        print(f"loaded cache {cache_path} ({len(data)} samples)")
    else:
        dt = coastalLoader(root, split=split, hyperlocal=config.hyperlocal,
                           splits_ids=splits_ids, stats=stats_data,
                           stats_ibtracs=stats_ibtracs, input_len=config.input_t,
                           drop_in=0.0, context_window=config.context, res=config.res,
                           lead_time=config.lead_time, center_gauge=config.center_gauge,
                           no_gesla_context=config.no_gesla_context, seed=2,
                           gtsm_years=gtsm_years)
        data, idx = [], 0
        while len(data) < min(config.max_samples_count, 8) and idx < len(dt) * 4:
            try:
                s_ = dt[idx % len(dt)]
            except Exception:
                s_ = None
            if s_ is not None:
                data.append(s_)
            idx += 1
        print(f"built {len(data)} samples from split '{split}'")

    # ---- prepare tensors ----
    def prep(sample):
        # single sample: [T,C,H,W], channels are dim=1
        def _t(v):
            return v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
        x = torch.cat((_t(sample["input"]["sparse"]), _t(sample["input"]["valid_mask"])), dim=1)
        if config.era5:
            x = torch.cat((x, _t(sample["input"]["era5"])), dim=1)
        if config.gtsm:
            x = torch.cat((x, _t(sample["input"]["gtsm"])), dim=1)
        lead = torch.as_tensor(sample["input"]["td_lead"], dtype=torch.float32).reshape(1)
        # target sparse: [1,H,W] -> [1,1,H,W]
        y_sparse = _t(sample["target"]["sparse"]).unsqueeze(1)
        mask = (~torch.isnan(y_sparse)).float()
        x0 = torch.nan_to_num(y_sparse, 0.0)
        return x.unsqueeze(0), x0, mask, lead   # add batch dim: [1,T,C,H,W]

    # ---- posterior sampling + metrics ----
    R = max(config.obs_noise, 1e-3)
    all_metrics = []
    for i in range(min(config.max_samples_count, len(data))):
        x, x0, mask, lead = prep(data[i])
        x, x0, mask, lead = x.to(device), x0.to(device), mask.to(device), lead.to(device)
        y_obs = torch.where(mask > 0, x0, torch.full_like(x0, float("nan")))

        out = model.sample_posterior(
            x, lead, y=y_obs, mask=mask, R=R,
            steps=config.sample_steps, guidance=config.sda_guidance,
            ensemble=config.ensemble, sigma_max=config.sigma_max,
            sigma_min=config.sigma_min, seed=42 + i,
            device="cpu" if device.type == "mps" else None)

        mean, q05, q50, q95 = (v.detach().cpu().numpy() for v in
                               (out["mean"], out["q05"], out["q50"], out["q95"]))
        samples = out["samples"].detach().cpu().numpy()
        y_np, m_np = x0.cpu().numpy(), mask.cpu().numpy()

        # gauge-pixel metrics (densification skill at observation sites)
        met = {
            "sample": i,
            "gauge_lon": float(data[i]["target"].get("lon_gauge", np.nan)),
            "gauge_lat": float(data[i]["target"].get("lat_gauge", np.nan)),
            "crps": mets.crps_ensemble(samples[0], y_np[0], m_np[0]),
            "coverage_90": mets.interval_coverage(samples[0], y_np[0], m_np[0], alpha=0.1),
            "rmse_mean_m": float(np.sqrt(((mean[0, 0][m_np[0, 0] > 0] - y_np[0, 0][m_np[0, 0] > 0]) ** 2).mean())) if (m_np[0, 0] > 0).any() else float("nan"),
        }
        pit = mets.pit_values(samples[0], y_np[0], m_np[0])
        met["pit_mean"] = float(pit.mean()) if pit.size else float("nan")
        met["pit_hist"] = np.histogram(pit, bins=10, range=(0, 1))[0].tolist() if pit.size else []
        spread, skill = mets.spread_skill(samples[0], y_np[0], m_np[0])
        met["spread_mean"] = float(spread.mean()) if spread.size else float("nan")
        met["skill_mean"] = float(skill.mean()) if skill.size else float("nan")

        # exceedance probability over the FULL field (dense, not only gauges)
        exceed = (samples[0][:, 0] > config.surge_threshold).mean(axis=0)
        met["exceed_mean_frac"] = float(exceed.mean())
        all_metrics.append(met)
        print(f"sample {i}: CRPS {met['crps']:.4f} | cov90 {met['coverage_90']:.3f} | "
              f"RMSE {met['rmse_mean_m']:.4f} m | PIT {met['pit_mean']:.3f}")

        # diagnostic plot (first sample)
        if i == 0:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
                vmin, vmax = float(np.nanpercentile(mean, 2)), float(np.nanpercentile(mean, 98))
                axes[0].imshow(mean[0, 0], cmap="RdBu_r", vmin=vmin, vmax=vmax)
                axes[0].set_title("posterior mean (surge)")
                axes[1].imshow(q95[0, 0] - q05[0, 0], cmap="viridis")
                axes[1].set_title("90% interval width")
                axes[2].imshow(exceed, cmap="magma", vmin=0, vmax=1)
                axes[2].set_title(f"P(surge>{config.surge_threshold}m)")
                axes[3].imshow(np.where(m_np[0, 0] > 0, y_np[0, 0], np.nan), cmap="RdBu_r",
                               vmin=vmin, vmax=vmax)
                axes[3].set_title("gauge observations")
                for a in axes:
                    a.axis("off")
                fig.suptitle(f"SDA posterior (N={config.ensemble}, steps={config.sample_steps}, "
                             f"guidance={config.sda_guidance})")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "sda_example.png"), dpi=200)
                print("saved", os.path.join(out_dir, "sda_example.png"))
            except Exception as e:
                print("plot skipped:", e)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nmetrics saved to {out_dir}/metrics.json")


if __name__ == "__main__":
    parser = create_parser(mode="test")
    config = utils.str2list(parser.parse_args(),
                            list_args=["encoder_widths", "decoder_widths", "out_conv"])
    seed_packages(config.rdm_seed)
    main(config)
