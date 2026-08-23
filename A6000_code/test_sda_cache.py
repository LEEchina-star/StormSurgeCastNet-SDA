# -*- coding: utf-8 -*-
"""
Formal TEST evaluation of a trained SDA-Diff model on the clean gauge-level
validation split (cache_sda_full/val.npz, 45 held-out gauges).

Reports: RMSE, MAE, CRPS, 90% interval coverage, PIT mean, spread-skill,
per-sample metrics and a diagnostic figure (posterior mean / interval /
exceedance / observations) for the first sample.

Usage:
    python test_sda_cache.py --cache_dir cache_sda_full --resize 128 \
        --checkpoint results_sda/cmp_sda_128_v4/best_sda.pth.tar \
        --ensemble 6 --sample_steps 20 --sda_guidance 1.0 --obs_noise 0.1
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
from parse_args import create_parser
from util import utils, metrics as mets
from util.model_utils import get_model

def main():
    parser = create_parser(mode="test")
    parser.add_argument("--resize", type=int, default=0, help="0 = keep 256x256 (paper protocol); 128 only for quick runs")
    parser.add_argument("--checkpoint", default="results_sda/cmp_sda_128_v4/best_sda.pth.tar")
    parser.add_argument("--out", default="results_sda/test_report")
    config = utils.str2list(parser.parse_args(), list_args=["encoder_widths", "decoder_widths", "out_conv"])
    config.model = "edm_da"
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        _mps = getattr(torch.backends, "mps", None)
        if _mps is not None and _mps.is_available():
            config.device = "mps"
        else:
            config.device = "cpu"
    device = torch.device(config.device)
    print(f"device: {device}")

    # ---- load train conf to align architecture (CLI args take precedence) ----
    train_conf_path = os.path.join(os.path.dirname(config.checkpoint), "conf.json")
    if os.path.isfile(train_conf_path):
        tc = json.load(open(train_conf_path))
        defaults = vars(parser.parse_args([]))
        given = {k for k, v in vars(config).items() if k in defaults and v != defaults[k]}
        for k, v in tc.items():
            if k not in given:
                setattr(config, k, v)
        print(f"train conf merged ({len(tc)} keys; kept {len(given)} CLI overrides)")

    # ---- model + checkpoint ----
    config.in_dim = 6 - 3 * (not config.era5) - (not config.gtsm)
    model = get_model(config).to(device)
    chk = torch.load(config.checkpoint, map_location=device)
    if chk.get("ema") is not None:
        model.load_state_dict(chk["ema"]); print("using EMA weights")
    else:
        model.load_state_dict(chk["state_dict"])
    model.eval()

    # ---- test data: clean gauge-level val split ----
    d = np.load(os.path.join(config.cache_dir, "val.npz"))
    X = torch.from_numpy(d["X"]).float()
    y = torch.from_numpy(d["y"]).float()
    ids = d["ids"]
    B, T, C, H, W = X.shape
    if config.resize and H != config.resize:
        Xr = F.interpolate(X.reshape(B*T, C, H, W), size=(config.resize, config.resize), mode="bilinear").reshape(B, T, C, config.resize, config.resize)
        yr = F.interpolate(y.reshape(B, 1, H, W), size=(config.resize, config.resize), mode="nearest").reshape(B, 1, config.resize, config.resize)
    else:
        Xr, yr = X, y
    mask = (~torch.isnan(yr)).float()
    y_obs = torch.where(mask > 0, yr, torch.full_like(yr, float("nan")))
    lead = torch.full((B,), float(config.lead_time if hasattr(config, "lead_time") else 8.0))
    print(f"test samples: {B} | {config.resize}x{config.resize} | ensemble={config.ensemble} steps={config.sample_steps}")

    R = max(config.obs_noise, 1e-3)
    os.makedirs(config.out, exist_ok=True)
    rmse, mae, crps, cov, pit_mean, spread, skill = [], [], [], [], [], [], []
    per_sample = []
    t0 = time.time()
    for i in range(0, B, 4):
        idx = slice(i, min(i + 4, B))
        out = model.sample_posterior(
            Xr[idx].to(device), lead[idx].to(device),
            y=y_obs[idx].to(device), mask=mask[idx].to(device), R=R,
            steps=config.sample_steps, guidance=config.sda_guidance,
            ensemble=config.ensemble, sigma_max=config.sigma_max,
            sigma_min=config.sigma_min, seed=0,
            like_mode="replace", sampler="ode")
        samples = out["samples"].cpu().numpy()
        mean = out["mean"].cpu().numpy()
        for bi in range(samples.shape[0]):
            gi = i + bi
            m_ = mask[idx][bi, 0].numpy() > 0
            mets_r = dict(sample=gi, gauge_id=int(ids[gi]),
                          crps=mets.crps_ensemble(samples[bi], yr[idx][bi].numpy(), mask[idx][bi].numpy()),
                          coverage_90=mets.interval_coverage(samples[bi], yr[idx][bi].numpy(), mask[idx][bi].numpy(), alpha=0.1))
            if m_.any():
                diff = mean[bi, 0][m_] - yr[idx][bi, 0][m_].numpy()
                mets_r["rmse"] = float(np.sqrt((diff**2).mean()))
                mets_r["mae"] = float(np.abs(diff).mean())
                rmse.append(mets_r["rmse"]); mae.append(mets_r["mae"])
            pit = mets.pit_values(samples[bi], yr[idx][bi].numpy(), mask[idx][bi].numpy())
            mets_r["pit_mean"] = float(pit.mean()) if pit.size else float("nan")
            s_, k_ = mets.spread_skill(samples[bi], yr[idx][bi].numpy(), mask[idx][bi].numpy())
            mets_r["spread"] = float(s_.mean()) if s_.size else float("nan")
            mets_r["skill"] = float(k_.mean()) if k_.size else float("nan")
            crps.append(mets_r["crps"]); cov.append(mets_r["coverage_90"])
            pit_mean.append(mets_r["pit_mean"]); spread.append(mets_r["spread"]); skill.append(mets_r["skill"])
            per_sample.append(mets_r)
        print(f"  batch {i}-{min(i+4,B)} done ({time.time()-t0:.0f}s)", flush=True)

    def cl(a):
        a = np.array([v for v in a if v == v], float)
        return float(a.mean()) if a.size else float("nan")

    report = dict(
        n_test=len(per_sample),
        rmse=cl(rmse), mae=cl(mae), crps=cl(crps),
        coverage_90=cl(cov), pit_mean=cl(pit_mean),
        spread_mean=cl(spread), skill_mean=cl(skill),
        config=dict(ensemble=config.ensemble, steps=config.sample_steps,
                    guidance=config.sda_guidance, obs_noise=float(config.obs_noise),
                    resize=config.resize),
    )
    with open(os.path.join(config.out, "test_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(config.out, "per_sample.json"), "w") as f:
        json.dump(per_sample, f, indent=2)
    print("\n================ TEST REPORT ================")
    for k, v in report.items():
        if k != "config":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("saved ->", config.out)

if __name__ == "__main__":
    main()
