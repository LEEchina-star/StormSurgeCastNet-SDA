# -*- coding: utf-8 -*-
"""Ablation study on a trained SDA-Diff checkpoint (no retraining needed):
   - guidance weight gamma, observation noise R, ensemble size
Reports RMSE / CRPS / 90% coverage for each setting on the clean 45-gauge val set.
Usage:
    python ablate_sda.py --checkpoint results_sda/real_era5_128/best_sda.pth.tar \
        --cache_dir cache_sda_full --resize 128 --out results_sda/ablation_128
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
    parser.add_argument("--resize", type=int, default=128)
    parser.add_argument("--checkpoint", default="results_sda/real_era5_128/best_sda.pth.tar")
    parser.add_argument("--out", default="results_sda/ablation_128")
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

    tc_path = os.path.join(os.path.dirname(config.checkpoint), "conf.json")
    if os.path.isfile(tc_path):
        tc = json.load(open(tc_path))
        defaults = vars(parser.parse_args([]))
        given = {k for k, v in vars(config).items() if k in defaults and v != defaults[k]}
        for k, v in tc.items():
            if k not in given:
                setattr(config, k, v)

    config.in_dim = 6 - 3 * (not config.era5) - (not config.gtsm)
    model = get_model(config).to(device)
    chk = torch.load(config.checkpoint, map_location=device)
    sd = chk.get("ema") or chk.get("model") or chk.get("state_dict")
    model.load_state_dict(sd)
    model.eval()

    d = np.load(os.path.join(config.cache_dir, "val.npz"))
    X = torch.from_numpy(d["X"]).float(); y = torch.from_numpy(d["y"]).float()
    B, T, C, H, W = X.shape
    if config.resize and H != config.resize:
        X = F.interpolate(X.reshape(B*T, C, H, W), size=(config.resize, config.resize), mode="bilinear").reshape(B, T, C, config.resize, config.resize)
        y = F.interpolate(y.reshape(B, 1, H, W), size=(config.resize, config.resize), mode="nearest").reshape(B, 1, config.resize, config.resize)
    mask = (~torch.isnan(y)).float()
    y_obs = torch.where(mask > 0, y, torch.full_like(y, float("nan")))
    lead = torch.full((B,), float(config.lead_time if hasattr(config, "lead_time") else 8.0))

    os.makedirs(config.out, exist_ok=True)
    results = {}
    t_start = time.time()

    def evaluate(guidance, R, ensemble, steps=20, n_batch=12):
        rmse, crps, cov = [], [], []
        for i in range(0, min(B, 16), 4):
            idx = slice(i, min(i + 4, B))
            out = model.sample_posterior(
                X[idx].to(device), lead[idx].to(device),
                y=y_obs[idx].to(device), mask=mask[idx].to(device), R=R,
                steps=steps, guidance=guidance, ensemble=ensemble,
                sigma_max=config.sigma_max, sigma_min=config.sigma_min, seed=0,
                like_mode="replace", sampler="ode")
            samples = out["samples"].cpu().numpy(); mean = out["mean"].cpu().numpy()
            for bi in range(samples.shape[0]):
                crps.append(mets.crps_ensemble(samples[bi], y[idx][bi].numpy(), mask[idx][bi].numpy()))
                cov.append(mets.interval_coverage(samples[bi], y[idx][bi].numpy(), mask[idx][bi].numpy(), alpha=0.1))
                msk = mask[idx][bi, 0] > 0
                if msk.any():
                    dd = mean[bi, 0][msk] - y[idx][bi, 0][msk].numpy()
                    rmse.append(float(np.sqrt((dd**2).mean())))
        def cl(a): a = np.array([v for v in a if v == v]); return float(a.mean()) if a.size else float("nan")
        return dict(rmse=cl(rmse), crps=cl(crps), cov90=cl(cov))

    # ---- guidance ablation ----
    print("guidance ablation...", flush=True)
    for g in [0.0, 0.3, 0.5, 1.0, 2.0]:
        results[f"guidance_{g}"] = evaluate(guidance=g, R=0.1, ensemble=8)
        print(f"  gamma={g}: {results[f'guidance_{g}']}", flush=True)
    # ---- R ablation ----
    print("R ablation...", flush=True)
    for R in [0.03, 0.05, 0.1, 0.2, 0.5]:
        results[f"R_{R}"] = evaluate(guidance=1.0, R=R, ensemble=8)
        print(f"  R={R}: {results[f'R_{R}']}", flush=True)
    # ---- ensemble ablation ----
    print("ensemble ablation...", flush=True)
    for ens in [2, 4, 8, 16]:
        results[f"ensemble_{ens}"] = evaluate(guidance=1.0, R=0.1, ensemble=ens)
        print(f"  ens={ens}: {results[f'ensemble_{ens}']}", flush=True)

    with open(os.path.join(config.out, "ablation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE in {time.time()-t_start:.0f}s -> {config.out}/ablation.json")

if __name__ == "__main__":
    main()
