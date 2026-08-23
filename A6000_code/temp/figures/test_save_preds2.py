# -*- coding: utf-8 -*-
"""EXACT mirror of test_sda_cache.py (batch=4, per-batch seed=0) but SAVES
posterior-mean fields. Run for SDA-128 / SDA-256 at resize=0, steps=25, ens=4.
Usage: python temp/figures/test_save_preds2.py --checkpoint CKPT --out OUT --resize 0
"""
import os, sys, json, time, argparse
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
from util.utils import get_device
from types import SimpleNamespace
from util.model_utils import get_model

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="results_sda/real_era5_256/best_sda.pth.tar")
ap.add_argument("--out", default="temp/figures/pred256")
ap.add_argument("--resize", type=int, default=0)
ap.add_argument("--ensemble", type=int, default=4)
ap.add_argument("--steps", type=int, default=25)
a = ap.parse_args()

device = get_device()
conf = json.load(open(os.path.join(os.path.dirname(a.checkpoint), "conf.json")))
cfg = SimpleNamespace(**{k: v for k, v in conf.items() if not isinstance(v, (dict, list))})
cfg.model = "edm_da"
model = get_model(cfg).to(device); model.eval()
chk = torch.load(a.checkpoint, map_location=device)
sd = chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk
model.load_state_dict(sd)

d = np.load("cache_sda_full/val.npz")
X = torch.from_numpy(d["X"]).float(); y = torch.from_numpy(d["y"]).float()
B, T, C, H, W = X.shape
if a.resize and H != a.resize:
    Xr = F.interpolate(X.reshape(B*T, C, H, W), size=(a.resize, a.resize), mode="bilinear").reshape(B, T, C, a.resize, a.resize)
    yr = F.interpolate(y.reshape(B, 1, H, W), size=(a.resize, a.resize), mode="nearest").reshape(B, 1, a.resize, a.resize)
else:
    Xr, yr = X, y
mask = (~torch.isnan(yr)).float()
y_obs = torch.where(mask > 0, yr, torch.full_like(yr, float("nan")))
lead = torch.full((B,), 8.0)
R = 0.1
pred = np.zeros((B, *yr.shape[1:]), np.float32)
t0 = time.time()
for i in range(0, B, 4):
    idx = slice(i, min(i+4, B))
    out = model.sample_posterior(Xr[idx].to(device), lead[idx].to(device),
                                 y=y_obs[idx].to(device), mask=mask[idx].to(device), R=R,
                                 steps=a.steps, guidance=1.0, ensemble=a.ensemble,
                                 sigma_max=1.0, sigma_min=0.002, seed=0,
                                 like_mode="replace", sampler="ode")
    pred[i:i+4] = out["mean"].cpu().numpy()
    print(f"  batch {i}-{min(i+4,B)} ({time.time()-t0:.0f}s)", flush=True)
# gauge RMSE sanity
rms = []
for bi in range(B):
    mm = mask[bi, 0].numpy() > 0
    if mm.any():
        diff = pred[bi, 0][mm] - yr[bi, 0][mm].numpy()
        rms.append(float(np.sqrt((diff**2).mean())))
print(f"gauge RMSE = {np.mean(rms):.4f}")
os.makedirs(a.out, exist_ok=True)
np.savez(f"{a.out}/pred.npz", pred=pred, resize=a.resize)
print("saved ->", a.out)
