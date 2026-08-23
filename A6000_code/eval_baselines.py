# -*- coding: utf-8 -*-
"""E1: statistical baselines (Ebel Table 1 hyperlocal) + NNSE, on the 45 held-out
gauges, in METRES. Baselines: input average, input extrapolation, GTSM extrapolation,
gauge climatology (proxy for seasonal average). Compares vs SDA-256/U-TAE preds."""
import numpy as np, os, sys
sys.path.insert(0, os.getcwd())
STD, MEAN = 0.16917373301090485, -0.0004041124621210444

d = np.load("cache_sda_full/val.npz")
X, y, yg, lead = d["X"], d["y"], d["yg"], d["lead"]
B, T, C, H, W = X.shape

def gauge_pred(fn):
    """fn(sample_i, t) -> gauge-pixel pred per sample; returns (pred, obs) arrays."""
    pred, obs = [], []
    for i in range(B):
        oy, ox = np.where(~np.isnan(y[i, 0]))
        if len(oy) == 0: continue
        a, b = oy[0], ox[0]
        pred.append(fn(i, a, b)); obs.append(float(y[i, 0, a, b]))
    return np.array(pred), np.array(obs)

def m_to_m_norm(v):
    return v  # values already normalized; convert below

# ---- baselines (normalized units, then to metres) ----
def inp_avg(i, a, b):   return float(np.nanmean(X[i, :, 0, a, b]))
def inp_extr(i, a, b):  # linear fit over T frames, extrapolate by lead
    ts = X[i, :, 0, a, b].astype(float)
    tt = np.arange(T)
    ok = ~np.isnan(ts)
    if ok.sum() < 2: return float(np.nanmean(ts))
    k = np.polyfit(tt[ok], ts[ok], 1)
    t_tgt = T - 1 + float(lead[i])          # target at last input frame + lead hours
    return float(k[0] * t_tgt + k[1])
def gtsm(i, a, b):      return float(yg[i, 0, a, b])

res = {}
for name, fn in [("input average", inp_avg), ("input extrapolation", inp_extr), ("GTSM extrapolation", gtsm)]:
    pr, ob = gauge_pred(fn)
    res[name] = (pr, ob)

def metrics(pr, ob):
    pm, om = pr * STD + MEAN, ob * STD + MEAN
    mae = float(np.mean(np.abs(pm - om)))
    mse = float(np.mean((pm - om) ** 2))
    nnse = 1.0 / (2.0 - (1.0 - mse / max(np.var(om), 1e-9)))
    return mae, mse, nnse

print("=" * 74)
print(f"{'model':<22}{'MAE (m)':>10}{'MSE (m2)':>10}{'NNSE':>8}")
print("-" * 74)
for name, (pr, ob) in res.items():
    mae, mse, nnse = metrics(pr, ob)
    print(f"{name:<22}{mae:>10.3f}{mse:>10.4f}{nnse:>8.3f}")

# ---- SDA-256 & U-TAE from saved preds ----
fd = np.load("temp/figures/preds_final.npz")
for nm, pk in [("SDA-Diff 256", "p256_g"), ("SDA-Diff 128", "p128_g"), ("FiLM U-TAE", "put_g")]:
    mae, mse, nnse = metrics(fd[pk], fd["obs_g"])
    print(f"{nm:<22}{mae:>10.3f}{mse:>10.4f}{nnse:>8.3f}")
print("=" * 74)
print("done. (seasonal average needs dates -> proxy = gauge climatology, to add)")
