# -*- coding: utf-8 -*-
"""Gulf-based thumbnails for the SDA-Diff framework figure (north-up, land-masked)."""
import numpy as np, glob, torch, json, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from types import SimpleNamespace
from models.edm_da import EDMDataAssimilation

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "font.family": "sans-serif"})
OUT = "temp/figures"
STD = dict(GESLA=0.16917373301090485, GTSM=0.1284419447183609,
           msl=1378.7696533203125, u10=4.459522724151611, v10=4.0246148109436035)
MEAN = dict(GESLA=-0.0004041124621210444, GTSM=0.0009648051927797496,
            msl=101002.3515625, u10=0.4772440791130066, v10=0.20318500697612762)

for f in sorted(glob.glob("cache_gulf/part_*.npz")):
    d = np.load(f); i = np.where(d["ids"] == 528)[0][0]
    X = d["X"][i]; y = d["y"][i]; yg = d["yg"][i]; break
land = np.isnan(yg[0]); fr = -1
GREY = "#D4D4D4"

def save(fig, fname, size=(1.5, 1.5)):
    fig.savefig(f"{OUT}/{fname}", dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("saved", fname)

def field(a, cmap, vmin, vmax, fname, title, units):
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    cm = plt.get_cmap(cmap).copy(); cm.set_bad(GREY)
    im = ax.imshow(a, cmap=cm, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04); cb.set_label(units, fontsize=6)
    cb.ax.tick_params(labelsize=5)
    save(fig, fname)

# ---- input thumbnails (final frame, north-up) ----
# GESLA-3 sparse
fig, ax = plt.subplots(figsize=(1.5, 1.5))
ax.imshow(np.where(land, np.nan, 1), cmap="binary", vmin=0, vmax=1, origin="upper")
oy, ox = np.where(X[fr, 0] != 0)
vals = X[fr, 0][oy, ox] * STD["GESLA"] + MEAN["GESLA"]
sc = ax.scatter(ox, oy, c=vals, cmap="RdBu_r", s=70, edgecolor="k", linewidth=0.6)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("GESLA-3 in-situ", fontsize=7)
cb = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.04); cb.set_label("m", fontsize=6); cb.ax.tick_params(labelsize=5)
save(fig, "gulf_thumb_sparse.png")
# ERA5 msl (hPa)
field(X[fr, 2] * STD["msl"] / 100 + MEAN["msl"] / 100, "RdBu_r", 995, 1015, "gulf_thumb_msl.png", "ERA5 msl", "hPa")
# ERA5 wind speed (m/s)
ws = np.sqrt((X[fr, 3] * STD["u10"] + MEAN["u10"]) ** 2 + (X[fr, 4] * STD["v10"] + MEAN["v10"]) ** 2)
field(ws, "viridis", 0, 18, "gulf_thumb_wind.png", "ERA5 wind", "m/s")
# GTSM surge (land-masked, m)
g = X[fr, 5].astype(np.float32).copy(); g[land] = np.nan
gm = g * STD["GTSM"] + MEAN["GTSM"]
vm = max(float(np.nanpercentile(np.abs(gm), 98)), 1e-3)
field(gm, "RdBu_r", -vm, vm, "gulf_thumb_gtsm.png", "coarse GTSM", "m")

# ---- output thumbnails from posterior ensemble ----
conf = json.load(open("results_sda/real_era5_256/conf.json"))
cfg = SimpleNamespace(**{k: v for k, v in conf.items() if not isinstance(v, (dict, list))})
cfg.model = "edm_da"
model = EDMDataAssimilation(cfg).to("mps"); model.eval()
chk = torch.load("results_sda/real_era5_256/best_sda.pth.tar", map_location="mps")
model.load_state_dict(chk.get("ema") or chk["state_dict"])
c = torch.from_numpy(X).float().unsqueeze(0).to("mps"); lead = torch.tensor([8.0]).to("mps")
oy2, ox2 = np.where(~np.isnan(y[0]))
mask = torch.zeros(1, 1, 256, 256); mask[0, 0, oy2, ox2] = 1.0
y_obs = torch.full((1, 1, 256, 256), float("nan")); y_obs[0, 0, oy2, ox2] = torch.from_numpy(y[0][oy2, ox2]).float()
kw = dict(R=0.1, steps=20, guidance=1.0, ensemble=8, sigma_max=1.0, seed=0, like_mode="replace", sampler="ode")
t0 = time.time()
post = model.sample_posterior(c, lead, y=y_obs.to("mps"), mask=mask.to("mps"), **kw)
print(f"posterior sampled {time.time()-t0:.0f}s, ensemble={post['samples'].shape[1]}")
mean = post["mean"].cpu()[0, 0].numpy(); q05 = post["q05"].cpu()[0, 0].numpy(); q95 = post["q95"].cpu()[0, 0].numpy()
samples = post["samples"].cpu().numpy()[0]          # [N,1,H,W]
exceed = (samples[:, 0] > 0.5 / STD["GTSM"]).mean(axis=0)   # P(surge > 0.5 m) in normalized space
mean_m = (mean * STD["GTSM"] + MEAN["GTSM"]); mean_m[land] = np.nan
span_m = ((q95 - q05) * STD["GTSM"]); span_m[land] = np.nan
exceed[land] = np.nan
vm2 = max(float(np.nanpercentile(np.abs(mean_m), 98)), 1e-3)
field(mean_m, "RdBu_r", -vm2, vm2, "gulf_thumb_mean.png", "mean", "m")
field(span_m, "viridis", 0, float(np.nanpercentile(span_m, 98)), "gulf_thumb_interval.png", "90% interval", "m")
field(exceed, "viridis", 0, 1, "gulf_thumb_exceed.png", "P(surge>0.5 m)", "")
print("ALL FRAMEWORK THUMBNAILS DONE")
