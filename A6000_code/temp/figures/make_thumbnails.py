# -*- coding: utf-8 -*-
"""Real Xiamen sample thumbnails: 4 input channels + 3 SDA output maps."""
import numpy as np, torch, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys; sys.path.insert(0, '.')
from types import SimpleNamespace
from models.edm_da import EDMDataAssimilation

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"]})

d = np.load("temp/figures/xiamen_sample.npz")
X = d["X"]; y = d["y"]; yg = d["yg"]   # X [12,6,256,256]
frame = -1

# ---- input thumbnails ----
def save(ax_data, fname, cmap, vmax=None, title=""):
    fig, ax = plt.subplots(figsize=(1.8,1.8))
    ax.imshow(ax_data, cmap=cmap, vmin=-vmax if vmax else None, vmax=vmax, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("0.4"); s.set_linewidth(1)
    if title: ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"temp/figures/thumb_{fname}.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# sparse in-situ: scatter of observed pixels
sp = X[frame,0]
obs_y, obs_x = np.where(sp != 0)
fig, ax = plt.subplots(figsize=(1.8,1.8))
ax.imshow(np.zeros_like(sp), cmap="binary", vmin=0, vmax=1, origin="lower")
ax.scatter(obs_x, obs_y, c="red", s=4)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color("0.4")
fig.tight_layout(pad=0.2); fig.savefig("temp/figures/thumb_sparse.png", dpi=200, bbox_inches="tight", pad_inches=0.05); plt.close(fig)

save(X[frame,2], "era5_msl", "RdBu_r", title="ERA5 msl")
# wind: u10/v10 quiver
fig, ax = plt.subplots(figsize=(1.8,1.8))
u, v = X[frame,3][::16,::16], X[frame,4][::16,::16]
ax.quiver(u, v, scale=20, color="steelblue", width=0.004)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color("0.4")
fig.tight_layout(pad=0.2); fig.savefig("temp/figures/thumb_wind.png", dpi=200, bbox_inches="tight", pad_inches=0.05); plt.close(fig)

save(X[frame,5], "gtsm", "jet", vmax=np.nanpercentile(np.abs(X[frame,5]),98), title="GTSM surge")

# ---- SDA output maps (128 checkpoint sampling of Xiamen) ----
conf = json.load(open("results_sda/real_era5_128/conf.json"))
cfg = SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
m = EDMDataAssimilation(cfg).to("mps")
chk = torch.load("results_sda/real_era5_128/best_sda.pth.tar", map_location="mps")
sd = chk.get("ema") or chk.get("model") or chk["state_dict"]; m.load_state_dict(sd); m.eval()

Xr = F.interpolate(torch.from_numpy(X).float().reshape(1,72,256,256), size=(128,128), mode="bilinear").reshape(12,6,128,128).unsqueeze(0)
mask = torch.zeros(1,1,128,128); mask[0,0,52,64]=1.0
y_obs = torch.full((1,1,128,128), float("nan")); y_obs[0,0,52,64] = float(y[0,104,129])
out = m.sample_posterior(Xr.to("mps"), torch.tensor([8.0]).to("mps"), y=y_obs.to("mps"), mask=mask.to("mps"),
                         R=0.1, steps=25, guidance=1.0, ensemble=8, sigma_max=1.0, seed=0,
                         like_mode="replace", sampler="ode")
mean = out["mean"].cpu()[0,0].numpy(); q05 = out["q05"].cpu()[0,0].numpy(); q95 = out["q95"].cpu()[0,0].numpy()
samples = out["samples"].cpu()[0,:,0].numpy(); h = float(np.nanpercentile(np.abs(samples),90)); exceed = (samples>h).mean(axis=0)

save(mean, "out_mean", "RdBu_r", vmax=np.nanpercentile(np.abs(mean),98), title="posterior mean")
save(q95-q05, "out_interval", "viridis", title="90% interval")
save(exceed, "out_exceed", "magma", vmax=1.0, title="P(surge>h)")
print("缩略图生成完成")
