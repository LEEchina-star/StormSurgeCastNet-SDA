# -*- coding: utf-8 -*-
"""Gulf of Mexico case (id=528): geographic map + input subplots + process diagrams."""
import numpy as np, glob, torch, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from global_land_mask import globe
import sys; sys.path.insert(0,'.')
from types import SimpleNamespace
from models.edm_da import EDMDataAssimilation

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"]})

# ---- locate sample id=528 ----
samp = None
for f in sorted(glob.glob("cache_gulf/part_*.npz")):
    d = np.load(f)
    hit = np.where(d["ids"]==528)[0]
    if len(hit):
        i = hit[0]
        X = d["X"][i]; y = d["y"][i]; yg = d["yg"][i]
        lon, lat = float(d["lon"][i]), float(d["lat"][i])
        samp = dict(X=X, y=y, yg=yg, lon=lon, lat=lat)
        # 观测像素位置
        oy, ox = np.where(~np.isnan(y[0]))
        print(f"样本 id=528: lon={lon} lat={lat} 观测值={y[0][oy,ox]} 像素=({ox},{oy})")
        break

X, y, yg = samp["X"], samp["y"], samp["yg"]
lon, lat = samp["lon"], samp["lat"]

# ===== 1) geographic map (Gulf of Mexico) =====
lon0, lon1, lat0, lat1 = -97, -80, 18, 30
N = 300
g_lon = np.linspace(lon0, lon1, N); g_lat = np.linspace(lat0, lat1, N)
GLON, GLAT = np.meshgrid(g_lon, g_lat)
land = globe.is_land(GLAT, GLON)
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.imshow(land, extent=[lon0,lon1,lat0,lat1], origin="lower", cmap="Greys", vmin=0,vmax=1, alpha=0.5)
ax.imshow(np.where(land, np.nan, 1), extent=[lon0,lon1,lat0,lat1], origin="lower", cmap="Blues", vmin=0,vmax=1, alpha=0.35)
ax.contour(g_lon, g_lat, land.astype(float), levels=[0.5], colors="k", linewidths=0.7)
# all gulf gauges
for f in sorted(glob.glob("cache_gulf/part_*.npz")):
    dd = np.load(f)
    ax.scatter(dd["lon"], dd["lat"], s=12, c="#3E6B9E", alpha=0.7, edgecolor="white", linewidth=0.3, zorder=3)
# this gauge
ax.plot(lon, lat, marker="*", color="red", ms=18, mec="white", mew=0.8, zorder=6)
d_roi = 3.2
ax.add_patch(plt.Rectangle((lon-d_roi, lat-d_roi), 2*d_roi, 2*d_roi, fill=False, ec="#1A5A9E", lw=1.8, ls="--", zorder=5))
ax.annotate("ROI 256×256 px\n(0.025°/px)", (lon-d_roi, lat+d_roi), xytext=(5,-5), textcoords="offset points",
            fontsize=7, color="#1A5A9E", va="top")
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect("equal")
ax.grid(alpha=0.2, lw=0.4)
ax.set_title("Gulf of Mexico — storm-surge gauges", fontsize=9)
fig.tight_layout(pad=0.3)
fig.savefig("temp/figures/gulf_map.png", dpi=220, bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ===== 2) input subplots (sparse / ERA5 msl / wind / GTSM) =====
fr = -1
names = ["sparse in-situ gauges", "ERA5 msl", "ERA5 wind (u10,v10)", "GTSM surge"]
fig, axes = plt.subplots(2, 2, figsize=(4.4, 4.4))
# sparse
axes[0,0].imshow(np.zeros_like(X[fr,0]), cmap="binary", vmin=0, vmax=1, origin="lower")
oy, ox = np.where(X[fr,0]!=0)
axes[0,0].scatter(ox, oy, c="red", s=4)
axes[0,0].set_title(names[0], fontsize=8); axes[0,0].set_xticks([]); axes[0,0].set_yticks([])
# msl
im = axes[0,1].imshow(X[fr,2], cmap="RdBu_r", origin="lower"); axes[0,1].set_title(names[1], fontsize=8); axes[0,1].set_xticks([]); axes[0,1].set_yticks([])
fig.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)
# wind quiver
axes[1,0].quiver(X[fr,3][::16,::16], X[fr,4][::16,::16], scale=30, color="steelblue", width=0.004)
axes[1,0].set_title(names[2], fontsize=8); axes[1,0].set_xticks([]); axes[1,0].set_yticks([])
# gtsm
im = axes[1,1].imshow(X[fr,5], cmap="jet", origin="lower"); axes[1,1].set_title(names[3], fontsize=8); axes[1,1].set_xticks([]); axes[1,1].set_yticks([])
fig.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)
fig.tight_layout(pad=0.4)
fig.savefig("temp/figures/gulf_inputs.png", dpi=220, bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ===== 3) denoising process (clean->noisy->denoised) =====
conf = json.load(open("results_sda/real_era5_128/conf.json"))
cfg = SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
m = EDMDataAssimilation(cfg).to("mps")
chk = torch.load("results_sda/real_era5_128/best_sda.pth.tar", map_location="mps")
sd = chk.get("ema") or chk.get("model") or chk["state_dict"]; m.load_state_dict(sd); m.eval()
Xr = F.interpolate(torch.from_numpy(X).float().reshape(1,72,256,256), size=(128,128), mode="bilinear").reshape(12,6,128,128).unsqueeze(0)
yg128 = F.interpolate(torch.from_numpy(yg).float()[None], size=(128,128), mode="nearest")[0,0]
x0 = torch.nan_to_num(yg128, 0.0)
sig = torch.tensor([0.8])
noise = torch.randn_like(x0)*0.8
x_t = x0 + noise
with torch.no_grad():
    xhat = m.denoise(x_t[None,None].to("mps"), sig.to("mps"), Xr.to("mps"), torch.tensor([8.0]).to("mps"))[0,0].cpu().numpy()
vm = np.nanpercentile(np.abs(x0.numpy()), 98)
fig, axes = plt.subplots(1,3, figsize=(4.4,1.6))
for ax, arr, ttl in [(axes[0], x0.numpy(), "clean x\u2080"), (axes[1], x_t.numpy(), "noisy x\u209c=x\u2080+\u03c3\u03b5"), (axes[2], xhat, "denoised x\u0302\u2080")]:
    ax.imshow(arr, cmap="RdBu_r", vmin=-vm, vmax=vm, origin="lower"); ax.set_title(ttl, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(pad=0.3)
fig.savefig("temp/figures/gulf_denoise.png", dpi=220, bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ===== 4) assimilation process (prior->obs->posterior) =====
oy, ox = np.where(~np.isnan(y[0]))
obs_px, obs_py = int(ox[0]*0.5), int(oy[0]*0.5)
mask = torch.zeros(1,1,128,128); mask[0,0,obs_py,obs_px]=1.0
y_obs = torch.full((1,1,128,128), float("nan")); y_obs[0,0,obs_py,obs_px] = float(y[0,oy[0],ox[0]])
lead = torch.tensor([8.0])
prior = m.sample_posterior(Xr.to("mps"), lead.to("mps"), y=None, mask=None, R=0.1, steps=25, guidance=0.0, ensemble=8, sigma_max=1.0, seed=0, like_mode="replace", sampler="ode")["mean"].cpu()[0,0].numpy()
post = m.sample_posterior(Xr.to("mps"), lead.to("mps"), y=y_obs.to("mps"), mask=mask.to("mps"), R=0.1, steps=25, guidance=1.0, ensemble=8, sigma_max=1.0, seed=0, like_mode="replace", sampler="ode")["mean"].cpu()[0,0].numpy()
vm2 = np.nanpercentile(np.abs(post), 98)
fig, axes = plt.subplots(1,3, figsize=(4.4,1.6))
axes[0].imshow(prior, cmap="RdBu_r", vmin=-vm2, vmax=vm2, origin="lower"); axes[0].set_title("prior (no obs)", fontsize=8); axes[0].set_xticks([]); axes[0].set_yticks([])
axes[1].imshow(prior, cmap="RdBu_r", vmin=-vm2, vmax=vm2, origin="lower"); axes[1].plot(obs_px, obs_py, "r*", ms=14, mec="white"); axes[1].set_title("observation y", fontsize=8); axes[1].set_xticks([]); axes[1].set_yticks([])
axes[2].imshow(post, cmap="RdBu_r", vmin=-vm2, vmax=vm2, origin="lower"); axes[2].set_title("posterior (assimilated)", fontsize=8); axes[2].set_xticks([]); axes[2].set_yticks([])
fig.tight_layout(pad=0.3)
fig.savefig("temp/figures/gulf_assim.png", dpi=220, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
print("Gulf 子图全部生成")
