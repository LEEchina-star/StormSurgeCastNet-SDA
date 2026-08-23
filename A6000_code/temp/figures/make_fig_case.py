# -*- coding: utf-8 -*-
"""Fig.3 case study: Xiamen storm-surge event — SDA posterior (mean / 90% interval /
exceedance probability) vs observation."""
import numpy as np, torch, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys; sys.path.insert(0, '.')
from types import SimpleNamespace
from models.edm_da import EDMDataAssimilation

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
                     "mathtext.fontset":"dejavusans"})

# load 128 checkpoint
conf = json.load(open("results_sda/real_era5_128/conf.json"))
cfg = SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
dev = "mps"
m = EDMDataAssimilation(cfg).to(dev)
chk = torch.load("results_sda/real_era5_128/best_sda.pth.tar", map_location=dev)
sd = chk.get("ema") or chk.get("model") or chk["state_dict"]
m.load_state_dict(sd); m.eval()

# load Xiamen sample (256) -> resize 128
d = np.load("temp/figures/xiamen_sample.npz")
X = torch.from_numpy(d["X"]).float()  # [12,6,256,256]
y = torch.from_numpy(d["y"]).float()  # [1,256,256]
T,C,H,W = X.shape
Xr = F.interpolate(X.reshape(T*C,H,W).reshape(1,T*C,H,W), size=(128,128), mode="bilinear").reshape(T,C,128,128).unsqueeze(0)
yr = F.interpolate(y.reshape(1,1,H,W), size=(128,128), mode="nearest")
# manual observation mask: Xiamen gauge pixel (row=104, col=129) -> (52, 64) in 128 grid
mask = torch.zeros(1,1,128,128)
obs_val = float(y[0, 104, 129])
mask[0,0, 52, 64] = 1.0
yr = torch.zeros(1,1,128,128); yr[0,0,52,64] = obs_val
y_obs = torch.where(mask>0, yr, torch.full_like(yr, float("nan")))
lead = torch.tensor([8.0])

out = m.sample_posterior(Xr.to(dev), lead.to(dev), y=y_obs.to(dev), mask=mask.to(dev), R=0.1,
                         steps=25, guidance=1.0, ensemble=8, sigma_max=1.0, seed=0,
                         like_mode="replace", sampler="ode")
mean = out["mean"].cpu()[0,0].numpy()
q05 = out["q05"].cpu()[0,0].numpy(); q95 = out["q95"].cpu()[0,0].numpy()
samples = out["samples"].cpu()[0,:,0].numpy()
h = float(np.nanpercentile(np.abs(samples), 90))  # threshold = 90th percentile of surge
exceed = (samples > h).mean(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
vm = np.nanpercentile(np.abs(mean), 98)
im0 = axes[0].imshow(mean, cmap="RdBu_r", vmin=-vm, vmax=vm, origin="lower")
axes[0].set_title("(a) Posterior mean", fontsize=9)
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
im1 = axes[1].imshow(q95-q05, cmap="viridis", origin="lower")
axes[1].set_title("(b) 90% interval width", fontsize=9)
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
im2 = axes[2].imshow(exceed, cmap="magma", vmin=0, vmax=1, origin="lower")
axes[2].set_title(f"(c) P(surge > {h:.2f})", fontsize=9)
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
for a in axes: a.set_xticks([]); a.set_yticks([])
# mark observation pixel (manual coordinate)
for a in axes: a.plot(64, 52, "r*", ms=10, mec="white", mew=0.5)
fig.suptitle("Xiamen storm-surge event (lead 8 h) — SDA posterior (N=8, γ=1.0)", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig("temp/figures/fig_case.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("temp/figures/fig_case.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
mean_val = mean[52, 64]
print(f"观测值={obs_val} | 后验均值@观测={mean_val:.3f} | 阈值 h={h:.2f}")
print("saved fig_case.{pdf,png}")
