# -*- coding: utf-8 -*-
"""Process thumbnails with REAL surge fields:
  (1) denoising: clean -> noisy -> denoised  (EDM)
  (2) assimilation: prior -> observation -> posterior  (SDA)
"""
import numpy as np, torch, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys; sys.path.insert(0,'.')
from types import SimpleNamespace
from models.edm_da import EDMDataAssimilation

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"]})

d = np.load("temp/figures/xiamen_sample.npz")
X, yg = d["X"], d["yg"]   # X[12,6,256,256], yg[1,256,256] GTSM target

# load model
conf = json.load(open("results_sda/real_era5_128/conf.json"))
cfg = SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
m = EDMDataAssimilation(cfg).to("mps")
chk = torch.load("results_sda/real_era5_128/best_sda.pth.tar", map_location="mps")
sd = chk.get("ema") or chk.get("model") or chk["state_dict"]; m.load_state_dict(sd); m.eval()

def show(ax, a, cmap="RdBu_r", vmax=None, title=""):
    ax.imshow(a, cmap=cmap, vmin=-vmax if vmax else None, vmax=vmax, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("0.4"); s.set_linewidth(1)
    if title: ax.set_title(title, fontsize=8)

# ===== (1) denoising: clean -> noisy -> denoised =====
yg128 = F.interpolate(torch.from_numpy(yg).float()[None], size=(128,128), mode="nearest")[0,0]  # [128,128]
x0 = torch.nan_to_num(yg128, 0.0)
sig = torch.tensor([0.8])
noise = torch.randn_like(x0) * 0.8
x_t = x0 + noise
with torch.no_grad():
    xhat = m.denoise(x_t[None,None].to("mps"), sig.to("mps"), 
                     F.interpolate(torch.from_numpy(X).float().reshape(1,72,256,256), size=(128,128), mode="bilinear").reshape(12,6,128,128).unsqueeze(0).to("mps"),
                     torch.tensor([8.0]).to("mps"))
xhat = xhat[0,0].cpu().numpy()
vm = np.nanpercentile(np.abs(x0.numpy()), 98)
fig, axes = plt.subplots(1,3, figsize=(4.2,1.5))
show(axes[0], x0.numpy(), vmax=vm, title="clean x\u2080")
show(axes[1], x_t.numpy(), vmax=vm, title="noisy x\u209c = x\u2080+\u03c3\u03b5")
show(axes[2], xhat, vmax=vm, title="denoised x\u0302\u2080")
fig.tight_layout(pad=0.3)
fig.savefig("temp/figures/proc_denoise.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ===== (2) assimilation: prior -> observation -> posterior =====
Xr = F.interpolate(torch.from_numpy(X).float().reshape(1,72,256,256), size=(128,128), mode="bilinear").reshape(12,6,128,128).unsqueeze(0)
lead = torch.tensor([8.0])
mask = torch.zeros(1,1,128,128); mask[0,0,52,64]=1.0
y_obs = torch.full((1,1,128,128), float("nan")); y_obs[0,0,52,64] = float(d["y"][0,104,129])
prior = m.sample_posterior(Xr.to("mps"), lead.to("mps"), y=None, mask=None, R=0.1, steps=25, guidance=0.0,
                           ensemble=8, sigma_max=1.0, seed=0, like_mode="replace", sampler="ode")["mean"].cpu()[0,0].numpy()
post  = m.sample_posterior(Xr.to("mps"), lead.to("mps"), y=y_obs.to("mps"), mask=mask.to("mps"), R=0.1, steps=25, guidance=1.0,
                           ensemble=8, sigma_max=1.0, seed=0, like_mode="replace", sampler="ode")["mean"].cpu()[0,0].numpy()
fig, axes = plt.subplots(1,3, figsize=(4.2,1.5))
vm2 = np.nanpercentile(np.abs(post), 98)
show(axes[0], prior, vmax=vm2, title="prior (no obs)")
# observation panel: sparse point on prior field
axes[1].imshow(prior, cmap="RdBu_r", vmin=-vm2, vmax=vm2, origin="lower")
axes[1].plot(64, 52, "r*", ms=12, mec="white")
axes[1].set_xticks([]); axes[1].set_yticks([])
for s in axes[1].spines.values(): s.set_color("0.4")
axes[1].set_title("observation y", fontsize=8)
show(axes[2], post, vmax=vm2, title="posterior (assimilated)")
fig.tight_layout(pad=0.3)
fig.savefig("temp/figures/proc_assim.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("过程示意图生成完成")
