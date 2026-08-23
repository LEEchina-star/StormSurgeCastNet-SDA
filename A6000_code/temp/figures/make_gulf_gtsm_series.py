# -*- coding: utf-8 -*-
"""Gulf case (id=528): the 12 input time-step GTSM surge fields,
noised versions, and model-denoised output. 3 figures x 12 subplots (3x4 grid)."""
import numpy as np, glob, torch, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from types import SimpleNamespace
from models.edm_da import EDMDataAssimilation

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.titlesize": 8, "axes.labelsize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": False,
})
STD_GTSM = 0.1284419447183609; MEAN_GTSM = 0.0009648051927797496
OUT = "temp/figures"

# ---- sample 528 ----
for f in sorted(glob.glob("cache_gulf/part_*.npz")):
    d = np.load(f); i = np.where(d["ids"] == 528)[0][0]
    X = d["X"][i]; break
gtsm = X[:, 5, :, :].astype(np.float32)          # [12,256,256] GTSM-normalized
print(f"GTSM input channel: {gtsm.shape}  normalized range [{gtsm.min():.2f},{gtsm.max():.2f}]")

# ---- 256 model ----
conf = json.load(open("results_sda/real_era5_256/conf.json"))
cfg = SimpleNamespace(**{k: v for k, v in conf.items() if not isinstance(v, (dict, list))})
cfg.model = "edm_da"
model = EDMDataAssimilation(cfg).to("mps"); model.eval()
chk = torch.load("results_sda/real_era5_256/best_sda.pth.tar", map_location="mps")
model.load_state_dict(chk.get("ema") or chk.get("model") or chk["state_dict"])
c = torch.from_numpy(X).float().unsqueeze(0).to("mps")        # [1,12,6,256,256]
lead = torch.tensor([8.0]).to("mps")

# ---- add noise (sigma=0.8 in normalized units) & denoise with model ----
SIGMA = 0.8
rng = np.random.default_rng(0)
noisy = np.zeros_like(gtsm); deno = np.zeros_like(gtsm)
for t in range(12):
    noisy[t] = gtsm[t] + SIGMA * rng.standard_normal(gtsm[t].shape)
    xt = torch.from_numpy(noisy[t])[None, None].to("mps")
    sig = torch.tensor([SIGMA], device=("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        deno[t] = model.denoise(xt, sig, c, lead)[0, 0].cpu().numpy()
print(f"denoised 12 frames (sigma={SIGMA})")

def to_m(a):
    return a * STD_GTSM + MEAN_GTSM

# ---- colourbar range (symmetric, 98th pct of clean field) ----
allc = to_m(gtsm)
vmax = float(np.nanpercentile(np.abs(allc), 98))
vmax = max(vmax, 1e-3)
tlabels = [f"t\u2212{11-t}h" for t in range(12)]   # t-11h ... t0 (forecast issue time)

def grid_plot(fields, title, fname):
    """3x4 grid of 12 subplots, one shared colourbar (units m) on the right."""
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.4))
    axes = axes.ravel()
    for t, ax in enumerate(axes):
        im = ax.imshow(fields[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(tlabels[t], fontsize=8, pad=1)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    # shared colourbar on the right, outside the grid (no overlap)
    fig.subplots_adjust(right=0.88, left=0.04, top=0.90, bottom=0.04, wspace=0.12, hspace=0.30)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.72])
    cb = fig.colorbar(im, cax=cbar_ax); cb.set_label("surge (m)", fontsize=9)
    fig.savefig(f"{OUT}/{fname}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig); print(f"saved {fname}")

grid_plot(to_m(gtsm), "Input GTSM surge \u2014 12 hourly frames (clean)", "gulf_gtsm_input.png")
grid_plot(to_m(noisy), "GTSM surge + Gaussian noise  ($\\sigma$=0.8)", "gulf_gtsm_noisy.png")
grid_plot(to_m(deno),  "Denoised GTSM surge \u2014 model output (256\u00d7256)", "gulf_gtsm_denoised.png")

# ---- sanity numbers ----
print(f"clean   mean|.| = {np.abs(to_m(gtsm)).mean():.4f} m  max = {np.abs(to_m(gtsm)).max():.4f} m")
print(f"noisy   RMSE vs clean = {np.sqrt(np.mean((to_m(noisy)-to_m(gtsm))**2)):.4f} m")
print(f"denoised RMSE vs clean = {np.sqrt(np.mean((to_m(deno)-to_m(gtsm))**2)):.4f} m")
print("ALL GTSM SERIES DONE")
