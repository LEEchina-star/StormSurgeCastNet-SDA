# -*- coding: utf-8 -*-
"""Gulf case (id=528) v4: land-masked GTSM + corrected time labels + fixed assimilation.

Fixes:
  1. GTSM valid mask removes LAND (storm surge is ocean/coastal, cannot be on land).
  2. Time labels t-0h ... t-11h (hours BEFORE forecast), top-left = t-0h (latest).
  3. Assimilation figure: observation = dense storm-surge distribution (GTSM target),
     NOT scattered noise points.
Deliverables: input 12 (clean), noisy 12, assimilation 3-panel (prior->obs->posterior).
"""
import numpy as np, glob, torch, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_land_mask import globe
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
STD_GESLA = 0.16917373301090485; MEAN_GESLA = -0.0004041124621210444
OUT = "temp/figures"; SIGMA = 1.0
ROI_CX, ROI_CY = -88.052, 30.052     # true ROI centre (ocean point)
RES, SIZE = 0.025, 256

def m_gtsm(a): return np.asarray(a, np.float32) * STD_GTSM + MEAN_GTSM
def m_gesla(a): return np.asarray(a, np.float32) * STD_GESLA + MEAN_GESLA

def land_mask(cx, cy, res, size):
    west = cx - size/2*res; east = cx + size/2*res
    south = cy - size/2*res; north = cy + size/2*res
    lon = west + (np.arange(size)+0.5)*res          # column -> lon
    lat = north - (np.arange(size)+0.5)*res         # row 0 = north (rasterio)
    LON, LAT = np.meshgrid(lon, lat)
    return globe.is_land(LAT, LON)                  # [size,size] bool

# ---- sample 528 ----
for f in sorted(glob.glob("cache_gulf/part_*.npz")):
    d = np.load(f); i = np.where(d["ids"] == 528)[0][0]
    X = d["X"][i]; y = d["y"][i]; yg = d["yg"][i]; break
gtsm = X[:, 5, :, :].astype(np.float32)             # [12,256,256] coarse GTSM
# land mask = the data's own valid mask (yg is NaN on land, from get_lsm).
# Using this (NOT globe) keeps coastal tide-gauge pixels as ocean, so the
# assimilation effect at the gauge locations stays visible.
land = np.isnan(yg[0])
print(f"land fraction in ROI: {land.mean():.3f}")

# ---- (a) clean (land-masked) & (b) noisy (land-masked) ----
clean = gtsm.copy(); clean[:, land] = np.nan
rng = np.random.default_rng(0)
noisy = clean + SIGMA * rng.standard_normal(clean.shape).astype(np.float32)  # land stays NaN

# ---- model + prior/posterior ----
conf = json.load(open("results_sda/real_era5_256/conf.json"))
cfg = SimpleNamespace(**{k: v for k, v in conf.items() if not isinstance(v, (dict, list))})
cfg.model = "edm_da"
model = EDMDataAssimilation(cfg).to("mps"); model.eval()
chk = torch.load("results_sda/real_era5_256/best_sda.pth.tar", map_location="mps")
model.load_state_dict(chk.get("ema") or chk["state_dict"])
c = torch.from_numpy(X).float().unsqueeze(0).to("mps")
lead = torch.tensor([8.0]).to("mps")

oy, ox = np.where(~np.isnan(y[0]))
mask = torch.zeros(1, 1, 256, 256); mask[0, 0, oy, ox] = 1.0
y_obs = torch.full((1, 1, 256, 256), float("nan")); y_obs[0, 0, oy, ox] = torch.from_numpy(y[0][oy, ox]).float()
kw = dict(R=0.1, steps=20, guidance=1.0, ensemble=4, sigma_max=1.0,
          seed=0, like_mode="replace", sampler="ode")
prior = model.sample_posterior(c, lead, y=None, mask=None, **kw)["mean"].cpu()[0, 0].numpy()
post  = model.sample_posterior(c, lead, y=y_obs.to("mps"), mask=mask.to("mps"), **kw)["mean"].cpu()[0, 0].numpy()

prior_m = m_gtsm(prior); prior_m[land] = np.nan
post_m  = m_gtsm(post);  post_m[land]  = np.nan
obs_m   = m_gtsm(yg[0]); obs_m[land] = np.nan       # dense GTSM target = storm-surge distribution

# ---- colourbar range (from observation) ----
vmax = max(float(np.nanpercentile(np.abs(obs_m), 98)), 1e-3)

# ---- time labels: t-0h ... t-11h, top-left = t-0h (latest, frame 11) ----
GREY = "#D4D4D4"                                   # land colour (matches panel (b) background)
def grid_plot(fields, title, fname):
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(GREY)   # land (NaN) -> grey, not white
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.4))
    axes = axes.ravel()
    for p, ax in enumerate(axes):
        t = 11 - p                                   # frame 11 (t-0h) first -> frame 0 (t-11h) last
        # origin="upper": array row 0 = north (land) drawn at TOP (geographically correct)
        im = ax.imshow(m_gtsm(fields[t]), cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper")
        ax.set_title(f"t-{p}h", fontsize=8, pad=1)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(right=0.88, left=0.04, top=0.90, bottom=0.04,
                        wspace=0.12, hspace=0.30)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.72])
    cb = fig.colorbar(im, cax=cbar_ax); cb.set_label("surge (m)", fontsize=9)
    fig.savefig(f"{OUT}/{fname}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig); print(f"saved {fname}")

grid_plot(clean, "Input GTSM surge (land-masked) \u2014 12 hourly frames", "gulf_gtsm_input.png")
grid_plot(noisy, "GTSM surge + Gaussian noise ($\\sigma$=1.0, land-masked)", "gulf_gtsm_noisy.png")

# ---- assimilation 3-panel: prior -> observation (GESLA-3 in ROI) -> posterior ----
oy, ox = np.where(~np.isnan(y[0]))
gesla_vals = m_gesla(y[0][oy, ox])              # physical surge at the gauges (m)
sel_px, sel_py = 84, 170                         # st528 (selected station)
vmax_a = max(np.nanmax(np.abs(prior_m)), np.nanmax(np.abs(post_m)),
             np.nanmax(np.abs(gesla_vals)), 1e-3)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(GREY)      # land -> grey
# (a) prior
im = axes[0].imshow(prior_m, cmap=cmap, vmin=-vmax_a, vmax=vmax_a, origin="upper")
axes[0].set_title("(a) prior  $p(x|c)$", fontsize=9)
axes[0].set_xticks([]); axes[0].set_yticks([])
# (b) observation: GESLA-3 gauges inside the ROI on land/ocean background
bg = np.zeros((256, 256, 3), np.float32)
bg[land] = [0.83, 0.83, 0.83]                    # grey land
bg[~land] = [0.78, 0.90, 0.98]                   # light blue ocean
axes[1].imshow(bg, origin="upper")
sc = axes[1].scatter(ox, oy, c=gesla_vals, cmap="RdBu_r",
                     vmin=-vmax_a, vmax=vmax_a, s=140,
                     edgecolor="k", linewidth=1.0, zorder=5)
axes[1].plot(sel_px, sel_py, "r*", ms=22, mec="white", mew=0.8, zorder=6)
for (a, b), v in zip(zip(oy, ox), gesla_vals):
    axes[1].annotate(f"{v:.2f}", (b, a), xytext=(5, 5), textcoords="offset points",
                     fontsize=7, color="k", zorder=7)
axes[1].set_title("(b) observation $y$ (GESLA-3 in ROI)", fontsize=9)
axes[1].set_xticks([]); axes[1].set_yticks([])
# (c) posterior
im = axes[2].imshow(post_m, cmap=cmap, vmin=-vmax_a, vmax=vmax_a, origin="upper")
axes[2].set_title("(c) posterior  $p(x|c,y)$", fontsize=9)
axes[2].set_xticks([]); axes[2].set_yticks([])
fig.subplots_adjust(right=0.90, left=0.03, top=0.88, bottom=0.05, wspace=0.12)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.65])
cb = fig.colorbar(sc, cax=cbar_ax); cb.set_label("surge (m)", fontsize=9)
fig.suptitle("SDA data assimilation \u2014 prior \u2192 GESLA-3 observation \u2192 posterior (m)", fontsize=10)
fig.savefig(f"{OUT}/gulf_assim.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig); print("saved gulf_assim.png")

print(f"\nprior |max| = {np.nanmax(np.abs(prior_m)):.3f} m, "
      f"obs |max| = {np.nanmax(np.abs(obs_m)):.3f} m, "
      f"posterior |max| = {np.nanmax(np.abs(post_m)):.3f} m")
print("ALL GULF v4 DONE")
