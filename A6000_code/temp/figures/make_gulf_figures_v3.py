# -*- coding: utf-8 -*-
"""Gulf of Mexico case (id=528) — final publication figures.

Corrections applied vs earlier versions:
  * ROI frame fully visible (map lat 17-33 degN)
  * tide-gauge scatter limited to ROI-internal GESLA-3 stations only
  * ERA5 wind as speed magnitude heatmap (no arrows), units m/s
  * ERA5 msl as RdBu_r, units hPa
  * GTSM surge as RdBu_r, units m
  * output = 256x256 high-resolution predicted surge (RdBu_r, m) using the 256 model
  * T=12 hourly time-series input annotated
  * per-channel denormalization (GESLA std=0.1692, GTSM std=0.1284, ERA5 exact)
"""
import numpy as np, glob, torch, json, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_land_mask import globe
sys.path.insert(0, ".")
from types import SimpleNamespace
from models.edm_da import EDMDataAssimilation

# ---- publication style (scientific-plotting skill) --------------------------
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": False,
})

# ---- exact normalization stats (Data2/aux/stats.npy) ------------------------
MEAN = dict(GESLA=-0.0004041124621210444, GTSM=0.0009648051927797496,
            msl=101002.3515625, u10=0.4772440791130066, v10=0.20318500697612762)
STD  = dict(GESLA=0.16917373301090485, GTSM=0.1284419447183609,
            msl=1378.7696533203125, u10=4.459522724151611, v10=4.0246148109436035)

def surge_m(a, which="GTSM"):   # normalized surge -> metres
    return a * STD[which] + MEAN[which]
def msl_hpa(a):                 # normalized msl -> hPa
    return (a * STD["msl"] + MEAN["msl"]) / 100.0
def wind_ms(u, v):              # normalized (u10,v10) -> wind speed m/s
    uu = u * STD["u10"] + MEAN["u10"]; vv = v * STD["v10"] + MEAN["v10"]
    return np.sqrt(uu**2 + vv**2)

OUT = "temp/figures"

# ---- locate sample id=528 ---------------------------------------------------
for f in sorted(glob.glob("cache_gulf/part_*.npz")):
    d = np.load(f); hit = np.where(d["ids"] == 528)[0]
    if len(hit):
        i = hit[0]
        X, y, yg = d["X"][i], d["y"][i], d["yg"][i]
        lon, lat = float(d["lon"][i]), float(d["lat"][i]); break
print(f"id=528: lon={lon:.3f} lat={lat:.3f}  X={X.shape}")

# ---- true ROI geometry (recovered from rasterization) ------------------------
# The 256x256 ROI is centred on an *ocean* point (sample_roi draws ~N(gauge,1)),
# NOT on the gauge itself. Matching the 5 observed target pixels to GESLA-3
# station lat/lon gives ROI centre = (-88.052, 30.052) E/N.
ROI_CX, ROI_CY = -88.052, 30.052
ROI_HALF = 128 * 0.025           # 3.2 deg
roi_w, roi_e = ROI_CX - ROI_HALF, ROI_CX + ROI_HALF
roi_s, roi_n = ROI_CY - ROI_HALF, ROI_CY + ROI_HALF
print(f"true ROI: lon [{roi_w:.2f},{roi_e:.2f}]  lat [{roi_s:.2f},{roi_n:.2f}]")

# The 5 GESLA-3 stations observed at target time inside the ROI (id + lon/lat):
#   st528 (target, red star) + st977 / st1118 / st1319 / st2641
roi_stations = [("528", -89.1400, 28.9900),   # selected station
                ("977", -87.2112, 30.4044),
                ("1118", -89.3250, 30.3250),
                ("1319", -90.6617, 29.2450),
                ("2641", -89.9468, 29.2728)]
print(f"ROI-internal GESLA-3 stations: {len(roi_stations)}")

# ============================================================================
# FIG A — geographic map (coastline + ROI gauges + selected star + ROI frame)
# ============================================================================
lon0, lon1, lat0, lat1 = -98, -79, 17, 34.5
N = 360
glon = np.linspace(lon0, lon1, N); glat = np.linspace(lat0, lat1, N)
GLON, GLAT = np.meshgrid(glon, glat); land = globe.is_land(GLAT, GLON)
fig, ax = plt.subplots(figsize=(4.6, 4.1))
ax.imshow(np.where(land, np.nan, 1), extent=[lon0, lon1, lat0, lat1],
          origin="lower", cmap="Blues", vmin=0, vmax=1, alpha=0.45, zorder=0)
ax.imshow(np.where(land, 1, np.nan), extent=[lon0, lon1, lat0, lat1],
          origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.75, zorder=1)
ax.contour(glon, glat, land.astype(float), levels=[0.5], colors="k",
           linewidths=0.6, zorder=2)
for sid, gx, gy in roi_stations:
    ax.plot(gx, gy, "o", ms=6, color="#2E6E9E", mec="white", mew=0.6, zorder=3)
ax.plot(-89.1400, 28.9900, marker="*", color="red", ms=20, mec="white", mew=0.8, zorder=6)
ax.add_patch(plt.Rectangle((roi_w, roi_s), 2 * ROI_HALF, 2 * ROI_HALF,
             fill=False, ec="#1A5A9E", lw=1.6, ls="--", zorder=5))
ax.annotate("ROI 256\u00d7256 px  (0.025\u00b0/px)", (roi_w, roi_n),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
            color="#1A5A9E", va="bottom")
ax.set_xlabel("Longitude (\u00b0E)"); ax.set_ylabel("Latitude (\u00b0N)")
ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect("equal")
ax.set_title("Gulf of Mexico \u2014 ROI GESLA-3 stations", fontsize=9)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/gulf3_map.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig); print("saved gulf3_map.png")

# ============================================================================
# FIG B — inputs (T=12 hourly frames -> final frame t=-1 shown)
# ============================================================================
fr = -1
fig, axes = plt.subplots(2, 2, figsize=(4.6, 4.6))
# (a) GESLA-3 in-situ surge (m)
axes[0, 0].imshow(np.zeros_like(X[fr, 0]), cmap="binary", vmin=0, vmax=1, origin="lower")
oy, ox = np.where(X[fr, 0] != 0)
sc = axes[0, 0].scatter(ox, oy, c=surge_m(X[fr, 0][oy, ox], "GESLA"),
                        cmap="RdBu_r", s=26, edgecolor="k", linewidth=0.35)
axes[0, 0].set_title("(a) GESLA-3 in-situ surge (m)", fontsize=8)
axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
cb = fig.colorbar(sc, ax=axes[0, 0], fraction=0.046, pad=0.04); cb.set_label("m", fontsize=8)
# (b) ERA5 msl (hPa)
im = axes[0, 1].imshow(msl_hpa(X[fr, 2]), cmap="RdBu_r", origin="lower")
axes[0, 1].set_title("(b) ERA5 mean sea-level pressure (hPa)", fontsize=8)
axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
cb = fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04); cb.set_label("hPa", fontsize=8)
# (c) ERA5 wind speed (m/s, no arrows)
im = axes[1, 0].imshow(wind_ms(X[fr, 3], X[fr, 4]), cmap="viridis", origin="lower")
axes[1, 0].set_title("(c) ERA5 wind speed (m/s)", fontsize=8)
axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
cb = fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04); cb.set_label("m/s", fontsize=8)
# (d) GTSM surge (m, RdBu_r)
im = axes[1, 1].imshow(surge_m(X[fr, 5], "GTSM"), cmap="RdBu_r", origin="lower")
axes[1, 1].set_title("(d) GTSM surge (m)", fontsize=8)
axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])
cb = fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04); cb.set_label("m", fontsize=8)
fig.suptitle("Input  c = [B, T=12, 6, H, W]  (12 hourly frames \u2192 final frame)", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
fig.savefig(f"{OUT}/gulf3_inputs.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig); print("saved gulf3_inputs.png")

# ============================================================================
# Load 256 model
# ============================================================================
conf = json.load(open("results_sda/real_era5_256/conf.json"))
cfg = SimpleNamespace(**{k: v for k, v in conf.items() if not isinstance(v, (dict, list))})
cfg.model = "edm_da"
model = EDMDataAssimilation(cfg).to("mps"); model.eval()
chk = torch.load("results_sda/real_era5_256/best_sda.pth.tar", map_location="mps")
sd = chk.get("ema") or chk.get("model") or chk["state_dict"]
model.load_state_dict(sd)
print("256 model loaded")

c = torch.from_numpy(X).float().unsqueeze(0).to("mps")        # [1,12,6,256,256]
lead = torch.tensor([8.0]).to("mps")

# ============================================================================
# FIG C — denoising: clean x0 -> noisy xt=x0+sigma*eps -> denoised x0_hat
# ============================================================================
x0_np = yg[0].astype(np.float32)                                 # dense target [256,256]
x0 = torch.nan_to_num(torch.from_numpy(x0_np), 0.0)              # [256,256]
sig = torch.tensor([0.8], device=("cuda" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(0); noise = torch.randn_like(x0) * 0.8; x_t = x0 + noise
with torch.no_grad():
    xhat = model.denoise(x_t[None, None].to("mps"), sig, c, lead)[0, 0].cpu().numpy()
x0m, xtm, xhm = surge_m(x0.numpy()), surge_m(x_t.numpy()), surge_m(xhat)
valid = ~np.isnan(x0_np)
vm = np.nanpercentile(np.abs(x0m[valid]), 98)
fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.85))
for ax, arr, ttl in [(axes[0], x0m, "(a) clean $x_0$"), (axes[1], xtm, "(b) noisy $x_t = x_0 + \\sigma\\epsilon$"),
                     (axes[2], xhm, "(c) denoised $\\hat{x}_0$")]:
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vm, vmax=vm, origin="lower")
    ax.set_title(ttl, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=axes, location="right", fraction=0.046, pad=0.03)
cb.set_label("surge (m)", fontsize=8)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/gulf3_denoise.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig); print("saved gulf3_denoise.png")

# ============================================================================
# FIG D + E — SDA assimilation: prior -> observation -> posterior (+ uncertainty)
# ============================================================================
# observation at target time: 5 GESLA-3 pixels (sparse y). The selected/target
# station is st528 at (-89.14, 28.99) -> raster pixel (ax=84, ay=170).
oy, ox = np.where(~np.isnan(y[0]))
obs_px, obs_py = 84, 170                                # st528 (target station)
obs_val_m = surge_m(float(y[0][obs_py, obs_px]), "GESLA")
print(f"target station st528: pixel ({obs_px},{obs_py}), observed surge = {obs_val_m:.3f} m")
mask = torch.zeros(1, 1, 256, 256); mask[0, 0, oy, ox] = 1.0
y_obs = torch.full((1, 1, 256, 256), float("nan"))
y_obs[0, 0, oy, ox] = torch.from_numpy(y[0][oy, ox]).float()

kw = dict(R=0.1, steps=20, guidance=1.0, ensemble=4, sigma_max=1.0,
          seed=0, like_mode="replace", sampler="ode")
t0 = time.time()
prior = model.sample_posterior(c, lead, y=None, mask=None, **kw)["mean"].cpu()[0, 0].numpy()
print(f"prior sampled {time.time()-t0:.0f}s")
t0 = time.time()
post = model.sample_posterior(c, lead, y=y_obs.to("mps"), mask=mask.to("mps"), **kw)
print(f"posterior sampled {time.time()-t0:.0f}s")
post_mean = post["mean"].cpu()[0, 0].numpy()
post_std  = post["samples"].std(dim=1).cpu()[0, 0].numpy()
post_q05  = post["q05"].cpu()[0, 0].numpy(); post_q95 = post["q95"].cpu()[0, 0].numpy()

prior_m, post_m = surge_m(prior), surge_m(post_mean)
post_std_m = post_std * STD["GTSM"]
vm2 = np.nanpercentile(np.abs(post_m), 98)

# FIG D — prior -> obs -> posterior
fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.85))
# (a) prior
im = axes[0].imshow(prior_m, cmap="RdBu_r", vmin=-vm2, vmax=vm2, origin="lower")
axes[0].set_title("(a) prior  $p(x|c)$  (no obs)", fontsize=8)
axes[0].set_xticks([]); axes[0].set_yticks([])
# (b) sparse GESLA-3 observation (5 stations, coloured by surge)
axes[1].imshow(np.zeros_like(y[0]), cmap="binary", vmin=0, vmax=1, origin="lower")
obs_vals_m = surge_m(y[0][oy, ox], "GESLA")
sc = axes[1].scatter(ox, oy, c=obs_vals_m, cmap="RdBu_r", s=40,
                     edgecolor="k", linewidth=0.4)
axes[1].set_title("(b) observation $y$  (GESLA-3)", fontsize=8)
axes[1].set_xticks([]); axes[1].set_yticks([])
# (c) posterior
im = axes[2].imshow(post_m, cmap="RdBu_r", vmin=-vm2, vmax=vm2, origin="lower")
axes[2].set_title("(c) posterior  $p(x|c,y)$", fontsize=8)
axes[2].set_xticks([]); axes[2].set_yticks([])
# red star on target station (st528) in both obs & posterior panels
for ax in (axes[1], axes[2]):
    ax.plot(obs_px, obs_py, "r*", ms=14, mec="white", mew=0.6)
axes[1].annotate(f"st528  {obs_val_m:.2f} m", (obs_px, obs_py), xytext=(5, 5),
                 textcoords="offset points", color="red", fontsize=7)
cb = fig.colorbar(im, ax=axes, location="right", fraction=0.046, pad=0.03)
cb.set_label("surge (m)", fontsize=8)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/gulf3_assim.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig); print("saved gulf3_assim.png")

# FIG E — final 256x256 output: posterior mean + ensemble uncertainty (spread)
fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4))
im = axes[0].imshow(post_m, cmap="RdBu_r", origin="lower")
axes[0].set_title("(a) predicted surge 256\u00d7256 (m)", fontsize=8)
axes[0].set_xticks([]); axes[0].set_yticks([])
cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04); cb.set_label("m", fontsize=8)
im2 = axes[1].imshow(post_std_m, cmap="viridis", origin="lower")
axes[1].set_title("(b) ensemble spread $\\sigma$ (m)", fontsize=8)
axes[1].set_xticks([]); axes[1].set_yticks([])
cb = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04); cb.set_label("m", fontsize=8)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/gulf3_output.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig); print("saved gulf3_output.png")

print(f"\ntarget st528 observed surge = {obs_val_m:.3f} m  |  max ROI obs = {surge_m(np.nanmax(y[0]), 'GESLA'):.3f} m")
print(f"prior mean |max| = {np.nanmax(np.abs(prior_m)):.3f} m, posterior |max| = {np.nanmax(np.abs(post_m)):.3f} m")
print("ALL GULF v3 FIGURES DONE")
