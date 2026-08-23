# -*- coding: utf-8 -*-
"""Fig.4 regional skill on a geographic map: per-gauge RMSE of 45 held-out
stations (256x256, normalised surge units). Uses cartopy coastlines with a
global_land_mask fallback."""
import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "axes.linewidth": 0.8})

val = np.load("cache_sda_full/val.npz")
lon, lat = val["lon"], val["lat"]
ps = json.load(open("results_sda/test_real_era5_256/per_sample.json"))
STD, MEAN = 0.16917373301090485, -0.0004041124621210444
rmse = np.array([p["rmse"] for p in ps]) * STD + MEAN   # metres
vmax = float(np.nanpercentile(rmse, 95))

USE_CARTOPY = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    USE_CARTOPY = True
except Exception as e:
    print("cartopy unavailable, fallback:", str(e)[:60])

if USE_CARTOPY:
    fig = plt.figure(figsize=(7.2, 3.8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#e7e7e7", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#d7e9f5", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555555")
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color="gray", alpha=0.35, linestyle="--", linewidth=0.4)
    gl.top_labels = gl.bottom_labels = gl.left_labels = gl.right_labels = False
    sc = ax.scatter(lon, lat, c=rmse, cmap="RdBu_r", s=26, edgecolor="k", linewidth=0.3,
                    vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), zorder=5)
else:
    from global_land_mask import globe
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    N = 600
    glon = np.linspace(-180, 180, N); glat = np.linspace(-60, 75, int(N*0.75))
    GLON, GLAT = np.meshgrid(glon, glat)
    land = globe.is_land(GLAT, GLON)
    ax.imshow(np.where(land, 1, np.nan), extent=[-180, 180, -60, 75], origin="lower",
              cmap="Greys", vmin=0, vmax=1, alpha=0.55, zorder=0)
    sc = ax.scatter(lon, lat, c=rmse, cmap="RdBu_r", s=26, edgecolor="k", linewidth=0.3,
                    vmin=0, vmax=vmax, zorder=5)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 75)
    ax.grid(alpha=0.3, lw=0.4)

cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, extend="both", shrink=0.7)
cb.set_label("Per-gauge RMSE (m)", fontsize=9)
cb.ax.tick_params(labelsize=8)

i_b = np.nanargmin(rmse); i_w = np.nanargmax(rmse)
ax.annotate(f"best  {rmse[i_b]:.3f}", (lon[i_b], lat[i_b]), fontsize=7, color="#0a6e2f",
            xytext=(6, 6), textcoords="offset points", fontweight="bold")
ax.annotate(f"worst  {rmse[i_w]:.3f}", (lon[i_w], lat[i_w]), fontsize=7, color="#b00020",
            xytext=(6, 6), textcoords="offset points", fontweight="bold")

fig.savefig("temp/figures/fig_regional.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("temp/figures/fig_regional.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print(f"map: cartopy={USE_CARTOPY} | RMSE {np.nanmin(rmse):.3f}~{np.nanmax(rmse):.3f} mean {np.nanmean(rmse):.3f} | vmax {vmax:.3f}")
print("saved fig_regional.{pdf,png}")
