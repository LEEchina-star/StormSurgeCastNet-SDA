# -*- coding: utf-8 -*-
"""Geographic map: Fujian coast (Xiamen), land-sea mask + gauge + ROI box."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_land_mask import globe

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"]})

# region: Fujian coast
lon0, lon1, lat0, lat1 = 116.0, 122.0, 21.5, 28.0
N = 220
lon = np.linspace(lon0, lon1, N)
lat = np.linspace(lat0, lat1, N)
LON, LAT = np.meshgrid(lon, lat)
land = globe.is_land(LAT, LON)   # True = land

fig, ax = plt.subplots(figsize=(3.6, 4.0))
# ocean light blue, land light grey
ocean = np.where(land, np.nan, 1.0)
ax.imshow(land, extent=[lon0,lon1,lat0,lat1], origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.55)
ax.imshow(ocean, extent=[lon0,lon1,lat0,lat1], origin="lower", cmap="Blues", vmin=0, vmax=1, alpha=0.4)
# coast outline
ax.contour(lon, lat, land.astype(float), levels=[0.5], colors="k", linewidths=0.8)
# Xiamen gauge
ax.plot(118.067, 24.45, marker="*", color="red", ms=16, mec="white", mew=0.6, zorder=5)
ax.annotate("Xiamen", (118.067, 24.45), xytext=(6,6), textcoords="offset points", fontsize=9, color="red", fontweight="bold")
# ROI box (6.4 deg square centred on Xiamen, 256 px * 0.025 deg)
d = 3.2
ax.add_patch(plt.Rectangle((118.067-d, 24.45-d), 2*d, 2*d, fill=False, ec="#1A5A9E", lw=2, ls="--", zorder=4))
ax.annotate("ROI\n256×256 px\n(0.025°/px)", (118.067-d, 24.45+d), xytext=(4,-4), textcoords="offset points",
            fontsize=7, color="#1A5A9E", va="top", ha="left")
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
ax.set_aspect("equal")
ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.3)
fig.savefig("temp/figures/geo_fujian.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("geo_fujian.png 生成")
