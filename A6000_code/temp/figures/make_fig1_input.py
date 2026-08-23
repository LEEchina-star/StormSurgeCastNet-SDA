# -*- coding: utf-8 -*-
"""Fig.1(a): input ROI with REAL Xiamen sample (256x256, 6 channels, T=12)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans"})

d = np.load("temp/figures/xiamen_sample.npz")
X, y, yg = d["X"], d["y"], d["yg"]   # X: [12,6,256,256]

# 取最后一帧展示各通道（风暴时刻）
frame = -1
ch_names = ["sparse (in-situ)", "valid mask", "ERA5 msl", "ERA5 u10", "ERA5 v10", "GTSM surge"]
cmaps = ["RdBu_r", "gray_r", "RdBu_r", "RdBu_r", "RdBu_r", "RdBu_r"]

fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))  # 183mm
fig.suptitle("(a) Multi-modal input c = [T=12, 6, H=W=256] — ROI centred on Xiamen (118.07°E, 24.45°N)",
             fontsize=10, y=0.99)

for i, ax in enumerate(axes.ravel()):
    data = X[frame, i]
    vmax = np.nanpercentile(np.abs(data), 98)
    im = ax.imshow(data, cmap=cmaps[i], vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_title(f"ch{i}: {ch_names[i]}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    # 厦门站观测像素位置（sparse 通道有效像素）
    if i == 0:
        yy, xx = np.where(~np.isnan(y[0]))
        for a, b in zip(xx, yy):
            ax.plot(a, b, "r*", ms=8)
    if i == 1:
        ax.set_xlabel("256 px = ~2.5 km/px", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("temp/figures/fig1_input.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("temp/figures/fig1_input.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("saved fig1_input.{pdf,png}")
print("观测像素(厦门站)坐标:", np.where(~np.isnan(y[0])))
print("目标 y 值:", y[0][~np.isnan(y[0])], "| GTSM 同像素:", yg[0][~np.isnan(yg[0])])
