# -*- coding: utf-8 -*-
"""
Framework figure v2 - Nature/Science style.
SDA (Score-based Data Assimilation) + conditional diffusion for dense,
global storm surge forecasting with sparse in-situ assimilation.

Design language (inspired by GenCast/Nature framework figures):
  * three phase-band containers with light fills + dashed borders
  * iconographic data thumbnails (mini map with gauges, grids, mask)
  * U-shaped UNet icon for the conditional denoiser
  * minimal text, one accent colour for the SDA highlight
  * formula chips in white boxes with coloured left borders

Output: 183 mm double column, vector PDF + 600 dpi PNG (NC compliant).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                Circle, Wedge)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
})

INK    = "#2E3440"
LIGHT  = "#7A8699"
B1, E1 = "#F5F9FD", "#CFDCE9"
B2, E2 = "#F4FAF6", "#CFE3D6"
B3, E3 = "#FDF6F3", "#EFD9CF"
C_IN,  EN_IN  = "#E2ECF7", "#4E7FB5"
C_CTX, EN_CTX = "#EDF0F4", "#7E8CA0"
C_GEN, EN_GEN = "#DFEDE4", "#4E8A5F"
C_DA,  EN_DA  = "#FBE7E0", "#D25B43"
C_OUT, EN_OUT = "#FDF2DC", "#D1923F"
C_WHITE       = "#FFFFFF"

W_IN, H_IN = 7.205, 4.05
fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def band(x, y, w, h, fc, ec, label, label_col):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                       fc=fc, ec=ec, lw=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.add_patch(p)
    ax.text(x + 0.012, y + h - 0.013, label, fontsize=7.4, color=label_col,
            ha="left", va="top", zorder=3, fontweight="bold")

def chip(x, y, w, h, fc, ec, text, fs=7.6, weight="normal", tc=INK, ls="-",
         lw=1.0, z=5, round=0.012):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={round}",
                       fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, color=tc, ha="center",
            va="center", zorder=z + 1, fontweight=weight)

def arrow(x1, y1, x2, y2, color="#5A6B7E", lw=1.4, style="-|>", ls="-",
          ms=11, z=4, rad=0.0):
    kw = dict(arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
              linestyle=ls, zorder=z, shrinkA=2, shrinkB=2)
    if rad:
        kw["connectionstyle"] = f"arc3,rad={rad}"
    a = FancyArrowPatch((x1, y1), (x2, y2), **kw)
    ax.add_patch(a)

def formula_box(x, y, w, h, fc, ec, text, note=None, fs=7.8, note_fs=6.4,
                tc=INK, nt="#7A8699", lw=1.2, z=5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    ax.add_patch(Rectangle((x - 0.002, y + 0.01), 0.007, h - 0.02, fc=ec,
                           ec="none", zorder=z + 1))
    ty = y + h - 0.028 if note else y + h / 2
    ax.text(x + w / 2, ty, text, fontsize=fs, color=tc, ha="center", va="center",
            zorder=z + 1)
    if note:
        ax.text(x + w / 2, y + 0.030, note, fontsize=note_fs, color=nt,
                ha="center", va="center", zorder=z + 1, style="italic")

def thumb_gauges(x, y, w, h, seed=7):
    ax.add_patch(Rectangle((x, y), w, h, fc="#C9E0F2", ec="#8FB4D6", lw=0.8, zorder=2))
    for cx, cy, r in [(x+0.30*w, y+0.35*h, 0.20*w), (x+0.68*w, y+0.60*h, 0.16*w),
                      (x+0.50*w, y+0.16*h, 0.11*w), (x+0.15*w, y+0.62*h, 0.13*w)]:
        ax.add_patch(Circle((cx, cy), r, fc="#AFC2D2", ec="none", zorder=3))
    rng = np.random.default_rng(seed)
    pts = rng.uniform([x, y], [x + w, y + h], size=(16, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=7, c="#16385F", zorder=5, marker="o",
               linewidths=0)
    ax.text(x + w/2, y + h + 0.012, "in-situ GESLA (sparse)", fontsize=6.6,
            color=INK, ha="center", va="bottom")

def thumb_field(x, y, w, h, cmap, n=7, seed=1, label=""):
    rng = np.random.default_rng(seed)
    z = rng.random((n, n))
    try:
        from scipy.ndimage import gaussian_filter
        z = gaussian_filter(z, sigma=0.9)
    except Exception:
        pass
    ax.imshow(z, extent=(x, x + w, y, y + h), cmap=cmap, aspect="auto",
              interpolation="bilinear", zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#8FA6BC", lw=0.8, zorder=3))
    ax.text(x + w/2, y + h + 0.012, label, fontsize=6.6, color=INK,
            ha="center", va="bottom")

def thumb_mask(x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fc="#C3CBD6", ec="#8FA0B2", lw=0.8, zorder=2))
    for gx, gy in [(0.30, 0.55), (0.62, 0.30)]:
        ax.add_patch(Rectangle((x + gx*w, y + gy*h), 0.16*w, 0.22*h, fc="white",
                               ec="#8FA0B2", lw=0.5, zorder=3))
    ax.text(x + w/2, y + h + 0.012, "valid mask", fontsize=6.6, color=INK,
            ha="center", va="bottom")

def unet_icon(cx, cy, w, h, col, edge):
    bw, bh = 0.135*w, 0.115*h
    x0 = cx - w/2
    y0 = cy - h/2
    blocks = {
        "enc1": (x0,           y0 + 0.62*h),
        "enc2": (x0,           y0 + 0.28*h),
        "bot":  (x0 + 0.435*w, y0 + 0.28*h),
        "dec2": (x0 + 0.87*w,  y0 + 0.28*h),
        "dec1": (x0 + 0.87*w,  y0 + 0.62*h),
    }
    for k, (bx, by) in blocks.items():
        p = FancyBboxPatch((bx, by), bw, bh,
                           boxstyle="round,pad=0.004,rounding_size=0.05",
                           fc=col, ec=edge, lw=1.0, zorder=6)
        ax.add_patch(p)
        lab = {"enc1": "enc", "enc2": "enc", "bot": "bottleneck",
               "dec2": "dec", "dec1": "dec"}[k]
        ax.text(bx + bw/2, by + bh/2, lab, fontsize=5.6, color=edge,
                ha="center", va="center", zorder=7)
    arrow(x0 + bw, y0 + 0.62*h + bh/2, x0 + bw, y0 + 0.28*h + bh/2,
          color=edge, lw=1.2, ms=8)
    arrow(x0 + bw, y0 + 0.28*h + bh/2, x0 + 0.435*w, y0 + 0.28*h + bh/2,
          color=edge, lw=1.2, ms=8)
    arrow(x0 + 0.435*w + bw, y0 + 0.28*h + bh/2, x0 + 0.87*w, y0 + 0.28*h + bh/2,
          color=edge, lw=1.2, ms=8)
    arrow(x0 + 0.87*w + bw, y0 + 0.28*h + bh/2, x0 + 0.87*w + bw, y0 + 0.62*h + bh/2,
          color=edge, lw=1.2, ms=8)

# ============================== BAND 1: DATA
band(0.01, 0.635, 0.98, 0.33, B1, E1, "(a)  Data & conditioning", "#3E6B9E")
tw, th = 0.105, 0.135
ty = 0.775
thumb_gauges(0.035, ty, tw, th)
thumb_field(0.175, ty, tw, th, "Blues", n=7, seed=1, label="ERA5 (msl, u10, v10)")
thumb_field(0.315, ty, tw, th, "YlOrBr", n=5, seed=3, label="GTSM coarse surge")
thumb_mask(0.455, ty, tw, th)
arrow(0.575, ty + th/2, 0.655, ty + th/2, lw=1.6, ms=12)
chip(0.665, ty - 0.005, 0.155, 0.14, C_CTX, EN_CTX, r"$c=[B,T,6,H,W]$", fs=7.8, weight="bold")
chip(0.85, ty + 0.028, 0.12, 0.062, C_CTX, EN_CTX, r"lead time $L$", fs=7.2)
arrow(0.91, ty + 0.028, 0.835, ty + 0.055, ls="--", color=LIGHT, ms=9)
ax.text(0.875, ty + 0.115, "FiLM", fontsize=6.0, color=LIGHT, ha="center")

# ============================== BAND 2: TRAINING
band(0.01, 0.335, 0.98, 0.28, B2, E2, "(b)  Generative prior - conditional diffusion (training)", "#3E7C43")
chip(0.035, 0.522, 0.095, 0.062, C_CTX, EN_CTX, r"$x_0$ (target)", fs=7.2)
chip(0.035, 0.438, 0.095, 0.062, C_CTX, EN_CTX, r"$\varepsilon\sim N(0,I)$", fs=7.2)
arrow(0.13, 0.553, 0.175, 0.553, lw=1.4)
arrow(0.13, 0.469, 0.175, 0.545, lw=1.2, color=LIGHT, ms=9)
chip(0.185, 0.512, 0.165, 0.082, C_CTX, EN_CTX, r"$x_t=x_0+\sigma_t\varepsilon$", fs=7.4)
arrow(0.35, 0.553, 0.415, 0.545, lw=1.4)
unet_icon(0.545, 0.495, 0.235, 0.175, C_GEN, EN_GEN)
ax.text(0.545, 0.388, r"conditional denoiser  $D_\theta(x_t,\sigma,c,L)$",
        fontsize=7.4, color=EN_GEN, ha="center", va="center", zorder=6, fontweight="bold")
chip(0.365, 0.372, 0.145, 0.058, C_CTX, EN_CTX, r"$\sigma_t\sim$ lognormal", fs=6.8)
arrow(0.43, 0.372, 0.50, 0.415, ls="--", color=LIGHT, ms=9)
arrow(0.72, 0.635, 0.62, 0.585, ls="--", color=LIGHT, ms=9)
arrow(0.79, 0.635, 0.79, 0.585, ls="--", color=LIGHT, ms=9)
chip(0.815, 0.512, 0.135, 0.082, C_GEN, EN_GEN, r"$\hat{x}_0\approx x_0$", fs=7.4)
arrow(0.68, 0.553, 0.815, 0.553, lw=1.4, ms=11)
formula_box(0.375, 0.348, 0.605, 0.088, C_WHITE, EN_GEN,
            r"$\mathcal{L}=\mathbb{E}_{\sigma,\varepsilon}\left[\lambda(\sigma)\left\|"
            r"D_\theta(x_0+\sigma\varepsilon,\sigma,c,L)-x_0\right\|^2\right]$",
            note="sparse-masked, weighted-l1  +  auxiliary GTSM supervision (residual)",
            fs=6.9, note_fs=5.8)

# ============================== BAND 3: INFERENCE
band(0.01, 0.02, 0.98, 0.295, B3, E3, "(c)  Data assimilation - SDA posterior sampling (inference)", "#C05A43")
chip(0.035, 0.175, 0.145, 0.075, C_IN, EN_IN, r"new gauge obs.", fs=7.0, tc="#2C5E8C")
ax.text(0.1075, 0.155, r"$y=\mathcal{A}x_0+\eta,\ \eta\sim N(0,R)$", fontsize=6.2,
        color="#4E7FB5", ha="center", va="top", zorder=6)
chip(0.035, 0.075, 0.145, 0.062, C_CTX, EN_CTX, r"$\mathcal{A}$: sparse mask", fs=6.8)
arrow(0.18, 0.20, 0.245, 0.185, lw=1.3, ms=10)
formula_box(0.245, 0.055, 0.40, 0.21, C_DA, EN_DA,
            r"$\nabla\log p(x_t\,|\,y,c)\ =\ s_\theta(x_t,\sigma,c)$"
            r"$\ +\ \nabla\log\mathcal{N}(y\,|\,\mathcal{A}\hat{x}_0,\,R)$",
            note="prior score  +  likelihood guidance  -  zero-shot, no retraining",
            fs=6.8, note_fs=5.8, tc="#8A3424")
arrow(0.55, 0.335, 0.47, 0.27, ls="--", color=LIGHT, ms=9)
ax.add_patch(Circle((0.745, 0.165), 0.052, fc="none", ec=EN_OUT, lw=1.4, zorder=5))
ax.add_patch(Wedge((0.745, 0.165), 0.052, 70, 140, width=0.004, fc=EN_OUT,
                   ec="none", zorder=6))
ax.text(0.745, 0.232, "annealed", fontsize=6.2, color=INK, ha="center", va="bottom")
ax.text(0.745, 0.218, "Langevin / SDE", fontsize=6.2, color=INK, ha="center", va="bottom")
ax.text(0.745, 0.095, r"$\times N$", fontsize=7.2, color=EN_OUT, ha="center", va="top",
        fontweight="bold")
arrow(0.70, 0.185, 0.685, 0.17, ls="--", color=LIGHT, ms=9)
ex, ew, eh = 0.812, 0.055, 0.135
ey = 0.085
for i, (cmap, lab, seed) in enumerate([("coolwarm", "mean", 11),
                                       ("coolwarm", "quantile", 12),
                                       ("YlOrRd", r"$P_{exc}$", 13)]):
    x = ex + i * (ew + 0.008)
    rng = np.random.default_rng(seed)
    z = rng.random((6, 6))
    try:
        from scipy.ndimage import gaussian_filter
        z = gaussian_filter(z, sigma=0.9)
    except Exception:
        pass
    ax.imshow(z, extent=(x, x + ew, ey, ey + eh), cmap=cmap, aspect="auto",
              interpolation="bilinear", zorder=2)
    ax.add_patch(Rectangle((x, ey), ew, eh, fc="none", ec="#C9A06B", lw=0.8, zorder=3))
    ax.text(x + ew/2, ey + eh + 0.010, lab, fontsize=5.9, color=INK,
            ha="center", va="bottom")
arrow(0.80, 0.165, 0.812, 0.145, lw=1.3, ms=10)

# ------------------------------------------------------------------ sanity
fig.canvas.draw()
r = fig.canvas.get_renderer()
ab = ax.get_window_extent(r)
issues = 0
for t in ax.texts:
    bb = t.get_window_extent(r)
    if (bb.x0 < ab.x0 - 8 or bb.x1 > ab.x1 + 8 or bb.y0 < ab.y0 - 8 or
            bb.y1 > ab.y1 + 8):
        issues += 1
        print(f"OVERFLOW W={bb.width:.0f}: {t.get_text()[:55]!r}")
print("OK: all texts fit" if issues == 0 else f"{issues} overflow(s)")

out = "/Volumes/code_copy/科研工作2026/SDADiff/temp/figures"
fig.savefig(f"{out}/framework_v2.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out}/framework_v2.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out}/framework_v2.svg", bbox_inches="tight", pad_inches=0.02)
print("saved framework_v2: PDF / PNG(600dpi) / SVG")
