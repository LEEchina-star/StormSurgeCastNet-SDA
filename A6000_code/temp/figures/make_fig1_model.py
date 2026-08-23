# -*- coding: utf-8 -*-
"""Fig.1(b,c): model architecture — conditional diffusion denoiser (EDM) + SDA
posterior sampling, colour-coded by operation type."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "axes.unicode_minus": False})

# colour code (legend)
C_DATA = "#DCE9F7"   # data
C_CONV = "#E3F0E6"   # convolution (generation)
C_DOWN = "#CDE3F5"   # downsample
C_UP   = "#C8EAD3"   # upsample
C_NOISE= "#F5E6D3"   # noising / drop
C_DA   = "#FBE3DE"   # data assimilation (highlight)
C_LOSS = "#EDE3F6"   # loss
E = dict(DATA="#3E6B9E", CONV="#2F7D46", DOWN="#1A5A9E", UP="#1E7A4A",
         NOISE="#B06A1B", DA="#C0392B", LOSS="#6A3D9A", GREY="#444444")

fig = plt.figure(figsize=(7.2, 8.6))
fig.patch.set_facecolor("white")

def box(ax, x, y, w, h, fc, ec, lw=1.2, text="", fs=8, tc="#222222", bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    if text:
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=3)

def arr(ax, x1, y1, x2, y2, color="#333333", lw=1.2, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                        lw=lw, color=color, linestyle=ls, zorder=1,
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)

def rect(ax, x, y, w, h, fc, ec):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8, zorder=2))

# ============================ (b) TRAINING ============================
axT = fig.add_axes([0.02, 0.52, 0.96, 0.46]); axT.set_xlim(0, 10); axT.set_ylim(0, 10); axT.axis("off")
axT.text(0.02, 9.7, "(b) Training — conditional score-based generation (EDM)", fontsize=11, weight="bold", va="top")

# inputs
box(axT, 0.2, 8.3, 1.8, 0.9, C_DATA, E["DATA"], text="target\n$x_0$ [B,1,H,W]", fs=7.5)
box(axT, 0.2, 6.6, 1.8, 0.9, C_DATA, E["DATA"], text="noise\n$\\varepsilon\\sim\\mathcal{N}(0,I)$", fs=7.5)
box(axT, 0.2, 4.9, 1.8, 0.9, C_DATA, E["DATA"], text="context\n$c$ [B,T,6,H,W]", fs=7.5)

# forward noising (drop)
box(axT, 2.6, 7.4, 2.4, 1.0, C_NOISE, E["NOISE"], text="$x_t = x_0 + \\sigma_t\\varepsilon$\n(add noise · drop)", fs=7.5)
arr(axT, 2.0, 8.75, 2.6, 8.0); arr(axT, 2.0, 7.05, 2.6, 7.7)

# conditional denoiser (UNet) — big box
box(axT, 5.6, 3.4, 4.0, 6.2, C_CONV, E["CONV"], lw=1.8, text="conditional denoiser  $D_\\theta(x_t, \\sigma, c, L)$", fs=9, bold=True)
# UNet internals (encoder-decoder)
# encoder
rect(axT, 5.9, 8.6, 1.6, 0.7, C_DOWN, E["DOWN"]); axT.text(6.7, 8.95, "Conv+Down", ha="center", fontsize=6.5)
rect(axT, 5.9, 7.5, 1.6, 0.7, C_DOWN, E["DOWN"]); axT.text(6.7, 7.85, "Conv+Down", ha="center", fontsize=6.5)
rect(axT, 5.9, 6.4, 1.6, 0.7, C_DOWN, E["DOWN"]); axT.text(6.7, 6.75, "Conv+Down", ha="center", fontsize=6.5)
# bottleneck (attention)
rect(axT, 5.9, 5.2, 1.6, 0.9, C_CONV, E["CONV"]); axT.text(6.7, 5.65, "Attention\n(L-TAE ctx)", ha="center", fontsize=6.3)
# decoder (upsample)
rect(axT, 8.1, 5.2, 1.4, 0.7, C_UP, E["UP"]); axT.text(8.8, 5.55, "Up+Skip", ha="center", fontsize=6.5)
rect(axT, 8.1, 6.4, 1.4, 0.7, C_UP, E["UP"]); axT.text(8.8, 6.75, "Up+Skip", ha="center", fontsize=6.5)
rect(axT, 8.1, 7.6, 1.4, 0.7, C_UP, E["UP"]); axT.text(8.8, 7.95, "Up+Skip", ha="center", fontsize=6.5)
# FiLM conditioning (sigma + lead)
box(axT, 7.6, 3.7, 1.9, 0.8, C_NOISE, E["NOISE"], text="FiLM\n($\\nu(\\sigma)$, lead $L$)", fs=6.3)
arr(axT, 8.55, 4.5, 8.55, 5.2, color=E["NOISE"], lw=1.0, ls="--")

# context encoder
box(axT, 2.6, 4.9, 2.2, 1.2, C_CONV, E["CONV"], text="temporal\ncontext encoder\n(shared conv + attn)", fs=6.5)
arr(axT, 2.0, 5.35, 2.6, 5.5); arr(axT, 4.8, 5.5, 5.6, 6.4, color=E["CONV"], lw=1.0, ls="--")

# denoiser output
arr(axT, 6.6, 3.4, 6.6, 2.9)
box(axT, 5.6, 1.9, 2.0, 1.0, C_CONV, E["CONV"], text="$\\hat{x}_0 = D_\\theta$", fs=8)

# loss
arr(axT, 6.6, 1.9, 6.6, 1.4)
box(axT, 3.4, 0.3, 6.4, 1.1, C_LOSS, E["LOSS"], lw=1.4,
    text="$\\mathcal{L} = \\mathbb{E}_{\\sigma,\\varepsilon}\\,\\lambda(\\sigma)\\,\\left\\| D_\\theta(x_0+\\sigma\\varepsilon, \\sigma, c, L) - x_0 \\right\\|^2$  (masked)", fs=8)

# ============================ (c) INFERENCE ============================
axI = fig.add_axes([0.02, 0.02, 0.96, 0.47]); axI.set_xlim(0, 10); axI.set_ylim(0, 10); axI.axis("off")
axI.text(0.02, 9.7, "(c) Inference — SDA likelihood-guided posterior sampling (assimilation)", fontsize=11, weight="bold", va="top")

# start from noise
box(axI, 0.3, 8.0, 1.7, 0.9, C_DATA, E["DATA"], text="$x_T \\sim \\mathcal{N}(0,\\sigma^2_{\\max})$", fs=7)
# denoise loop (EDM Heun)
box(axI, 2.5, 8.0, 2.2, 0.9, C_CONV, E["CONV"], text="denoise (EDM Heun)\n$\\sigma_{\\max}\\to\\sigma_{\\min}$", fs=7)
arr(axI, 2.0, 8.45, 2.5, 8.45)
# Tweedie
box(axI, 5.1, 8.0, 2.0, 0.9, C_CONV, E["CONV"], text="Tweedie estimate\n$\\hat{x}_0 = \\frac{x_t+\\sigma^2 s_\\theta}{\\mu}$", fs=7)
arr(axI, 4.7, 8.45, 5.1, 8.45)
# observation
box(axI, 7.5, 8.0, 1.9, 0.9, C_DATA, E["DATA"], text="new gauge obs. $y$\n$y=\\mathcal{A}x_0+\\eta$, $\\eta\\sim\\mathcal{N}(0,R)$", fs=6.5)
# SDA likelihood guidance (highlight)
box(axI, 5.1, 6.0, 4.3, 1.5, C_DA, E["DA"], lw=2.0,
    text="SDA posterior score\n$\\nabla\\log p(x_t|y,c)=s_\\theta + \\nabla\\log\\mathcal{N}(y|\\mathcal{A}\\hat{x}_0,R)$\n(assimilation · zero-shot · no retraining)", fs=7, tc=E["DA"], bold=True)
arr(axI, 6.1, 8.0, 6.1, 7.5); arr(axI, 8.45, 8.0, 7.3, 7.5)
arr(axI, 4.1, 8.0, 5.6, 7.5)   # denoise -> SDA
# loop arrow
arr(axI, 6.1, 6.0, 3.6, 8.9, color=E["DA"], lw=1.0, ls="--")
axI.text(4.6, 7.2, "×N steps", fontsize=6.5, color=E["DA"], style="italic")

# ensemble output
box(axI, 3.0, 3.6, 5.0, 1.6, C_DATA, E["DATA"], lw=1.4,
    text="posterior ensemble $\\{x_0^{(i)}\\}_{i=1}^{N}$\nmean · quantiles · $\\mathbb{P}(\\mathrm{surge}>h)$", fs=8)
arr(axI, 6.1, 6.0, 5.5, 5.2)
# dense field
box(axI, 3.4, 1.6, 4.2, 1.4, C_UP, E["UP"], lw=1.4, text="dense surge forecast\n(uncertainty-aware)", fs=8, bold=True)
arr(axI, 5.0, 3.6, 4.6, 3.0)

# legend (colour code)
axL = fig.add_axes([0.02, 0.015, 0.96, 0.0]); 
legend_items = [("data / tensors", C_DATA, E["DATA"]), ("convolution / denoise", C_CONV, E["CONV"]),
                ("downsample", C_DOWN, E["DOWN"]), ("upsample / skip", C_UP, E["UP"]),
                ("noise / drop / FiLM", C_NOISE, E["NOISE"]), ("SDA assimilation (highlight)", C_DA, E["DA"]),
                ("loss", C_LOSS, E["LOSS"])]
lx = 0.02
for lbl, fc, ec in legend_items:
    axL.add_patch(FancyBboxPatch((lx, 0), 0.13, 0.03, boxstyle="round,pad=0.003", fc=fc, ec=ec, lw=0.8,
                                  transform=axL.transAxes, zorder=2))
    axL.text(lx+0.145, 0.015, lbl, transform=axL.transAxes, fontsize=6.5, va="center")
    lx += 0.145 + 0.03*len(lbl)/6
axL.axis("off")

fig.savefig("temp/figures/fig1_model.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("temp/figures/fig1_model.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("saved fig1_model.{pdf,png}")
