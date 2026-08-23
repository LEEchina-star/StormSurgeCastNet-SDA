# -*- coding: utf-8 -*-
"""Training-loss variable legend: coloured-background figure for the framework."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
BG = "#EDE3F6"; BORDER = "#6A3D9A"

fig, ax = plt.subplots(figsize=(9.5, 4.0))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
for sp in ax.spines.values(): sp.set_color(BORDER); sp.set_linewidth(1.2)

ax.text(0.5, 0.94, "Training loss & variable definitions", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#4a2a6e", transform=ax.transAxes)

ax.text(0.5, 0.80,
        r"$L \;=\; \mathrm{E}_{\sigma,\varepsilon}\left[ \lambda(\sigma)\;"
        r"\left\| D_\theta\left(x_0 + \sigma\varepsilon,\ \sigma,\ c,\ L\right) - x_0 \right\|^{2} \right]$"
        r"   (masked, valid pixels only)",
        ha="center", va="center", fontsize=14, color="#3a3a3a", transform=ax.transAxes)

Lx, Rx = 0.045, 0.55; y0, dy = 0.60, 0.125
items_l = [
    (r"$L$", "training loss (objective to minimise)"),
    (r"$\mathrm{E}_{\sigma,\varepsilon}[\cdot]$", "expectation over noise level $\\sigma$ and noise $\\varepsilon$"),
    (r"$\lambda(\sigma)$", r"noise-level weight $= \frac{\sigma^2+\sigma_{\mathrm{data}}^2}{(\sigma\,\sigma_{\mathrm{data}})^2}$"),
    (r"$\left\|\cdot\right\|^2$", "squared L2 norm, summed over pixels"),
]
items_r = [
    (r"$x_0$", "clean surge field (target = GTSM + GESLA)"),
    (r"$\sigma\varepsilon$", r"Gaussian noise, $\varepsilon \sim \mathcal{N}(0,I)$"),
    (r"$D_\theta$", "conditional denoiser (U-Net, learnable $\\theta$)"),
    (r"$c,\; L$", "multi-source context $c$ and lead time $L$"),
]
for i, (sym, mean) in enumerate(items_l + items_r):
    xx = Lx if i < 4 else Rx
    ax.text(xx, y0 - (i % 4) * dy, sym, ha="left", va="center", fontsize=12,
            color="#4a2a6e", transform=ax.transAxes)
    ax.text(xx + 0.13, y0 - (i % 4) * dy, mean, ha="left", va="center", fontsize=9.5,
            color="#333333", transform=ax.transAxes)

ax.text(0.5, 0.05, "masked: the squared error is averaged only over valid (ocean / observed) pixels",
        ha="center", va="center", fontsize=9, style="italic", color="#555555", transform=ax.transAxes)

fig.savefig("temp/figures/loss_legend.png", dpi=300, bbox_inches="tight", facecolor=BG, pad_inches=0.08)
print("saved temp/figures/loss_legend.png")
