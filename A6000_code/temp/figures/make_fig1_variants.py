# -*- coding: utf-8 -*-
"""Three layout variants of the SDA-Diff framework figure (Fig.1 candidates).
Colour code: conv=green, downsample=blue, upsample=cyan, noise/drop=orange,
SDA assimilation=red (sole highlight), loss=purple, data=blue-grey."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
                     "mathtext.fontset":"dejavusans","axes.unicode_minus":False})

# palette
FC = dict(DATA="#DCE9F7", CONV="#E3F0E6", DOWN="#CDE3F5", UP="#C8EAD3",
          NOISE="#F5E6D3", DA="#FBE3DE", LOSS="#EDE3F6")
EC = dict(DATA="#3E6B9E", CONV="#2F7D46", DOWN="#1A5A9E", UP="#1E7A7A",
          NOISE="#B06A1B", DA="#C0392B", LOSS="#6A3D9A", GREY="#444444")

def box(ax, x, y, w, h, kind, text="", fs=7.5, bold=False, lw=1.2, tc=None):
    p = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                       fc=FC.get(kind, kind), ec=EC.get(kind, kind), lw=lw, zorder=2)
    ax.add_patch(p)
    if text:
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
                color=tc or "#222", fontweight="bold" if bold else "normal", zorder=3)

def arr(ax, x1,y1,x2,y2, color=EC["GREY"], lw=1.2, ls="-", ms=10):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls, zorder=1))

def rect(ax, x,y,w,h, kind):
    ax.add_patch(Rectangle((x,y), w, h, fc=FC.get(kind, kind), ec=EC.get(kind, kind), lw=0.8, zorder=2))

def title(ax, s, x=0.02, y=0.98, fs=11):
    ax.text(x, y, s, fontsize=fs, weight="bold", va="top")

# ---- shared input icon helper (Ebel-style three-band stack) ----
def draw_input_bands(ax, x, y, w, h):
    """three stacked bands: sparse in-situ + mask / ERA5 / GTSM, with 1..T"""
    bands = [("sparse in-situ + valid mask", "DATA", "point gauges"),
             ("ERA5  msl·u10·v10", "DATA", "grid"),
             ("GTSM surge", "DATA", "field")]
    bh = h/3
    for i,(lab, k, icon) in enumerate(bands):
        yy = y + (2-i)*bh
        box(ax, x, yy, w, bh*0.92, k, text=lab, fs=6.5)
        ax.text(x+w-0.08, yy+bh*0.46, "1 … T", fontsize=5.5, color=EC["GREY"], va="center")
    ax.text(x-0.25, y+h/2, "c=[B,T,6,H,W]\n+ lead L", fontsize=6, rotation=90, va="center", ha="center")

# =====================================================================
# VARIANT A — horizontal 3-column (Ebel-style left->right)
# =====================================================================
def variant_A():
    fig = plt.figure(figsize=(7.2, 4.6)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,14); ax.set_ylim(0,9); ax.axis("off")
    title(ax, "SDA-Diff framework (horizontal)")
    # input (left)
    box(ax, 0.3, 0.8, 2.6, 7.4, "DATA", lw=1.5)
    ax.text(1.6, 8.4, "Input", fontsize=9, weight="bold", ha="center")
    draw_input_bands(ax, 0.55, 1.1, 2.1, 6.6)
    # model (center)
    box(ax, 3.3, 0.8, 6.2, 7.4, "CONV", lw=1.5)
    ax.text(6.4, 8.4, "Model — conditional diffusion (EDM) + SDA assimilation", fontsize=9, weight="bold", ha="center")
    # training path (top)
    box(ax, 3.6, 6.2, 5.6, 1.7, "CONV", text="", lw=1.0)
    ax.text(3.75, 7.85, "Training — learn to recover the surge field from noise", fontsize=7, weight="bold", color=EC["CONV"], ha="left")
    box(ax, 3.9, 6.45, 1.3, 0.7, "DATA", text="$x_0$ target", fs=6)
    box(ax, 5.5, 6.45, 1.5, 0.7, "NOISE", text="$x_t=x_0+\\sigma\\varepsilon$\n(noise/drop)", fs=6)
    arr(ax, 5.2, 6.8, 5.5, 6.8, color=EC["NOISE"])
    box(ax, 7.3, 6.45, 1.7, 0.7, "CONV", text="$D_\\theta$ UNet\n(conv+down+up)", fs=6)
    arr(ax, 7.0, 6.8, 7.3, 6.8)
    # inference path (bottom)
    box(ax, 3.6, 3.0, 5.6, 1.7, "DA", text="", lw=1.3)
    ax.text(3.75, 4.65, "Inference — correct the sampling path with new observations", fontsize=7, weight="bold", color=EC["DA"], ha="left")
    box(ax, 3.9, 3.25, 1.2, 0.65, "DATA", text="noise $x_T$", fs=6)
    box(ax, 5.3, 3.25, 1.5, 0.65, "CONV", text="denoise (Heun)", fs=6)
    arr(ax, 5.1, 3.57, 5.3, 3.57)
    box(ax, 7.0, 3.25, 1.5, 0.65, "CONV", text="Tweedie $\\hat{x}_0$", fs=6)
    arr(ax, 6.8, 3.57, 7.0, 3.57)
    box(ax, 5.3, 2.4, 3.2, 0.65, "DA", text="SDA: $\\nabla\\log p = s_\\theta + \\nabla\\log\\mathcal{N}(y|\\mathcal{A}\\hat{x}_0,R)$", fs=5.8, bold=True, tc=EC["DA"])
    arr(ax, 7.0, 3.25, 6.5, 3.05, color=EC["DA"])
    box(ax, 3.9, 2.4, 1.1, 0.65, "DATA", text="obs. $y$", fs=6)
    arr(ax, 5.0, 2.72, 5.3, 2.72, color=EC["DA"], ls="--")
    # output (right)
    box(ax, 9.9, 0.8, 3.7, 7.4, "DATA", lw=1.5)
    ax.text(11.75, 8.4, "Output (ensemble)", fontsize=9, weight="bold", ha="center")
    box(ax, 10.2, 6.2, 3.1, 0.8, "CONV", text="posterior samples\n$\\{x_0^{(i)}\\}_{i=1}^{N}$", fs=6.5)
    box(ax, 10.2, 4.6, 3.1, 1.2, "UP", text="mean · quantiles ·\n$\\mathbb{P}(\\mathrm{surge}>h)$", fs=6.5)
    box(ax, 10.2, 3.0, 3.1, 1.2, "UP", text="dense surge forecast\n(uncertainty-aware)", fs=6.5, bold=True)
    arr(ax, 6.4, 6.5, 9.9, 6.5); arr(ax, 6.4, 3.4, 9.9, 5.5, color=EC["DA"])
    # loss
    box(ax, 3.6, 1.3, 5.6, 1.1, "LOSS", lw=1.4,
        text="$\\mathcal{L}=\\mathbb{E}_{\\sigma,\\varepsilon}\\,\\lambda(\\sigma)\\,\\| D_\\theta(x_0+\\sigma\\varepsilon,\\sigma,c,L)-x_0 \\|^2$ (masked)", fs=6.5)
    arr(ax, 6.4, 6.2, 6.4, 2.4, color=EC["LOSS"], lw=1.0, ls="--")
    fig.savefig("temp/figures/fig1_vA.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("temp/figures/fig1_vA.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved fig1_vA (horizontal)")

# =====================================================================
# VARIANT B — vertical, training top / inference bottom (two paths)
# =====================================================================
def variant_B():
    fig = plt.figure(figsize=(7.2, 7.2)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,14); ax.set_ylim(0,14); ax.axis("off")
    title(ax, "SDA-Diff framework (vertical, two paths)")
    # ---- input (top) ----
    box(ax, 0.5, 10.6, 13.0, 3.0, "DATA", lw=1.5)
    ax.text(0.7, 13.55, "Input (identical to StormSurgeCastNet / Ebel)", fontsize=9, weight="bold")
    draw_input_bands(ax, 0.8, 10.9, 3.0, 2.4)
    # context -> model arrow
    box(ax, 4.2, 11.0, 2.2, 2.2, "CONV", text="temporal\ncontext encoder\n(shared conv + attn)", fs=6)
    ax.text(7.0, 12.4, "lead time L (FiLM)", fontsize=6.5, color=EC["GREY"])
    # ---- TRAINING (upper path, dashed) ----
    box(ax, 0.5, 6.6, 13.0, 3.6, "white", lw=1.0)
    ax.text(0.7, 10.0, "Training — learn to recover the surge field from noise", fontsize=8.5, weight="bold", color=EC["CONV"])
    box(ax, 1.0, 7.0, 1.6, 1.0, "DATA", text="$x_0$\ntarget", fs=6.5)
    box(ax, 3.0, 7.0, 1.9, 1.0, "NOISE", text="$x_t = x_0+\\sigma\\varepsilon$\n(add noise / drop)", fs=6.5)
    arr(ax, 2.6, 7.5, 3.0, 7.5, color=EC["NOISE"])
    box(ax, 5.3, 6.9, 2.6, 1.2, "CONV", text="conditional denoiser\n$D_\\theta(x_t,\\sigma,c,L)$", fs=6.5, bold=True)
    arr(ax, 4.9, 7.5, 5.3, 7.5)
    # UNet internals
    rect(ax, 5.6, 5.9, 0.8, 0.8, "DOWN"); ax.text(6.0, 6.3, "Conv\nDown", fontsize=5, ha="center")
    rect(ax, 6.6, 5.9, 0.8, 0.8, "CONV"); ax.text(7.0, 6.3, "Attn", fontsize=5, ha="center")
    rect(ax, 7.6, 5.9, 0.8, 0.8, "UP"); ax.text(8.0, 6.3, "Up\nSkip", fontsize=5, ha="center")
    box(ax, 8.6, 7.0, 1.7, 1.0, "CONV", text="$\\hat{x}_0 = D_\\theta$", fs=6.5)
    arr(ax, 7.9, 7.5, 8.6, 7.5)
    box(ax, 10.6, 6.9, 2.6, 1.2, "LOSS", lw=1.3,
        text="$\\mathcal{L}=\\mathbb{E}[\\lambda(\\sigma)\\|D_\\theta(x_0+\\sigma\\varepsilon)-x_0\\|^2]$", fs=6)
    arr(ax, 10.3, 7.5, 10.6, 7.5, color=EC["LOSS"])
    ax.text(1.0, 6.75, "masked to valid pixels (ocean + gauges)", fontsize=5.5, color=EC["GREY"], style="italic")
    # ---- INFERENCE (lower path, dashed, red) ----
    box(ax, 0.5, 1.2, 13.0, 5.0, "white", lw=1.0)
    ax.text(0.7, 6.05, "Inference — correct the sampling path with new observations", fontsize=8.5, weight="bold", color=EC["DA"])
    box(ax, 1.0, 2.0, 1.6, 0.9, "DATA", text="$x_T\\sim\\mathcal{N}(0,\\sigma^2)$", fs=6)
    box(ax, 3.0, 2.0, 2.0, 0.9, "CONV", text="denoise loop\n(EDM Heun, ×N steps)", fs=6)
    arr(ax, 2.6, 2.45, 3.0, 2.45)
    box(ax, 5.4, 2.0, 1.9, 0.9, "CONV", text="Tweedie\n$\\hat{x}_0$", fs=6)
    arr(ax, 5.0, 2.45, 5.4, 2.45)
    box(ax, 7.6, 2.0, 2.4, 0.9, "DATA", text="obs. $y$, $\\mathcal{A}$, $R$\n(gauge pixels)", fs=6)
    box(ax, 5.4, 3.4, 4.6, 1.4, "DA", lw=2.0, bold=True, tc=EC["DA"],
        text="SDA posterior score\n$\\nabla\\log p(x_t|y,c)=s_\\theta+\\nabla\\log\\mathcal{N}(y|\\mathcal{A}\\hat{x}_0,R)$", fs=6.2)
    arr(ax, 6.3, 2.0, 6.5, 3.4, color=EC["DA"]); arr(ax, 8.8, 2.9, 7.7, 3.4, color=EC["DA"], ls="--")
    arr(ax, 6.5, 3.4, 3.5, 2.9, color=EC["DA"], lw=1.0, ls="--")
    ax.text(4.8, 2.9, "×N (ensemble)", fontsize=6, color=EC["DA"], style="italic")
    box(ax, 10.4, 2.0, 2.6, 1.6, "UP", text="ensemble output\nmean · quantiles\n$\\mathbb{P}(\\mathrm{surge}>h)$", fs=6, bold=True)
    arr(ax, 10.0, 2.45, 10.4, 2.45, color=EC["DA"])
    box(ax, 10.4, 0.6, 2.6, 1.0, "UP", text="dense surge\n(uncertainty)", fs=6)
    # legend
    lx = 0.7
    legend = [("convolution / denoise","CONV"),("downsample","DOWN"),("upsample/skip","UP"),
              ("noise / drop","NOISE"),("SDA assimilation (highlight)","DA"),("loss","LOSS"),("data","DATA")]
    for lab,k in legend:
        ax.add_patch(FancyBboxPatch((lx,0.15),0.85,0.32, boxstyle="round,pad=0.01", fc=FC[k], ec=EC[k], lw=0.7))
        ax.text(lx+0.9,0.31,lab,fontsize=5.5,va="center")
        lx += 0.9 + 0.115*len(lab)
    fig.savefig("temp/figures/fig1_vB.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("temp/figures/fig1_vB.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved fig1_vB (vertical)")

# =====================================================================
# VARIANT C — centre-out / magazine style
# =====================================================================
def variant_C():
    fig = plt.figure(figsize=(7.2, 5.2)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,14); ax.set_ylim(0,10); ax.axis("off")
    title(ax, "SDA-Diff framework (centre-out)")
    # input left
    box(ax, 0.4, 1.0, 3.2, 8.0, "DATA", lw=1.4)
    ax.text(2.0, 9.2, "Input", fontsize=9, weight="bold", ha="center")
    draw_input_bands(ax, 0.6, 1.3, 2.8, 7.0)
    # model core (centre)
    box(ax, 4.2, 1.0, 6.0, 8.0, "CONV", lw=1.8)
    ax.text(7.2, 9.2, "Model — EDM generation + SDA assimilation", fontsize=9, weight="bold", ha="center")
    # training ring (top half)
    box(ax, 4.6, 5.6, 5.2, 3.0, "CONV", text="", lw=1.0)
    ax.text(4.8, 8.45, "Training — learn to recover the surge field from noise", fontsize=7, weight="bold", color=EC["CONV"], ha="left")
    box(ax, 5.0, 6.2, 1.4, 0.8, "DATA", text="$x_0$", fs=6)
    box(ax, 6.8, 6.2, 1.6, 0.8, "NOISE", text="+$\\sigma\\varepsilon$\n(noise/drop)", fs=5.8)
    arr(ax, 6.4, 6.6, 6.8, 6.6, color=EC["NOISE"])
    box(ax, 8.6, 6.0, 1.0, 1.2, "CONV", text="$D_\\theta$\nUNet", fs=6)
    arr(ax, 8.4, 6.6, 8.6, 6.6)
    # inference ring (bottom half, red)
    box(ax, 4.6, 1.4, 5.2, 3.6, "DA", text="", lw=1.3)
    ax.text(4.8, 4.85, "Inference — correct the sampling path with new observations", fontsize=7, weight="bold", color=EC["DA"], ha="left")
    box(ax, 5.0, 2.2, 1.3, 0.8, "DATA", text="noise", fs=6)
    box(ax, 6.6, 2.2, 1.4, 0.8, "CONV", text="denoise\n(Heun)", fs=5.8)
    arr(ax, 6.3, 2.6, 6.6, 2.6)
    box(ax, 8.2, 2.2, 1.4, 0.8, "CONV", text="Tweedie\n$\\hat{x}_0$", fs=5.8)
    arr(ax, 8.0, 2.6, 8.2, 2.6)
    box(ax, 6.4, 3.6, 3.2, 1.1, "DA", bold=True, tc=EC["DA"],
        text="SDA: $\\nabla\\log p = s_\\theta + \\nabla\\log\\mathcal{N}(y|\\mathcal{A}\\hat{x}_0,R)$", fs=5.8)
    arr(ax, 8.8, 2.6, 7.2, 3.6, color=EC["DA"]); arr(ax, 7.2, 3.6, 6.8, 3.0, color=EC["DA"], ls="--")
    # output right
    box(ax, 10.6, 1.0, 3.0, 8.0, "DATA", lw=1.4)
    ax.text(12.1, 9.2, "Output", fontsize=9, weight="bold", ha="center")
    box(ax, 10.9, 6.6, 2.4, 0.9, "CONV", text="ensemble\n$\\{x_0^{(i)}\\}$", fs=6)
    box(ax, 10.9, 4.9, 2.4, 1.3, "UP", text="mean · quantiles\n$\\mathbb{P}(\\mathrm{surge}>h)$", fs=6)
    box(ax, 10.9, 3.1, 2.4, 1.3, "UP", text="dense surge\n(uncertainty-aware)", fs=6, bold=True)
    arr(ax, 7.2, 6.5, 10.6, 6.5); arr(ax, 7.2, 2.8, 10.9, 4.0, color=EC["DA"])
    # loss bottom
    box(ax, 4.6, 0.15, 5.2, 0.9, "LOSS", lw=1.3,
        text="$\\mathcal{L}=\\mathbb{E}[\\lambda(\\sigma)\\|D_\\theta(x_0+\\sigma\\varepsilon)-x_0\\|^2]$", fs=6)
    arr(ax, 7.2, 5.6, 7.2, 1.05, color=EC["LOSS"], lw=1.0, ls="--")
    fig.savefig("temp/figures/fig1_vC.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("temp/figures/fig1_vC.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved fig1_vC (centre-out)")

variant_A()
variant_B()
variant_C()
