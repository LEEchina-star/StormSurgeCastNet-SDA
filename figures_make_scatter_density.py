# -*- coding: utf-8 -*-
"""Fig: predicted vs observed surge at the 45 held-out GESLA gauges, in METRES.
SDA points hug the 1:1 line (MAE 0.017/0.016 m), U-TAE scatters (0.178 m, matching
Ebel et al. 2024 paper values of 0.158-0.190 m)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "axes.linewidth": 0.8})

STD, MEAN = 0.16917373301090485, -0.0004041124621210444   # GESLA denormalisation (metres)
d = np.load("temp/figures/preds_final.npz")
obs_m = d["obs_g"] * STD + MEAN
models = [("SDA-Diff $256^2$", d["p256_g"] * STD + MEAN, "#1f77b4"),
          ("SDA-Diff $128^2$", d["p128_g"] * STD + MEAN, "#2ca02c"),
          ("FiLM U-TAE $128^2$", d["put_g"] * STD + MEAN, "#d62728")]

def mae_m(pm):
    return float(np.mean(np.abs(pm - obs_m)))   # per-sample mean |err| (paper convention, metres)

hi = float(np.nanmax(np.concatenate([obs_m] + [p for _, p, _ in models])) * 1.08)
lo = float(np.nanmin(np.concatenate([obs_m] + [p for _, p, _ in models])) * 1.08)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
for ax, (name, pm_, col) in zip(axes, models):
    ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=1.0, zorder=2)
    ax.scatter(obs_m, pm_, c=col, s=34, edgecolor="white", linewidth=0.4, zorder=3, alpha=0.9)
    r = mae_m(pm_); bias = float(np.mean(pm_ - obs_m))
    ax.set_title(name, fontsize=10)
    ax.text(0.03, 0.95, f"MAE {r:.3f} m\nbias {bias:+.3f} m", transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.9))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
fig.text(0.5, 0.01, "Observed surge (m)", ha="center", fontsize=9)
fig.text(0.008, 0.5, "Predicted surge (m)", va="center", rotation=90, fontsize=9)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.93, wspace=0.30)
fig.savefig("temp/figures/fig_scatter_density.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("temp/figures/fig_scatter_density.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("saved fig_scatter_density.{pdf,png}  (METRES)")
for name, pm_, _ in models:
    print(f"  {name:22s} MAE={mae_m(pm_):.4f} m  bias={np.mean(pm_-obs_m):+.4f} m")
