# -*- coding: utf-8 -*-
"""Figure: SDA-Diff ablation (guidance / observation noise R / ensemble size)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "axes.unicode_minus": False})

ab = json.load(open("results_sda/real_era5_128/ablation.json"))
test = json.load(open("results_sda/test_real_era5/test_report.json"))

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))  # 183mm double column

# (a) guidance
g = [0.0, 0.3, 0.5, 1.0, 2.0]
rmse = [ab[f"guidance_{x}"]["rmse"] for x in g]
cov = [ab[f"guidance_{x}"]["cov90"] for x in g]
ax = axes[0]
ax.plot(g, rmse, "o-", color="#2F7D46", label="RMSE (normalised)")
ax.set_xlabel("Guidance weight $\\gamma$"); ax.set_ylabel("RMSE", color="#2F7D46")
ax.set_title("(a) Likelihood guidance", fontsize=10)
ax2 = ax.twinx()
ax2.plot(g, cov, "s--", color="#C0392B", label="90% coverage")
ax2.set_ylabel("90% coverage", color="#C0392B"); ax2.set_ylim(0, 1)
ax.grid(alpha=0.3, lw=0.5)

# (b) R
R = [0.03, 0.05, 0.1, 0.2, 0.5]
rmse = [ab[f"R_{x}"]["rmse"] for x in R]
cov = [ab[f"R_{x}"]["cov90"] for x in R]
ax = axes[1]
ax.plot(R, rmse, "o-", color="#2F7D46")
ax.set_xlabel("Observation noise $R$ (m)"); ax.set_ylabel("RMSE", color="#2F7D46")
ax.set_title("(b) Observation noise", fontsize=10)
ax2 = ax.twinx()
ax2.plot(R, cov, "s--", color="#C0392B"); ax2.set_ylim(0, 1)
ax.grid(alpha=0.3, lw=0.5)

# (c) ensemble
e = [2, 4, 8, 16]
rmse = [ab[f"ensemble_{x}"]["rmse"] for x in e]
cov = [ab[f"ensemble_{x}"]["cov90"] for x in e]
ax = axes[2]
ax.plot(e, rmse, "o-", color="#2F7D46")
ax.set_xlabel("Ensemble size $N$"); ax.set_ylabel("RMSE", color="#2F7D46")
ax.set_title("(c) Posterior ensemble", fontsize=10)
ax2 = ax.twinx()
ax2.plot(e, cov, "s--", color="#C0392B"); ax2.set_ylim(0, 1)
ax.grid(alpha=0.3, lw=0.5)

fig.tight_layout()
fig.savefig("temp/figures/fig_ablation.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("temp/figures/fig_ablation.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("saved fig_ablation.{pdf,png}")
print(f"正式测试 RMSE={test['rmse']:.4f} CRPS={test['crps']:.4f} cov90={test['coverage_90']:.3f}")
