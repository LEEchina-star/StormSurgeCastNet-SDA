import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"axes.linewidth":0.8})
a=np.load("temp/figures/lead_sensitivity.npz"); p=np.load("temp/figures/lead_prior_sensitivity.npz")
L=a["leads"]
fig,ax=plt.subplots(figsize=(3.6,2.7))
ax.plot(L,a["mae_m"],"-o",color="#1f77b4",lw=1.6,label="SDA-Diff posterior (assimilated)")
ax.plot(L,p["mae_m"],"-s",color="#d62728",lw=1.6,label="generative prior (no obs)")
ax.axhline(0.178,color="#999999",ls=":",lw=1.2,label="FiLM U-TAE (0.178 m)")
ax.set_xlabel("Lead time $L$ (h)"); ax.set_ylabel("MAE (m)")
ax.set_xticks(L); ax.set_ylim(0,0.25); ax.grid(alpha=0.3,lw=0.4)
ax.legend(fontsize=7,frameon=False,loc="upper left")
fig.tight_layout()
fig.savefig("temp/figures/fig_lead_sensitivity.pdf",bbox_inches="tight",pad_inches=0.02)
fig.savefig("temp/figures/fig_lead_sensitivity.png",dpi=600,bbox_inches="tight",pad_inches=0.02)
print("saved fig_lead_sensitivity.{pdf,png}")
