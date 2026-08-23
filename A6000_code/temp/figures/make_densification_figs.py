# -*- coding: utf-8 -*-
"""Densification figure with a LARGE-surge sample (id=1118, GESLA 2.94 m, Gulf):
coarse GTSM / ERA5 / sparse inputs + generated dense 256x256 surge field."""
import numpy as np, torch, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from types import SimpleNamespace
from models.edm_da import EDMDataAssimilation

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica","Arial"],
                     "mathtext.fontset": "dejavusans", "axes.linewidth": 0.8})
STD = dict(GTSM=0.1284419447183609, GESLA=0.16917373301090485, msl=1378.7696533203125,
           u10=4.459522724151611, v10=4.0246148109436035)
MEAN = dict(GTSM=0.0009648051927797496, GESLA=-0.0004041124621210444, msl=101002.3515625,
            u10=0.4772440791130066, v10=0.20318500697612762)
GREY = "#D4D4D4"

sd = np.load("temp/figures/big_surge_sample.npz")
X, y, yg = sd["X"], sd["y"], sd["yg"]
land = np.isnan(yg[0]); fr = -1

def show(ax, arr, cmap, vmin, vmax, title, units=None):
    cm = plt.get_cmap(cmap).copy(); cm.set_bad(GREY)
    im = ax.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    return im

# ---- Fig 1: inputs ----
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
gtsm = X[fr, 5].astype(np.float32).copy(); gtsm[land] = np.nan
im = show(axes[0,0], gtsm*STD["GTSM"]+MEAN["GTSM"], "RdBu_r", -1.5, 3.0, "(a) coarse GTSM input (m)")
fig.colorbar(im, ax=axes[0,0], fraction=0.046, pad=0.04).set_label("m", fontsize=8)
im = show(axes[0,1], X[fr,2]*STD["msl"]/100+MEAN["msl"]/100, "RdBu_r", 985, 1015, "(b) ERA5 msl (hPa)")
fig.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04).set_label("hPa", fontsize=8)
ws = np.sqrt((X[fr,3]*STD["u10"]+MEAN["u10"])**2 + (X[fr,4]*STD["v10"]+MEAN["v10"])**2)
im = show(axes[1,0], ws, "viridis", 0, 22, "(c) ERA5 10-m wind speed (m/s)")
fig.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04).set_label("m/s", fontsize=8)
axes[1,1].imshow(np.where(land, np.nan, 1), cmap="binary", vmin=0, vmax=1, origin="upper")
oy, ox = np.where(X[fr, 0] != 0)
vals = X[fr, 0][oy, ox]*STD["GESLA"] + MEAN["GESLA"]
sc = axes[1,1].scatter(ox, oy, c=vals, cmap="RdBu_r", s=70, edgecolor="k", linewidth=0.5)
axes[1,1].set_title("(d) sparse GESLA-3 in-situ (m)", fontsize=9)
axes[1,1].set_xticks([]); axes[1,1].set_yticks([])
fig.colorbar(sc, ax=axes[1,1], fraction=0.046, pad=0.04).set_label("m", fontsize=8)
fig.suptitle("Large-surge event (GESLA 2.94 m): inputs, 256\u00d7256, north-up, land grey", fontsize=10)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("temp/figures/fig_densification_inputs.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
print("saved fig_densification_inputs.png")

# ---- Fig 2: coarse GTSM vs generated dense ----
conf = json.load(open("results_sda/real_era5_256/conf.json"))
cfg = SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
model = EDMDataAssimilation(cfg).to("mps"); model.eval()
chk = torch.load("results_sda/real_era5_256/best_sda.pth.tar", map_location="mps")
model.load_state_dict(chk.get("ema") or chk["state_dict"], strict=False)
c = torch.from_numpy(X).float().unsqueeze(0).to("mps")
lead = torch.tensor([float(sd['lead']) if 'lead' in sd else 8.0]).to("mps")
# posterior with current-observation assimilation (realistic dense output);
# the pure prior is under-dispersed because the context encoder global-pools
# the input fields (see notes) -> we show the assimilated dense field.
cur = torch.full((1,1,256,256), float("nan"))
oy, ox = np.where(~np.isnan(y[0]))
for (a,b) in zip(oy, ox): cur[0,0,a,b] = torch.tensor(X[-1,0,a,b])
maskc = (~torch.isnan(cur)).float()
with torch.no_grad():
    prior = model.sample_posterior(c, lead, y=cur.to("mps"), mask=maskc.to("mps"), R=0.1,
                                   steps=40, guidance=1.0, ensemble=4, sigma_max=1.0,
                                   sigma_min=0.002, seed=0,
                                   like_mode="replace", sampler="ode")["mean"][0,0].cpu().numpy()
prior_m = (prior*STD["GTSM"] + MEAN["GTSM"]); prior_m[land] = np.nan
vm = max(float(np.nanpercentile(np.abs(prior_m), 98)), 1.5)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
im = show(axes[0], gtsm*STD["GTSM"]+MEAN["GTSM"], "RdBu_r", -vm, vm, "(a) coarse GTSM input")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04).set_label("m", fontsize=8)
im = show(axes[1], prior_m, "RdBu_r", -vm, vm, "(b) dense surge (SDA-Diff 256\u00b2, assim.)")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label("m", fontsize=8)
fig.suptitle("Densification: coarse GTSM (\u224872 stations) \u2192 dense 256\u00d7256 field (large-surge event, assimilated)", fontsize=10)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig("temp/figures/fig_densification_compare.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
print("saved fig_densification_compare.png")
