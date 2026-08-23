# -*- coding: utf-8 -*-
"""E4: lead-time sensitivity (Ebel Table 3) on the 45 held-out gauges.
Vary L = 4,6,8,10,12 at inference (model trained on L in {0..12}, so in-distribution).
Batch the 5 leads per sample -> 45 sample_posterior calls with lead [5]."""
import os, sys, json, numpy as np, torch, time
sys.path.insert(0, os.getcwd())
from util.utils import get_device
from types import SimpleNamespace
from util.model_utils import get_model
device=get_device()
LEADS=[4,6,8,10,12]

conf=json.load(open("results_sda/real_era5_256/conf.json"))
cfg=SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
m=get_model(cfg).to(device); m.eval()
chk=torch.load("results_sda/real_era5_256/best_sda.pth.tar",map_location=device)
m.load_state_dict(chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk)

d=np.load("cache_sda_full/val.npz")
X=torch.from_numpy(d["X"]).float(); y=torch.from_numpy(d["y"]).float()
B,T,C,H,W=X.shape
mask=(~torch.isnan(y)).float(); y_obs=torch.where(mask>0,y,torch.full_like(y,float("nan")))
lead_b=torch.tensor(LEADS,device=device)
mae={L:[] for L in LEADS}
t0=time.time()
for i in range(B):
    c=X[i][None].repeat(len(LEADS),1,1,1,1).to(device)   # [5,T,6,H,W]
    yi=y_obs[i][None].repeat(len(LEADS),1,1,1).to(device)
    mi=mask[i][None].repeat(len(LEADS),1,1,1).to(device)
    out=m.sample_posterior(c,lead_b,y=None,mask=None,R=0.1,steps=15,guidance=0.0,ensemble=4,
                           sigma_max=1.0,sigma_min=0.002,seed=0,like_mode="replace",sampler="ode")
    means=out["mean"].cpu().numpy()                        # [5,1,H,W]
    mm=mask[i,0].numpy()>0
    for k,L in enumerate(LEADS):
        d_=means[k,0][mm]-y[i,0][mm].numpy()
        mae[L].append(float(np.mean(np.abs(d_))))
    if (i+1)%9==0: print(f"  {i+1}/{B} ({time.time()-t0:.0f}s)",flush=True)
STD=0.16917373301090485
print("\nLead-time sensitivity (PRIOR, no assimilation) (45 held-out gauges, metres):")
for L in LEADS:
    print(f"  L={L:2d}  MAE = {np.mean(mae[L])*STD:.4f} m")
np.savez("temp/figures/lead_prior_sensitivity.npz", leads=np.array(LEADS),
         mae_m=np.array([np.mean(mae[L])*STD for L in LEADS]))
print("saved lead_prior_sensitivity.npz")
