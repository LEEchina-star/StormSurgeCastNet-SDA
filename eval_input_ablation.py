# -*- coding: utf-8 -*-
"""E5 input ablation (Ebel Table 4): zero out input channels, evaluate on 45
held-out gauges with the PRIOR (no assimilation) protocol, metres.
Channels: 0 sparse, 1 valid_mask, 2-4 ERA5(msl,u10,v10), 5 GTSM."""
import os, sys, json, numpy as np, torch, time
sys.path.insert(0, os.getcwd())
from types import SimpleNamespace
from util.model_utils import get_model
device="mps"; STD=0.16917373301090485; MEAN=-0.0004041124621210444
CKPT=sys.argv[1] if len(sys.argv)>1 else "results_sda/real_era5_256/best_sda.pth.tar"
CONF=sys.argv[2] if len(sys.argv)>2 else "results_sda/real_era5_256/conf.json"

conf=json.load(open(CONF)); cfg=SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
m=get_model(cfg).to(device); m.eval()
chk=torch.load(CKPT,map_location=device); sd=chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk
try: m.load_state_dict(sd)
except Exception as e: print("strict load failed:", str(e)[:50]); m.load_state_dict(sd, strict=False)

d=np.load("cache_sda_full/val.npz")
X=torch.from_numpy(d["X"]).float(); y=torch.from_numpy(d["y"]).float()
B,T,C,H,W=X.shape; lead=torch.full((B,),8.0)

def run(Xc):
    pred=np.zeros((B,H,W),np.float32)
    for i in range(0,B,4):
        idx=slice(i,min(i+4,B))
        o=m.sample_posterior(Xc[idx].to(device),lead[idx].to(device),y=None,mask=None,R=0.1,
            steps=15,guidance=0.0,ensemble=3,sigma_max=1.0,sigma_min=0.002,seed=0,
            like_mode="replace",sampler="ode")
        pred[i:i+4]=o["mean"].cpu()[:,0]
    return pred

def mae(pred):
    out=[]; yn=y.numpy()
    for i in range(B):
        mm=~np.isnan(yn[i,0])
        if mm.any(): out.append(float(np.abs(pred[i][mm]*STD+MEAN-(yn[i,0][mm]*STD+MEAN)).mean()))
    return float(np.mean(out))

ab={}
X0=X.clone(); X0[:,:,0]=0; X0[:,:,1]=0          # no sparse (reference)
Xg=X.clone(); Xg[:,:,5]=0                        # no GTSM
Xe=X.clone(); Xe[:,:,2:5]=0                      # no ERA5
Xge=X.clone(); Xge[:,:,2:5]=0; Xge[:,:,5]=0      # no GTSM + no ERA5
t0=time.time()
for name,Xc in [("full",X),("no GTSM",Xg),("no ERA5",Xe),("no GTSM+ERA5",Xge),("no sparse",X0)]:
    ab[name]=mae(run(Xc)); print(f"{name}: {ab[name]:.4f} m ({time.time()-t0:.0f}s)",flush=True)

print("\n=== E5 input ablation (prior, 45 held-out gauges, metres) ===")
for k,v in ab.items(): print(f"  {k:16s}: {v:.4f} m")
np.save('temp/figures/input_ablation.npy', ab, allow_pickle=True)
print("saved input_ablation.npy")
