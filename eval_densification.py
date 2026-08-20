# -*- coding: utf-8 -*-
"""E2 densification (Ebel protocol): target gauge REMOVED from input (history
zeroed + valid mask cleared), no assimilation; model must densify the field at a
held-out (official-val) location from neighbours/ERA5/GTSM only.
Compares: hyperlocal (target history in input, no assim) vs densification.
Run with the CURRENT model; re-run with real_era5_256_v2 after retraining."""
import os, sys, json, numpy as np, torch, time
sys.path.insert(0, os.getcwd())
from types import SimpleNamespace
from util.model_utils import get_model
device="mps"; STD=0.16917373301090485
CKPT=sys.argv[1] if len(sys.argv)>1 else "results_sda/real_era5_256/best_sda.pth.tar"
CONF=sys.argv[2] if len(sys.argv)>2 else "results_sda/real_era5_256/conf.json"

conf=json.load(open(CONF))
cfg=SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
m=get_model(cfg).to(device); m.eval()
chk=torch.load(CKPT,map_location=device)
sd=chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk
try: m.load_state_dict(sd)
except Exception as e:
    print("strict load failed (new sparse path), loading strict=False:", str(e)[:60]); m.load_state_dict(sd, strict=False)

d=np.load("cache_sda_full/val.npz")
X=torch.from_numpy(d["X"]).float(); y=torch.from_numpy(d["y"]).float(); ids=d["ids"]
B,T,C,H,W=X.shape
lead=torch.full((B,),8.0)

def run(Xc, assim=None):
    pred=np.zeros((B,H,W),np.float32)
    mask_cur=None; y_cur=None
    if assim=="current":
        cur=torch.full(y.shape,float("nan"))
        for i in range(B):
            oy,ox=np.where(~np.isnan(y[i,0]))
            for (a,b) in zip(oy,ox): cur[i,0,a,b]=Xc[i,-1,0,a,b]
        mask_cur=(~torch.isnan(cur)).float(); y_cur=cur
    for i in range(0,B,4):
        idx=slice(i,min(i+4,B))
        o=m.sample_posterior(Xc[idx].to(device),lead[idx].to(device),
            y=y_cur[idx].to(device) if y_cur is not None else None,
            mask=mask_cur[idx].to(device) if mask_cur is not None else None,
            R=0.1,steps=20,guidance=1.0 if assim else 0.0,ensemble=4,
            sigma_max=1.0,sigma_min=0.002,seed=0,like_mode="replace",sampler="ode")
        pred[i:i+4]=o["mean"].cpu()[:,0]
    return pred

yn = y.numpy()  # [B,1,256,256]
def mae(pred):
    out=[]
    for i in range(B):
        mm = ~np.isnan(yn[i,0])
        if mm.any():
            pv = pred[i][mm].astype(np.float64)
            ov = yn[i,0][mm].astype(np.float64)
            out.append(float(np.abs(pv-ov).mean()))
    return float(np.mean(out))*STD

# hyperlocal: keep target history, no assimilation
Xh=X.clone()
# densification: zero target-gauge history + valid mask at the target pixel
Xd=X.clone()
for i in range(B):
    oy,ox=np.where(~np.isnan(y[i,0]))
    for (a,b) in zip(oy,ox):
        Xd[i,:,0,a,b]=0.0; Xd[i,:,1,a,b]=0.0

t0=time.time()
p_hyper=run(Xh);            print(f"hyperlocal done {time.time()-t0:.0f}s",flush=True)
t0=time.time()
p_den=run(Xd);              print(f"densification done {time.time()-t0:.0f}s",flush=True)
t0=time.time()
p_den_cur=run(Xd,"current"); print(f"densification+current-obs done {time.time()-t0:.0f}s",flush=True)

print("\n=== E2 densification (45 official-val held-out gauges, metres) ===")
print(f"  hyperlocal (history in input)     : {mae(p_hyper):.4f} m")
print(f"  densification (history removed)   : {mae(p_den):.4f} m")
print(f"  densification + assim current obs : {mae(p_den_cur):.4f} m")
print(f"  ckpt: {CKPT.split('/')[-2]}")
np.savez("temp/figures/densification.npz", p_hyper=p_hyper,p_den=p_den,p_den_cur=p_den_cur)
print("saved densification.npz")
