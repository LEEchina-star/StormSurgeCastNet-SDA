# -*- coding: utf-8 -*-
"""Realistic-forecast protocols on the 45 held-out gauges (SDA-256, batch=4):
  * prior          : no assimilation                     (pure generation)
  * assim_current  : assimilate the CURRENT obs (last input frame X[:,-1,0])   [Option C]
  * assim_future   : assimilate the TARGET-time obs y    (the previous "cheat")
  * densification  : remove target history from input + no assimilation        [E2]
All MAE in METRES at the target gauge pixel."""
import os, sys, json, numpy as np, torch, time
sys.path.insert(0, os.getcwd())
from util.utils import get_device
from types import SimpleNamespace
from util.model_utils import get_model
device=get_device(); STD=0.16917373301090485

conf=json.load(open("results_sda/real_era5_256/conf.json"))
cfg=SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
m=get_model(cfg).to(device); m.eval()
chk=torch.load("results_sda/real_era5_256/best_sda.pth.tar",map_location=device)
m.load_state_dict(chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk)

d=np.load("cache_sda_full/val.npz")
X=torch.from_numpy(d["X"]).float(); y=torch.from_numpy(d["y"]).float()
B,T,C,H,W=X.shape
mask_f=(~torch.isnan(y)).float(); y_obs_f=torch.where(mask_f>0,y,torch.full_like(y,float("nan")))
lead=torch.full((B,),8.0)
R=0.1

def run(c_override=None, y_in=None, mask_in=None):
    """c_override: replace sparse channel (densification removes target history)."""
    pred=np.zeros((B,H,W),np.float32)
    for i in range(0,B,4):
        idx=slice(i,min(i+4,B))
        c=X[idx].clone()
        if c_override is not None:
            c[:,:,0]=c_override[idx]
        o=m.sample_posterior(c.to(device),lead[idx].to(device),y=y_in[idx].to(device) if y_in is not None else None,
            mask=mask_in[idx].to(device) if mask_in is not None else None,
            R=R,steps=25,guidance=1.0,ensemble=4,sigma_max=1.0,sigma_min=0.002,seed=0,
            like_mode="replace",sampler="ode")
        pred[i:i+4]=o["mean"].cpu()[:,0]
    return pred

# current-obs mask & values: target pixel, value = last input frame (issue-time obs)
cur=torch.full(y.shape, float("nan"))
for i in range(B):
    oy,ox=np.where(~np.isnan(y[i,0]))
    for (a,b) in zip(oy,ox):
        cur[i,0,a,b]=X[i,-1,0,a,b]     # current observation at t0
mask_c=(~torch.isnan(cur)).float()

t0=time.time()
p_pri=run(None,None,None);                             print(f"prior done {time.time()-t0:.0f}s",flush=True)
t0=time.time()
p_cur=run(None,cur,mask_c);                            print(f"assim_current done {time.time()-t0:.0f}s",flush=True)
t0=time.time()
p_fut=run(None,y_obs_f,mask_f);                        print(f"assim_future done {time.time()-t0:.0f}s",flush=True)
# densification: remove target history from input sparse (set 0) + no assimilation
X0=X[:,:,0].clone()
X0[X0!=0]=0.0
t0=time.time()
p_den=run(X0,None,None);                               print(f"densification done {time.time()-t0:.0f}s",flush=True)

def mae(pred):
    out=[]
    for i in range(B):
        mm=mask_f[i,0].numpy()>0
        if mm.any(): out.append(float(np.abs(pred[i][mm]-y[i,0][mm].numpy()).mean()))
    return float(np.mean(out))*STD

print("\n=== SDA-Diff 256 — forecast protocols (45 held-out gauges, metres) ===")
print(f"  prior (no assim)        : {mae(p_pri):.4f} m")
print(f"  assim CURRENT obs [C]   : {mae(p_cur):.4f} m   <- realistic forecast")
print(f"  assim FUTURE obs (old)  : {mae(p_fut):.4f} m   <- previous 'cheat'")
print(f"  densification (E2)      : {mae(p_den):.4f} m")
np.savez("temp/figures/protocols.npz", p_pri=p_pri,p_cur=p_cur,p_fut=p_fut,p_den=p_den)
print("saved protocols.npz")
