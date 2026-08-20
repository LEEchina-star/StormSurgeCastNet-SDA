# -*- coding: utf-8 -*-
"""Dense scatter collection: subset of cache_sda_real256 (all-ROI gauge obs,
~10 gauges/sample). 250 samples -> ~2500 gauge pixels. 3 models at 256 grid."""
import os, sys, json, glob, numpy as np, torch, time
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
from types import SimpleNamespace
from util.model_utils import get_model
from models.utae.utae import UTAE
device="mps"

# load all shards
Xs,Ys,YGs,Is=[],[],[],[]
for f in sorted(glob.glob("cache_sda_real256/part_*.npz")):
    d=np.load(f); Xs.append(d["X"]); Ys.append(d["y"]); YGs.append(d["yg"]); Is.append(d["ids"])
X=np.concatenate(Xs); y=np.concatenate(Ys); yg=np.concatenate(YGs); ids=np.concatenate(Is)
print("train cache:", X.shape, "unique gauges:", len(np.unique(ids)))
rng=np.random.default_rng(0); idx=np.sort(rng.choice(len(X), 250, replace=False))
X,y,yg=X[idx],y[idx],yg[idx]
B,T,C,H,W=X.shape
print("subset:", B, "samples; gauge px total:", int((~np.isnan(y)).sum()))

def load_edm(confp,ckp):
    conf=json.load(open(confp)); cfg=SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
    m=get_model(cfg).to(device); m.eval()
    chk=torch.load(ckp,map_location=device); sd=chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk
    m.load_state_dict(sd); return m
def load_utae(ckp):
    m=UTAE(input_dim=6,encoder_widths=[64,64,64,128],decoder_widths=[64,64,64,128],out_conv=[2],
           out_nonlin_mean=False,str_conv_k=4,str_conv_s=2,str_conv_p=1,agg_mode="att_group",
           encoder_norm="group",norm_skip="batch",norm_up="batch",decoder_norm="batch",
           n_head=16,d_model=256,d_k=4,encoder=False,return_maps=False,pad_value=0,
           padding_mode="reflect",positional_encoding=True,cond_dim=None,cond_norm_affine=None).to(device)
    m.eval(); chk=torch.load(ckp,map_location=device)
    m.load_state_dict(chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk); return m

m256=load_edm("results_sda/real_era5_256/conf.json","results_sda/real_era5_256/best_sda.pth.tar")
m128=load_edm("results_sda/real_era5_128/conf.json","results_sda/real_era5_128/best_sda.pth.tar")
mut=load_utae("results_sda/utae_real_era5/best_utae.pth.tar")
Xr=torch.from_numpy(X).float(); yr=torch.from_numpy(y).float()
mask=(~torch.isnan(yr)).float(); y_obs=torch.where(mask>0,yr,torch.full_like(yr,float("nan")))
lead=torch.full((B,),8.0); pos=torch.arange(T,dtype=torch.float32).unsqueeze(0).repeat(B,1)
p256=np.zeros((B,H,W),np.float32); p128=np.zeros((B,H,W),np.float32); put=np.zeros((B,H,W),np.float32)
t0=time.time()
for i in range(0,B,4):
    idx2=slice(i,min(i+4,B))
    o=m256.sample_posterior(Xr[idx2].to(device),lead[idx2].to(device),y=y_obs[idx2].to(device),
        mask=mask[idx2].to(device),R=0.1,steps=20,guidance=1.0,ensemble=4,sigma_max=1.0,
        sigma_min=0.002,seed=0,like_mode="replace",sampler="ode")
    p256[i:i+4]=o["mean"].cpu()[:,0]
    o=m128.sample_posterior(Xr[idx2].to(device),lead[idx2].to(device),y=y_obs[idx2].to(device),
        mask=mask[idx2].to(device),R=0.1,steps=20,guidance=1.0,ensemble=4,sigma_max=1.0,
        sigma_min=0.002,seed=0,like_mode="replace",sampler="ode")
    p128[i:i+4]=o["mean"].cpu()[:,0]
    with torch.no_grad():
        out=mut(Xr[idx2].to(device),batch_positions=pos[idx2].to(device))[:,:,0]
    put[i:i+4]=out[:,0].cpu()
    if (i+4)%24==0 or i+4>=B: print(f"  {min(i+4,B)}/{B} ({time.time()-t0:.0f}s)",flush=True)

# collect gauge pixels
obs=[]; a256=[]; a128=[]; aut=[]
for i in range(B):
    oy,ox=np.where(~np.isnan(y[i,0]))
    for (a,b) in zip(oy,ox):
        obs.append(float(y[i,0,a,b])); a256.append(float(p256[i,a,b])); a128.append(float(p128[i,a,b])); aut.append(float(put[i,a,b]))
obs=np.array(obs); a256=np.array(a256); a128=np.array(a128); aut=np.array(aut)
STD=0.16917373301090485
for nm,p in [("SDA256",a256),("SDA128",a128),("UTAE",aut)]:
    print(f"  {nm}: N={len(obs)} MAE={np.mean(np.abs(p-obs))*STD:.4f} m  bias={np.mean(p-obs)*STD:+.4f} m")
np.savez("temp/figures/preds_dense.npz", obs_g=obs, p256_g=a256, p128_g=a128, put_g=aut)
print("saved preds_dense.npz |", len(obs), "gauge pixels")
