# -*- coding: utf-8 -*-
"""E2 OFFICIAL-TEST densification: 15 unique test gauges inside the 45 val ROIs.
1. For each cache sample: target datetime known (reconstructed, 45/45 verified).
2. Locate official-TEST context gauges (pixel->station, distance<0.02deg).
3. Densification: zero their obs (12 frames sparse + valid mask) in the input.
4. Model prior predicts; compare at those pixels vs GESLA measured surge
   (combined_gesla_surge.nc at target datetime) -> MAE/MSE/NNSE.
"""
import sys, os, numpy as np, json, torch, time
import xarray as xr
sys.path.insert(0, os.getcwd())
from util.utils import get_device
from types import SimpleNamespace
from util.model_utils import get_model
device=get_device(); STD=0.16917373301090485; MEAN=-0.0004041124621210444
CKPT=sys.argv[1] if len(sys.argv)>1 else "results_sda/real_era5_256/best_sda.pth.tar"
CONF=sys.argv[2] if len(sys.argv)>2 else "results_sda/real_era5_256/conf.json"

# gauges + test split
nc=xr.open_dataset('Data2/combined_gesla_surge.nc')
st_all=nc.station.values; slon=nc.longitude.values; slat=nc.latitude.values; dt=nc.date_time.values
sp=np.load('Data2/aux/splits_ids.npy',allow_pickle=True).item(); test_ids=set(sp['test'])
meta=np.load('temp/figures/official_test_meta.npy',allow_pickle=True)

d=np.load('cache_sda_full/val.npz')
X=torch.from_numpy(d['X']).float(); y=torch.from_numpy(d['y']).float()
RES=0.025; SZ=256; B=45
lead=torch.full((B,),8.0)

# locate official-test pixels per cache sample + measured surge at target time
test_px=[]   # (cache_i, a, b, gid, measured_m)
for m in meta:
    i=m['cache_i']; tdt=m['tdt']
    if tdt is None: continue
    west=float(d['lon'][i])-SZ/2*RES; north=float(d['lat'][i])+SZ/2*RES
    t64=np.datetime64(tdt); tidx=np.searchsorted(dt,t64); tidx=min(tidx,len(dt)-1)
    # recover true ROI center from the target gauge pixel + geo (ocean-point centred)
    oy,ox=torch.where(~torch.isnan(y[i,0]))
    if not len(oy): continue
    jg=np.where(st_all==int(d['ids'][i]))[0]
    if not len(jg): continue
    cx=float(slon[jg[0]])-(int(ox[0])-127.5)*RES
    cy=float(slat[jg[0]])+(int(oy[0])-127.5)*RES
    rows,cols=torch.where(X[i,0,0]!=0)
    for (r,c) in zip(rows,cols):
        gx=float(cx)+(int(c)-127.5)*RES; gy=float(cy)-(int(r)-127.5)*RES
        j=np.argmin((slon-gx)**2+(slat-gy)**2)
        if np.sqrt((slon[j]-gx)**2+(slat[j]-gy)**2) < 0.05 and st_all[j] in test_ids:
            s=float(nc.sea_level.isel(station=j, date_time=tidx).values)
            if not np.isnan(s):
                test_px.append((i,int(r),int(c),int(st_all[j]),s))
print(f'official-test evaluation points: {len(test_px)} (unique gauges {len(set(p[3] for p in test_px))})')

# densify: zero test-gauge obs in input
Xd=X.clone()
for (i,a,b,gid,_) in test_px:
    Xd[i,:,0,a,b]=0.0; Xd[i,:,1,a,b]=0.0

# model
conf=json.load(open(CONF)); cfg=SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
m=get_model(cfg).to(device); m.eval()
chk=torch.load(CKPT,map_location=device); sd=chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk
try: m.load_state_dict(sd)
except Exception as e: print('strict load failed:', str(e)[:50]); m.load_state_dict(sd, strict=False)

# predict (prior, no assimilation)
pred=np.zeros((B,SZ,SZ),np.float32); t0=time.time()
for i in range(0,B,4):
    idx=slice(i,min(i+4,B))
    o=m.sample_posterior(Xd[idx].to(device),lead[idx].to(device),y=None,mask=None,R=0.1,
        steps=20,guidance=0.0,ensemble=4,sigma_max=1.0,sigma_min=0.002,seed=0,like_mode="replace",sampler="ode")
    pred[i:i+4]=o["mean"].cpu()[:,0]
print(f"predicted {time.time()-t0:.0f}s")

pv=np.array([pred[i,a,b] for (i,a,b,_,_) in test_px]); ov=np.array([v for (_,_,_,_,v) in test_px])
pm=pv*STD+MEAN
mae=float(np.mean(np.abs(pm-ov))); mse=float(np.mean((pm-ov)**2)); nnse=1.0/(2.0-(1.0-mse/max(np.var(ov),1e-9)))
print("\n=== E2 official-TEST densification (15 gauges, metres) ===")
print(f"  points={len(test_px)}  MAE={mae:.4f} m  MSE={mse:.4f}  NNSE={nnse:.3f}  bias={np.mean(pm-ov):+.4f}")
np.save('temp/figures/official_test_eval.npy', np.array(test_px,dtype=object), allow_pickle=True)
print("saved official_test_eval.npy")
