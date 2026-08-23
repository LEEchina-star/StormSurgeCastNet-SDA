# -*- coding: utf-8 -*-
"""Match cache val samples to val.npy (by gauge id + y) to recover td -> target
datetime. Verify reconstruction against combined_gesla_surge.nc.
Then locate OFFICIAL-TEST context gauges in each ROI for the densification eval."""
import sys, os, struct, numpy as np, json
import xarray as xr
sys.path.insert(0, os.getcwd())
from datetime import datetime, timedelta
from util.extract_npy_stream import NumpyStreamingUnpickler
STD=0.16917373301090485; MEAN=-0.0004041124621210444; START=datetime(1993,1,1)

# 1) stream val.npy -> (gid, td, lead, y_norm)
class Cap:
    def __init__(self): self.done=False; self.rows=[]
    def add(self, sample):
        inp,tgt=sample['input'],sample['target']
        y=np.asarray(tgt['sparse']).reshape(-1)
        self.rows.append(dict(gid=int(np.asarray(tgt['id']).reshape(-1)[0]),
                              td=np.asarray(inp['td'],np.float32),
                              lead=float(np.asarray(inp['td_lead']).reshape(-1)[0]),
                              y=float(y[0]) if y.size else float('nan')))
c=Cap()
f=open('Data2/cache/val.npy','rb')   # 注意：需要 Data2/cache（Tier 3）；通常已在 Mac 上生成 official_test_meta.npy，无需重跑
f.read(6); f.read(2); hlen=struct.unpack('<H', f.read(2))[0]; f.read(hlen)
try: NumpyStreamingUnpickler(f,c).load()
except Exception as e: pass
print(f'val.npy streamed {len(c.rows)} rows')

# 2) match cache val samples
d=np.load('cache_sda_full/val.npz')
meta=[]
for i in range(45):
    gid=int(d['ids'][i]); lead=float(d['lead'][i]); cy=float(np.nan_to_num(d['y'][i,0][~np.isnan(d['y'][i,0])][0]) if (~np.isnan(d['y'][i,0])).any() else float('nan'))
    cands=[r for r in c.rows if r['gid']==gid and abs(r['y']-cy)<0.05 if not np.isnan(cy) and not np.isnan(r['y'])]
    if not cands:
        cands=[r for r in c.rows if r['gid']==gid]  # fallback: any row with same gid
    r=cands[0] if cands else None
    meta.append(dict(cache_i=i, gid=gid, lead=lead, y=cy, td=r['td'] if r else None,
                     tdt=(START+timedelta(hours=float(r['td'][-1]+r['lead']))) if r else None))
np.save('temp/figures/official_test_meta.npy', np.array(meta,dtype=object), allow_pickle=True)
ok=0; tot=0
nc=xr.open_dataset('Data2/combined_gesla_surge.nc')
st=nc.station.values; dt=nc.date_time.values
for m in meta:
    if m['tdt'] is None or np.isnan(m['y']): continue
    j=np.where(st==m['gid'])[0]
    if not len(j): continue
    t64=np.datetime64(m['tdt']); idx=np.searchsorted(dt,t64)
    if idx>=len(dt): continue
    s=float(nc.sea_level.isel(station=j[0], date_time=min(idx,len(dt)-1)).values)
    y_m=m['y']*STD+MEAN
    tot+=1
    if abs(s-y_m)<0.06: ok+=1
print(f'datetime verified: {ok}/{tot} cache samples match GESLA at reconstructed time (6cm)')
print('matched cache samples with td:', sum(1 for m in meta if m['td'] is not None), '/45')
