# -*- coding: utf-8 -*-
"""Gulf case v2 (id=528): corrected ROI, ROI-internal gauges, wind-speed heatmap,
RdBu_r fields with physical units, time series & high-res output."""
import numpy as np, glob, torch, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from global_land_mask import globe
import sys; sys.path.insert(0,'.')
from types import SimpleNamespace
from models.edm_da import EDMDataAssimilation

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"]})

# normalization stats
ST = dict(GESLA=0.16917373301090485, GTSM=0.1284419447183609,
          msl=(101002.3515625, 1378.7696533203125),
          u10=4.459522724151611, v10=4.0246148109436035)

# locate sample id=528
for f in sorted(glob.glob("cache_gulf/part_*.npz")):
    d = np.load(f); hit = np.where(d["ids"]==528)[0]
    if len(hit):
        i = hit[0]; X = d["X"][i]; y = d["y"][i]; yg = d["yg"][i]
        lon, lat = float(d["lon"][i]), float(d["lat"][i]); break
print(f"id=528: lon={lon} lat={lat}")

# ROI gauge coords
roi_gauges = []
for f in sorted(glob.glob("cache_gulf/part_*.npz")):
    dd = np.load(f)
    m = (np.abs(dd["lon"]-lon)<3.2) & (np.abs(dd["lat"]-lat)<3.2)
    roi_gauges += list(zip(dd["lon"][m], dd["lat"][m]))
print(f"ROI 内验潮站数: {len(roi_gauges)}")

def inv_surge(a):  return a * ST["GESLA"]       # -> metres
def inv_msl(a):    return (a*ST["msl"][1] + ST["msl"][0])/100.0   # -> hPa
def inv_wind(u,v): return np.sqrt((u*ST["u10"])**2 + (v*ST["v10"])**2)  # -> m/s

# ===== 1) geographic map (lat 18-33 to fit ROI) =====
lon0,lon1,lat0,lat1 = -97,-80, 17, 33
N=320
glon=np.linspace(lon0,lon1,N); glat=np.linspace(lat0,lat1,N)
GLON,GLAT=np.meshgrid(glon,glat); land=globe.is_land(GLAT,GLON)
fig,ax=plt.subplots(figsize=(4.5,4.2))
ax.imshow(land,extent=[lon0,lon1,lat0,lat1],origin="lower",cmap="Greys",vmin=0,vmax=1,alpha=0.5)
ax.imshow(np.where(land,np.nan,1),extent=[lon0,lon1,lat0,lat1],origin="lower",cmap="Blues",vmin=0,vmax=1,alpha=0.35)
ax.contour(glon,glat,land.astype(float),levels=[0.5],colors="k",linewidths=0.7)
# ROI-internal gauges only
for (gx,gy) in roi_gauges: ax.plot(gx,gy,"o",ms=7,color="#3E6B9E",mec="white",mew=0.5,zorder=3)
ax.plot(lon,lat,marker="*",color="red",ms=20,mec="white",mew=0.8,zorder=6)
d_roi=3.2
ax.add_patch(plt.Rectangle((lon-d_roi,lat-d_roi),2*d_roi,2*d_roi,fill=False,ec="#1A5A9E",lw=2,ls="--",zorder=5))
ax.annotate("ROI 256×256 px (0.025°/px)",(lon-d_roi,lat+d_roi),xytext=(5,5),textcoords="offset points",fontsize=7,color="#1A5A9E",va="bottom")
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
ax.set_xlim(lon0,lon1); ax.set_ylim(lat0,lat1); ax.set_aspect("equal")
ax.grid(alpha=0.2,lw=0.4); ax.set_title("Gulf of Mexico — ROI gauges",fontsize=9)
fig.tight_layout(pad=0.3); fig.savefig("temp/figures/gulf2_map.png",dpi=220,bbox_inches="tight",pad_inches=0.05); plt.close(fig)

# ===== 2) input subplots (physical units) =====
fr=-1
fig,axes=plt.subplots(2,2,figsize=(4.6,4.6))
# sparse (GESLA in-situ, m)
axes[0,0].imshow(np.zeros_like(X[fr,0]),cmap="binary",vmin=0,vmax=1,origin="lower")
oy,ox=np.where(X[fr,0]!=0)
sc=axes[0,0].scatter(ox,oy,c=inv_surge(X[fr,0][oy,ox]),cmap="RdBu_r",s=30,edgecolor="k",linewidth=0.4)
axes[0,0].set_title("GESLA-3 in-situ surge (m)",fontsize=8); axes[0,0].set_xticks([]); axes[0,0].set_yticks([])
fig.colorbar(sc,ax=axes[0,0],fraction=0.046,pad=0.04).set_label("m")
# msl (hPa)
im=axes[0,1].imshow(inv_msl(X[fr,2]),cmap="RdBu_r",origin="lower")
axes[0,1].set_title("ERA5 msl (hPa)",fontsize=8); axes[0,1].set_xticks([]); axes[0,1].set_yticks([])
fig.colorbar(im,ax=axes[0,1],fraction=0.046,pad=0.04).set_label("hPa")
# wind speed (m/s)
ws=inv_wind(X[fr,3],X[fr,4])
im=axes[1,0].imshow(ws,cmap="viridis",origin="lower")
axes[1,0].set_title("ERA5 wind speed (m/s)",fontsize=8); axes[1,0].set_xticks([]); axes[1,0].set_yticks([])
fig.colorbar(im,ax=axes[1,0],fraction=0.046,pad=0.04).set_label("m/s")
# gtsm (m, RdBu_r)
im=axes[1,1].imshow(inv_surge(X[fr,5]),cmap="RdBu_r",origin="lower")
axes[1,1].set_title("GTSM surge (m)",fontsize=8); axes[1,1].set_xticks([]); axes[1,1].set_yticks([])
fig.colorbar(im,ax=axes[1,1],fraction=0.046,pad=0.04).set_label("m")
fig.suptitle("Input c=[B,T=12,6,H,W]  (T hourly frames)",fontsize=9,y=1.0)
fig.tight_layout(pad=0.5); fig.savefig("temp/figures/gulf2_inputs.png",dpi=220,bbox_inches="tight",pad_inches=0.05); plt.close(fig)

# ===== 3) denoising process (m) =====
conf=json.load(open("results_sda/real_era5_128/conf.json"))
cfg=SimpleNamespace(**{k:v for k,v in conf.items() if not isinstance(v,(dict,list))}); cfg.model="edm_da"
m=EDMDataAssimilation(cfg).to("mps")
chk=torch.load("results_sda/real_era5_128/best_sda.pth.tar",map_location="mps")
sd=chk.get("ema") or chk.get("model") or chk["state_dict"]; m.load_state_dict(sd); m.eval()
Xr=F.interpolate(torch.from_numpy(X).float().reshape(1,72,256,256),size=(128,128),mode="bilinear").reshape(12,6,128,128).unsqueeze(0)
yg128=F.interpolate(torch.from_numpy(yg).float()[None],size=(128,128),mode="nearest")[0,0]
x0=torch.nan_to_num(yg128,0.0)
sig=torch.tensor([0.8]); noise=torch.randn_like(x0)*0.8; x_t=x0+noise
with torch.no_grad():
    xhat=m.denoise(x_t[None,None].to("mps"),sig.to("mps"),Xr.to("mps"),torch.tensor([8.0]).to("mps"))[0,0].cpu().numpy()
vm=np.nanpercentile(np.abs(inv_surge(x0.numpy())),98)
fig,axes=plt.subplots(1,3,figsize=(4.6,1.8))
for ax,arr,ttl in [(axes[0],x0.numpy(),"clean x\u2080"),(axes[1],x_t.numpy(),"noisy x\u209c=x\u2080+\u03c3\u03b5"),(axes[2],xhat,"denoised x\u0302\u2080")]:
    im=ax.imshow(inv_surge(arr),cmap="RdBu_r",vmin=-vm,vmax=vm,origin="lower"); ax.set_title(ttl,fontsize=8); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im,ax=axes,location="right",fraction=0.046,pad=0.03).set_label("surge (m)")
fig.tight_layout(pad=0.3); fig.savefig("temp/figures/gulf2_denoise.png",dpi=220,bbox_inches="tight",pad_inches=0.05); plt.close(fig)

# ===== 4) assimilation process (m) =====
oy,ox=np.where(~np.isnan(y[0]))
obs_px,obs_py=int(ox[0]*0.5),int(oy[0]*0.5)
mask=torch.zeros(1,1,128,128); mask[0,0,obs_py,obs_px]=1.0
y_obs=torch.full((1,1,128,128),float("nan")); y_obs[0,0,obs_py,obs_px]=float(y[0,oy[0],ox[0]])
lead=torch.tensor([8.0])
prior=m.sample_posterior(Xr.to("mps"),lead.to("mps"),y=None,mask=None,R=0.1,steps=25,guidance=0.0,ensemble=8,sigma_max=1.0,seed=0,like_mode="replace",sampler="ode")["mean"].cpu()[0,0].numpy()
post=m.sample_posterior(Xr.to("mps"),lead.to("mps"),y=y_obs.to("mps"),mask=mask.to("mps"),R=0.1,steps=25,guidance=1.0,ensemble=8,sigma_max=1.0,seed=0,like_mode="replace",sampler="ode")["mean"].cpu()[0,0].numpy()
vm2=np.nanpercentile(np.abs(inv_surge(post)),98)
fig,axes=plt.subplots(1,3,figsize=(4.6,1.8))
for ax,arr,ttl in [(axes[0],prior,"prior (no obs)"),(axes[1],prior,"observation y (GESLA)"),(axes[2],post,"posterior (assimilated)")]:
    im=ax.imshow(inv_surge(arr),cmap="RdBu_r",vmin=-vm2,vmax=vm2,origin="lower"); ax.set_title(ttl,fontsize=8); ax.set_xticks([]); ax.set_yticks([])
axes[1].plot(obs_px,obs_py,"r*",ms=14,mec="white")
fig.colorbar(im,ax=axes,location="right",fraction=0.046,pad=0.03).set_label("surge (m)")
fig.tight_layout(pad=0.3); fig.savefig("temp/figures/gulf2_assim.png",dpi=220,bbox_inches="tight",pad_inches=0.05); plt.close(fig)
print("gulf v2 子图生成完成")
