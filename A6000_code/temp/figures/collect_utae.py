import sys, os, numpy as np, torch, time
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
from util.utils import get_device
from models.utae.utae import UTAE
device=get_device()
m=UTAE(input_dim=6, encoder_widths=[64,64,64,128], decoder_widths=[64,64,64,128],
       out_conv=[2], out_nonlin_mean=False, str_conv_k=4, str_conv_s=2, str_conv_p=1,
       agg_mode="att_group", encoder_norm="group", norm_skip="batch", norm_up="batch",
       decoder_norm="batch", n_head=16, d_model=256, d_k=4,
       encoder=False, return_maps=False, pad_value=0, padding_mode="reflect",
       positional_encoding=True, cond_dim=None, cond_norm_affine=None).to(device)
m.eval()
chk=torch.load("results_sda/utae_real_era5/best_utae.pth.tar",map_location=device)
sd=chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk
m.load_state_dict(sd)
d=np.load("cache_sda_full/val.npz")
X=torch.from_numpy(d["X"]).float(); y=torch.from_numpy(d["y"]).float()
B,T,C,H,W=X.shape; SZ=128
Xr=F.interpolate(X.reshape(B*T,C,H,W),size=(SZ,SZ),mode="bilinear").reshape(B,T,C,SZ,SZ)
yr=F.interpolate(y.reshape(B,1,H,W),size=(SZ,SZ),mode="nearest").reshape(B,1,SZ,SZ)
mask=(~torch.isnan(yr)).float()
pos=torch.arange(T,dtype=torch.float32).unsqueeze(0).repeat(X.shape[0],1)  # [B,T] like DataLoader collate
put=np.zeros((B,SZ,SZ),np.float32); t0=time.time()
for i in range(0,B,4):
    idx=slice(i,min(i+4,B))
    with torch.no_grad():
        out=m(Xr[idx].to(device),batch_positions=pos[idx].to(device))[:, :, 0]  # [B,1,H,W]
    put[i:i+4]=out[:, 0].cpu().numpy()
    if (i+4)%12==0 or i+4>=B: print(f"  {min(i+4,B)}/{B} ({time.time()-t0:.0f}s)",flush=True)
rms=[]
for bi in range(B):
    mm=mask[bi,0].numpy()>0
    if mm.any():
        rms.append(float(np.sqrt(((put[bi][mm]-yr[bi,0][mm].numpy())**2).mean())))
print(f"U-TAE gauge RMSE (128) = {np.mean(rms):.4f}  (Table: 1.153)")
np.savez("temp/figures/pred_utae.npz", pred=put)   # 128
