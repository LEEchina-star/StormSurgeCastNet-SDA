# -*- coding: utf-8 -*-
"""Collect (obs, pred) pairs for the 3 Table-1 models at the 256 grid.
SDA-256/SDA-128: from official-protocol batch=4 runs (pred256/pred128, RMSE 0.0985/0.0914).
U-TAE: forward at 256 with [B,T] positions (~1.05, matching 1.153 protocol)."""
import numpy as np, torch, os, sys
sys.path.insert(0, os.getcwd())
from models.utae.utae import UTAE
device="mps"

p256 = np.load("temp/figures/pred256/pred.npz")["pred"]   # [45,1,256,256]
p128 = np.load("temp/figures/pred128/pred.npz")["pred"]
d = np.load("cache_sda_full/val.npz")
X = torch.from_numpy(d["X"]).float(); y, yg = d["y"], d["yg"]
B, T, C, H, W = X.shape

# ---- U-TAE at 256 ([B,T] positions) ----
m = UTAE(input_dim=6, encoder_widths=[64,64,64,128], decoder_widths=[64,64,64,128],
         out_conv=[2], out_nonlin_mean=False, str_conv_k=4, str_conv_s=2, str_conv_p=1,
         agg_mode="att_group", encoder_norm="group", norm_skip="batch", norm_up="batch",
         decoder_norm="batch", n_head=16, d_model=256, d_k=4,
         encoder=False, return_maps=False, pad_value=0, padding_mode="reflect",
         positional_encoding=True, cond_dim=None, cond_norm_affine=None).to(device)
m.eval()
chk = torch.load("results_sda/utae_real_era5/best_utae.pth.tar", map_location=device)
m.load_state_dict(chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk)
pos = torch.arange(T, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
put = np.zeros((B, H, W), np.float32)
with torch.no_grad():
    for i in range(0, B, 4):
        out = m(X[i:i+4].to(device), batch_positions=pos[i:i+4].to(device))[:, :, 0]
        put[i:i+4] = out[:, 0].cpu().numpy()
print("U-TAE 256 done")

# ---- collect pairs ----
obs_d=[]; a256=[]; a128=[]; aut=[]; obs_g=[]; g256=[]; g128=[]; gut=[]
for i in range(B):
    mv = ~np.isnan(yg[i, 0])
    obs_d.append(yg[i, 0][mv]); a256.append(p256[i, 0][mv]); a128.append(p128[i, 0][mv]); aut.append(put[i][mv])
    oy, ox = np.where(~np.isnan(y[i, 0]))
    for (a, b) in zip(oy, ox):
        obs_g.append(float(y[i, 0, a, b])); g256.append(float(p256[i, 0, a, b]))
        g128.append(float(p128[i, 0, a, b])); gut.append(float(put[i, a, b]))
def rmse(x, yv): return float(np.sqrt(np.mean((np.asarray(x)-np.asarray(yv))**2)))
print(f"gauge RMSE -> SDA256 {rmse(g256,obs_g):.4f} | SDA128 {rmse(g128,obs_g):.4f} | UTAE {rmse(gut,obs_g):.4f}")
np.savez("temp/figures/preds_final.npz",
         obs_d=np.concatenate(obs_d), p256_d=np.concatenate(a256), p128_d=np.concatenate(a128), put_d=np.concatenate(aut),
         obs_g=np.array(obs_g), p256_g=np.array(g256), p128_g=np.array(g128), put_g=np.array(gut))
print(f"saved preds_final.npz | dense {len(np.concatenate(obs_d))} | gauge {len(obs_g)}")
