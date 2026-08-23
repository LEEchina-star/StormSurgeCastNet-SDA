# -*- coding: utf-8 -*-
"""Official-protocol evaluation (mirrors test_sda_cache.py: resize=0, steps=25)
but SAVES the full posterior-mean / U-TAE prediction per sample.
Three models: SDA-Diff 256, SDA-Diff 128 (size-agnostic at 256), FiLM U-TAE 128.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
from util.utils import get_device
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace
from util.model_utils import get_model
from models.utae.utae import UTAE

device = get_device(); STEPS = 25; ENS = 4
val = np.load("cache_sda_full/val.npz")
X, y, yg, lead, ids = val["X"], val["y"], val["yg"], val["lead"], val["ids"]
B, T, C, H, W = X.shape

def load_edm(confp, ckptp):
    conf = json.load(open(confp))
    cfg = SimpleNamespace(**{k: v for k, v in conf.items() if not isinstance(v, (dict, list))})
    cfg.model = "edm_da"
    m = get_model(cfg).to(device); m.eval()
    chk = torch.load(ckptp, map_location=device)
    sd = chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk
    m.load_state_dict(sd)
    return m

def load_utae(ckptp):
    m = UTAE(input_dim=6, encoder_widths=[64,64,64,128], decoder_widths=[64,64,64,128],
             out_conv=[2], out_nonlin_mean=False, str_conv_k=4, str_conv_s=2, str_conv_p=1,
             agg_mode="att_group", encoder_norm="group", norm_skip="batch", norm_up="batch",
             decoder_norm="batch", n_head=16, d_model=256, d_k=4,
             encoder=False, return_maps=False, pad_value=0, padding_mode="reflect",
             positional_encoding=True, cond_dim=None, cond_norm_affine=None).to(device)
    m.eval()
    chk = torch.load(ckptp, map_location=device)
    sd = chk.get("ema") or chk.get("model") or chk.get("state_dict") or chk
    m.load_state_dict(sd)
    return m

m256 = load_edm("results_sda/real_era5_256/conf.json", "results_sda/real_era5_256/best_sda.pth.tar")
m128 = load_edm("results_sda/real_era5_128/conf.json", "results_sda/real_era5_128/best_sda.pth.tar")
mut  = load_utae("results_sda/utae_real_era5/best_utae.pth.tar")
print("models loaded")

Xr = torch.from_numpy(X).float()          # [B,T,6,256,256] (resize=0 -> native)
yr = torch.from_numpy(y).float()
mask = (~torch.isnan(yr)).float()
y_obs = torch.where(mask > 0, yr, torch.full_like(yr, float("nan")))
lead_t = torch.full((B,), 8.0)

p256 = np.zeros((B, H, W), np.float32)
p128 = np.zeros((B, H, W), np.float32)
put  = np.zeros((B, H, W), np.float32)
pos = torch.arange(T, dtype=torch.float32)
t0 = time.time()
for i in range(B):
    c = Xr[i][None].to(device); l = lead_t[i][None].to(device)
    yi = y_obs[i][None].to(device); mi = mask[i][None].to(device)
    R = 0.1
    o256 = m256.sample_posterior(c, l, y=yi, mask=mi, R=R, steps=STEPS, guidance=1.0,
                                 ensemble=ENS, sigma_max=1.0, sigma_min=0.002, seed=0,
                                 like_mode="replace", sampler="ode")
    p256[i] = o256["mean"].cpu()[0, 0].numpy()
    o128 = m128.sample_posterior(c, l, y=yi, mask=mi, R=R, steps=STEPS, guidance=1.0,
                                 ensemble=ENS, sigma_max=1.0, sigma_min=0.002, seed=0,
                                 like_mode="replace", sampler="ode")
    p128[i] = o128["mean"].cpu()[0, 0].numpy()
    with torch.no_grad():
        out = mut(c, batch_positions=pos.to(device))[:, :, 0]
    put[i] = out.cpu()[0, 0].numpy()
    if (i+1) % 9 == 0: print(f"  {i+1}/{B}  {time.time()-t0:.0f}s", flush=True)

# ---- per-gauge RMSE sanity vs Table 1 ----
def g_rmse(pred):
    d = []
    for i in range(B):
        mm = mask[i, 0].numpy() > 0
        if mm.any():
            d.append((pred[i][mm] - yr[i, 0][mm].numpy()))
    d = np.concatenate(d); return float(np.sqrt((d**2).mean()))
print(f"gauge RMSE -> SDA256 {g_rmse(p256):.4f} | SDA128 {g_rmse(p128):.4f} | UTAE {g_rmse(put):.4f}  (Table1: 0.099/0.091/1.153)")

# ---- collect dense + gauge pairs ----
obs_d=[]; d256=[]; d128=[]; dut=[]; obs_g=[]; g256=[]; g128=[]; gut=[]
for i in range(B):
    mv = ~np.isnan(yg[i, 0]); obs_d.append(yg[i, 0][mv]); d256.append(p256[i][mv]); d128.append(p128[i][mv]); dut.append(put[i][mv])
    oy, ox = np.where(~np.isnan(y[i, 0]))
    for (a, b) in zip(oy, ox):
        obs_g.append(float(y[i, 0, a, b])); g256.append(float(p256[i, a, b])); g128.append(float(p128[i, a, b])); gut.append(float(put[i, a, b]))
np.savez("temp/figures/preds_official.npz",
         obs_d=np.concatenate(obs_d), p256_d=np.concatenate(d256), p128_d=np.concatenate(d128), put_d=np.concatenate(dut),
         obs_g=np.array(obs_g), p256_g=np.array(g256), p128_g=np.array(g128), put_g=np.array(gut))
print(f"saved preds_official.npz | dense {len(np.concatenate(obs_d))} | gauge {len(obs_g)}")
