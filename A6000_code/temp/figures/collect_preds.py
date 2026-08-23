# -*- coding: utf-8 -*-
"""Collect (obs, pred) pairs at dense + gauge pixels for SDA-256 / SDA-128 / U-TAE-128."""
import numpy as np, torch, json, sys, time
import torch.nn.functional as F
sys.path.insert(0, ".")
from types import SimpleNamespace
from models.utae.utae import UTAE

device = ("cuda" if torch.cuda.is_available() else "cpu")
val = np.load("cache_sda_full/val.npz")
X, y, yg, lead = val["X"], val["y"], val["yg"], val["lead"]
B = X.shape[0]; T = X.shape[1]

def load(cfg_path, ckpt_path):
    conf = json.load(open(cfg_path))
    cfg = SimpleNamespace(**{k: v for k, v in conf.items() if not isinstance(v, (dict, list))})
    cfg.model = conf["model"]
    if cfg.model == "utae":
        m = UTAE(input_dim=conf.get("in_dim", 6),
                 encoder_widths=[64,64,64,128], decoder_widths=[64,64,64,128],
                 out_conv=[2], out_nonlin_mean=False, str_conv_k=4, str_conv_s=2, str_conv_p=1,
                 agg_mode="att_group", encoder_norm="group", norm_skip="batch", norm_up="batch",
                 decoder_norm="batch", n_head=16, d_model=256, d_k=4,
                 encoder=False, return_maps=False, pad_value=0, padding_mode="reflect",
                 positional_encoding=True, cond_dim=None, cond_norm_affine=None).to(device)
    else:
        from models.edm_da import EDMDataAssimilation
        m = EDMDataAssimilation(cfg).to(device)
    m.eval()
    chk = torch.load(ckpt_path, map_location=device)
    sd = chk.get("ema") or chk.get("model") or chk["state_dict"]
    try:
        m.load_state_dict(sd)
    except Exception as e:
        print("state_dict mismatch, trying raw:", str(e)[:80])
        m.load_state_dict(chk["state_dict"])
    return m

def sda_mean(model, Xi, yi, lead_i, size):
    c = torch.from_numpy(Xi)[None].float()
    if size != 256:
        c = F.interpolate(c.reshape(1, T*6, 256, 256), size=(size, size), mode="bilinear").reshape(1, T, 6, size, size)
    c = c.to(device); l = torch.tensor([float(lead_i)]).to(device)
    oy, ox = np.where(~np.isnan(yi[0]))
    if size != 256:
        oy, ox = (oy * size // 256), (ox * size // 256)
    mask = torch.zeros(1, 1, size, size); mask[0, 0, oy, ox] = 1.0
    y_obs = torch.full((1, 1, size, size), float("nan")); y_obs[0, 0, oy, ox] = torch.from_numpy(yi[0][oy, ox]).float()
    out = model.sample_posterior(c, l, y=y_obs.to(device), mask=mask.to(device),
                                 R=0.1, steps=20, guidance=1.0, ensemble=4, sigma_max=1.0,
                                 seed=0, like_mode="replace", sampler="ode")
    pred = out["mean"].cpu()[0, 0].numpy()
    if size != 256:
        pred = F.interpolate(torch.from_numpy(pred)[None, None], size=(256, 256), mode="nearest")[0, 0].numpy()
    return pred

pred256 = np.zeros((B, 256, 256), np.float32)
pred128 = np.zeros((B, 256, 256), np.float32)
pred_utae = np.zeros((B, 256, 256), np.float32)
pos = torch.arange(T, dtype=torch.float32)

m256 = load("results_sda/real_era5_256/conf.json", "results_sda/real_era5_256/best_sda.pth.tar")
m128 = load("results_sda/real_era5_128/conf.json", "results_sda/real_era5_128/best_sda.pth.tar")
mut = load("results_sda/cmp_utae_128/conf.json", "results_sda/cmp_utae_128/best_utae.pth.tar")
print("models loaded")

t0 = time.time()
for i in range(B):
    pred256[i] = sda_mean(m256, X[i], y[i], lead[i], 256)
    pred128[i] = sda_mean(m128, X[i], y[i], lead[i], 128)
    # U-TAE at 128
    c = torch.from_numpy(X[i])[None].float()
    c128 = F.interpolate(c.reshape(1, T*6, 256, 256), size=(128, 128), mode="bilinear").reshape(1, T, 6, 128, 128).to(device)
    with torch.no_grad():
        out = mut(c128, batch_positions=pos.to(device))[:, :, 0]  # [1,1,128,128]
    p = out.cpu()[0, 0].numpy()
    pred_utae[i] = F.interpolate(torch.from_numpy(p)[None, None], size=(256, 256), mode="nearest")[0, 0].numpy()
    if (i+1) % 9 == 0:
        print(f"  {i+1}/{B}  {time.time()-t0:.0f}s")

# ---- collect pairs ----
obs_d = []; p256_d = []; p128_d = []; put_d = []
obs_g = []; p256_g = []; p128_g = []; put_g = []
for i in range(B):
    ygv = yg[i, 0]; m = ~np.isnan(ygv)
    obs_d.append(ygv[m].astype(np.float32))
    p256_d.append(pred256[i][m]); p128_d.append(pred128[i][m]); pred_utae[i]
    put_d.append(pred_utae[i][m])
    oy, ox = np.where(~np.isnan(y[i, 0]))
    for (a, b) in zip(oy, ox):
        obs_g.append(float(y[i, 0, a, b])); p256_g.append(float(pred256[i, a, b]))
        p128_g.append(float(pred128[i, a, b])); put_g.append(float(pred_utae[i, a, b]))
out = dict(obs_d=np.concatenate(obs_d), p256_d=np.concatenate(p256_d),
           p128_d=np.concatenate(p128_d), put_d=np.concatenate(put_d),
           obs_g=np.array(obs_g), p256_g=np.array(p256_g),
           p128_g=np.array(p128_g), put_g=np.array(put_g))
np.savez("temp/figures/preds_3models.npz", **out)
print(f"saved preds_3models.npz | dense {len(out['obs_d'])} pts | gauge {len(out['obs_g'])} pts")
