# -*- coding: utf-8 -*-
"""Smoke test for the SPATIAL conditioning fix in ContextEncoder/EDMDataAssimilation.

Diagnostics:
  1. model builds (denoiser input = 1 noisy + 6 raw context channels)
  2. c_agg carries the GTSM bump at the correct location
  3. FiLM params react to the GTSM structure (not just its global mean)
  4. denoiser output reacts to WHERE the bump is (spatially localised diff)
"""
import os, sys, math
import numpy as np
import torch
torch.manual_seed(0)
sys.path.insert(0, os.getcwd())
from types import SimpleNamespace
from util.model_utils import get_model

cfg = SimpleNamespace(model="edm_da", in_dim=6, diff_in_ch=1, diff_out_ch=1,
                      diff_dim=64, diff_levels=3, diff_emb_dim=256, diff_attn_level=2,
                      diff_dropout=0.0, sigma_data=1.2, p_mean=-1.2, p_std=1.2,
                      sigma_max=1.0, sigma_min=0.002, device="cpu")
m = get_model(cfg)
m.eval()
nparams = sum(p.numel() for p in m.parameters())
print(f"model built OK, params = {nparams/1e6:.2f}M")

B, T, C, H, W = 1, 12, 6, 256, 256

def make_c(bump_rc=None):
    c = torch.zeros(B, T, C, H, W)
    c[:, :, 2] = 0.0      # ERA5 msl (dummy)
    c[:, :, 3:5] = 0.0    # ERA5 wind
    c[:, :, 5] = 0.5      # GTSM baseline surge
    if bump_rc is not None:
        r, cc = bump_rc
        ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing="ij")
        d2 = (ys - r) ** 2 + (xs - cc) ** 2
        c[:, :, 5] += 2.5 * torch.exp(-d2 / (2 * 20.0 ** 2))   # +2.5 m peak
    # sparse gauges: a few pixels in ch0 (values ~1.0), valid mask ch1
    for (a, b) in [(100, 100), (150, 160), (120, 200)]:
        c[:, :, 0, a, b] = 1.0
        c[:, :, 1, a, b] = 1.0
    return c

cA = make_c(None)                 # flat GTSM
cB = make_c((64, 192))            # bump NE
cC = make_c((192, 64))            # bump SW
lead = torch.tensor([8.0])

with torch.no_grad():
    ctxA, aggA = m.context_encoder(cA, torch.zeros(B))
    ctxB, aggB = m.context_encoder(cB, torch.zeros(B))
    ctxC, aggC = m.context_encoder(cC, torch.zeros(B))

# --- 2. c_agg carries the bump at the right location ---
gtsm_agg = aggB[0, 5].numpy()           # [256,256]
peak_r, peak_c = np.unravel_index(np.argmax(gtsm_agg), gtsm_agg.shape)
print(f"\n[2] c_agg GTSM channel: bump at ({peak_r},{peak_c}), expected ~(64,192); "
      f"peak value {gtsm_agg.max():.3f} (input peak 3.0)")
assert abs(peak_r - 64) <= 3 and abs(peak_c - 192) <= 3, "c_agg lost bump location!"

# --- 3. FiLM params react to structure ---
dBA = sum(abs(a[0].squeeze() - b[0].squeeze()).sum().item() +
          abs(a[1].squeeze() - b[1].squeeze()).sum().item()
          for a, b in zip(ctxB, ctxA))
dBC = sum(abs(a[0].squeeze() - b[0].squeeze()).sum().item() +
          abs(a[1].squeeze() - b[1].squeeze()).sum().item()
          for a, b in zip(ctxB, ctxC))
print(f"[3] FiLM diff (flat vs bump NE) = {dBA:.3f}  (bump NE vs bump SW) = {dBC:.3f}")
assert dBA > 1e-3, "FiLM still insensitive to spatial structure!"
assert dBC > 1e-3, "FiLM does not react to bump LOCATION!"

# --- 4. denoiser output reacts to WHERE the bump is ---
x_t = torch.randn(B, 1, H, W) * 0.5
sig = torch.tensor([0.5])
with torch.no_grad():
    oA = m.denoise(x_t, sig, cA, lead)
    oB = m.denoise(x_t, sig, cB, lead)
    oC = m.denoise(x_t, sig, cC, lead)
diff_BA = (oB - oA).abs().squeeze().numpy()
diff_BC = (oB - oC).abs().squeeze().numpy()
dr, dc = np.unravel_index(np.argmax(diff_BC), diff_BC.shape)
print(f"[4] denoiser |out(B)-out(A)| max = {diff_BA.max():.4f}, "
      f"|out(B)-out(C)| max = {diff_BC.max():.4f} at ({dr},{dc})")
print(f"    B-C diff is localised: frac of mass within 40px of bump centres = "
      f"{diff_BC[24:104, 152:232].sum()/diff_BC.sum():.2f} (NE) + "
      f"{diff_BC[152:232, 24:104].sum()/diff_BC.sum():.2f} (SW)")
assert diff_BC.max() > 1e-3, "denoiser output does not react to bump location!"
assert diff_BA.max() > 1e-3, "denoiser output does not react to presence of bump!"

# --- 5. training step runs (denoise_loss) ---
x0 = torch.zeros(B, 1, H, W)
x0[0, 0, 100, 100] = 1.2; x0[0, 0, 150, 160] = 0.8
mask = (x0 != 0).float()
loss = m.denoise_loss(x0, cB, lead, mask)
loss.backward()
g_norm = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
print(f"[5] denoise_loss = {loss.item():.4f}, grad norm = {g_norm:.2f}")
assert torch.isfinite(loss) and g_norm > 0

print("\nALL SPATIAL-CONDITIONING SMOKE TESTS PASSED")
