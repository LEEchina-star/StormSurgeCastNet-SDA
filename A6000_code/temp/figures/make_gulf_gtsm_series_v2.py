# -*- coding: utf-8 -*-
"""Gulf case (id=528): 12-step GTSM surge series with FULL multi-step reverse
diffusion and 2x super-resolution output (256 -> 512).

  (a) input   : clean GTSM surge, 12 hourly frames  (256x256, m)
  (b) noisy   : + Gaussian noise sigma=0.8          (256x256, m)
  (c) denoised: reverse ODE sigma=0.8 -> 0, 2x SR   (512x512, m)

Time labels: t0h ... t11h, top-left = t0h (reading order).
"""
import numpy as np, glob, torch, json, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
sys.path.insert(0, ".")
from types import SimpleNamespace
from models.edm_da import EDMDataAssimilation

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.titlesize": 8, "axes.labelsize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": False,
})
STD_GTSM = 0.1284419447183609; MEAN_GTSM = 0.0009648051927797496
OUT = "temp/figures"; SIGMA0 = 0.8; STEPS = 12; RHO = 7.0; SIGMA_MIN = 0.002

def to_m(a):
    return np.asarray(a, np.float32) * STD_GTSM + MEAN_GTSM

# ---- sample 528 ----
for f in sorted(glob.glob("cache_gulf/part_*.npz")):
    d = np.load(f); i = np.where(d["ids"] == 528)[0][0]
    X = d["X"][i]; break
gtsm = X[:, 5, :, :].astype(np.float32)                    # [12,256,256] normalized
print(f"GTSM input: {gtsm.shape}  normalized [{gtsm.min():.2f},{gtsm.max():.2f}]")

# ---- 256 model ----
conf = json.load(open("results_sda/real_era5_256/conf.json"))
cfg = SimpleNamespace(**{k: v for k, v in conf.items() if not isinstance(v, (dict, list))})
cfg.model = "edm_da"
model = EDMDataAssimilation(cfg).to("mps"); model.eval()
chk = torch.load("results_sda/real_era5_256/best_sda.pth.tar", map_location="mps")
model.load_state_dict(chk.get("ema") or chk["state_dict"])

# context upscaled to 512 (shared by all frames)
c256 = torch.from_numpy(X).float().unsqueeze(0)            # [1,12,6,256,256]
c512 = F.interpolate(c256.reshape(1, 72, 256, 256), size=(512, 512),
                     mode="bilinear").reshape(1, 12, 6, 512, 512).to("mps")
lead = torch.tensor([8.0]).to("mps")

# ---- clean (256) + noisy (256) ----
rng = np.random.default_rng(0)
clean = gtsm                                              # [12,256,256] normalized
noisy = clean + SIGMA0 * rng.standard_normal(clean.shape).astype(np.float32)

# ---- full reverse ODE (EDM Heun, sigma0 -> 0) at 512, one frame at a time ----
def reverse_ode(model, x_t, c, lead, steps=STEPS):
    inv = torch.linspace(0, 1, steps, device=x_t.device)
    sigmas = (SIGMA0 ** (1/RHO) + inv * (SIGMA_MIN ** (1/RHO) - SIGMA0 ** (1/RHO))) ** RHO
    sigmas = torch.cat([sigmas, torch.zeros(1, device=x_t.device)])
    x = x_t
    for i in range(steps):
        sig = sigmas[i].expand(x.shape[0])
        with torch.no_grad():
            x0_hat = model.denoise(x, sig, c, lead)
        d_cur = (x - x0_hat) / sig.view(-1, 1, 1, 1)
        sig_next = sigmas[i + 1]
        dt = sig_next - sigmas[i]
        x_next = x + dt * d_cur
        if i < steps - 1:                                  # Heun 2nd-order correction
            with torch.no_grad():
                x0_hat2 = model.denoise(x_next, sig_next.expand(x.shape[0]), c, lead)
            d_prime = (x_next - x0_hat2) / sig_next.view(-1, 1, 1, 1)
            x_next = x + dt * (0.5 * d_cur + 0.5 * d_prime)
        x = x_next
    return x

deno = np.zeros((12, 512, 512), np.float32)
t0 = time.time()
for t in range(12):
    # clean 512 = bilinear upscale of clean frame; add fresh noise at 512
    clean512 = F.interpolate(torch.from_numpy(clean[t])[None, None],
                             size=(512, 512), mode="bilinear")[0, 0]
    xt = (clean512 + SIGMA0 * torch.randn_like(clean512))[None, None].to("mps")
    deno[t] = reverse_ode(model, xt, c512, lead)[0, 0].cpu().numpy()
    print(f"frame {t:2d}/12 denoised  ({time.time()-t0:.0f}s elapsed)")

# ---- figures (3x4 grids, labels t0h..t11h top-left reading order) ----
allc = to_m(clean)
vmax = max(float(np.nanpercentile(np.abs(allc), 98)), 1e-3)
tlabels = [f"t{h}h" for h in range(12)]                    # t0h .. t11h

def grid_plot(fields, title, fname):
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.4))
    axes = axes.ravel()
    for t, ax in enumerate(axes):
        im = ax.imshow(fields[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(tlabels[t], fontsize=8, pad=1)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(right=0.88, left=0.04, top=0.90, bottom=0.04,
                        wspace=0.12, hspace=0.30)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.72])
    cb = fig.colorbar(im, cax=cbar_ax); cb.set_label("surge (m)", fontsize=9)
    fig.savefig(f"{OUT}/{fname}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig); print(f"saved {fname}")

grid_plot(to_m(clean), "Input GTSM surge \u2014 12 hourly frames (clean, 256\u00d7256)", "gulf_gtsm_input.png")
grid_plot(to_m(noisy), "GTSM surge + Gaussian noise  ($\\sigma$=0.8, 256\u00d7256)", "gulf_gtsm_noisy.png")
grid_plot(to_m(deno),  "Denoised GTSM surge \u2014 reverse diffusion + 2\u00d7 super-resolution (512\u00d7512)",
          "gulf_gtsm_denoised.png")

# ---- sanity ----
def rough(a):
    a = np.asarray(a, np.float32)
    return float(np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean())
print(f"\nroughness: clean={np.mean([rough(clean[t]) for t in range(12)]):.4f} "
      f"noisy={np.mean([rough(noisy[t]) for t in range(12)]):.4f} "
      f"denoised={np.mean([rough(deno[t]) for t in range(12)]):.4f}")
print(f"noisy   RMSE vs clean(256) = {np.sqrt(np.mean((to_m(noisy)-to_m(clean))**2)):.4f} m")
print(f"denoised RMSE vs clean(256 upsampled) = "
      f"{np.sqrt(np.mean((to_m(deno) - to_m(F.interpolate(torch.from_numpy(clean)[:,None],size=(512,512),mode='bilinear')[:,0].numpy()))**2)):.4f} m")
print("ALL GTSM SERIES v2 DONE")
