# -*- coding: utf-8 -*-
"""
SDA-Diff training on the COMPACT real-data cache (cache_sda_full/{train,val}.npz),
which holds 256x256 / T=12 samples with REAL ERA5 -- identical inputs to the
original StormSurgeCastNet pipeline (built from Data2/cache/val.npy).

Gauge-level 70/30 split is already applied by util/build_compact_cache.py.

Usage:
    python train_sda_cache.py --cache_dir cache_sda_full --resize 128 \
        --epochs 30 --diff_dim 32 --batch_size 4 --ensemble 4 --sample_steps 15
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
from parse_args import create_parser
from util import utils, metrics as mets
from util.model_utils import get_model

def seed_packages(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_cache(cache_dir, split):
    path = os.path.join(cache_dir, f"{split}.npz")
    if os.path.isfile(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    # multi-shard cache: cache_dir/part_*.npz (streaming extraction output)
    import glob
    shards = sorted(glob.glob(os.path.join(cache_dir, "part_*.npz")))
    if shards and split == "train":
        print(f"loading {len(shards)} shards from {cache_dir} ...")
        Xs, Ys, YGs, Ls, Is = [], [], [], [], []
        for f in shards:
            d = np.load(f)
            Xs.append(d["X"]); Ys.append(d["y"]); YGs.append(d["yg"])
            Ls.append(d["lead"]); Is.append(d["ids"])
        return dict(X=np.concatenate(Xs), y=np.concatenate(Ys), yg=np.concatenate(YGs),
                    lead=np.concatenate(Ls), ids=np.concatenate(Is))
    raise FileNotFoundError(f"no cache found at {path} or {cache_dir}/part_*.npz")

class CompactDS(torch.utils.data.Dataset):
    def __init__(self, d):
        self.X = torch.from_numpy(d["X"]); self.y = torch.from_numpy(d["y"])
        self.yg = torch.from_numpy(d["yg"]); self.lead = torch.from_numpy(d["lead"])
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i], self.yg[i], self.lead[i]

def dense_target(y, yg):
    """Dense diffusion target: GTSM field, overwritten by gauge obs where available."""
    x0 = yg.clone()
    obs = (~torch.isnan(y)).float()
    x0 = torch.where(obs > 0, torch.nan_to_num(y, 0.0), x0)
    return x0, obs

def main():
    parser = create_parser(mode="train")
    parser.add_argument("--resize", type=int, default=0, help="bilinear resize spatial dims (128 for quick runs)")
    parser.add_argument("--val_cache_dir", default="", help="separate cache dir for validation (default: same as --cache_dir)")
    parser.add_argument("--out", default="results_sda/cache_compare")
    config = utils.str2list(parser.parse_args(), list_args=["encoder_widths", "decoder_widths", "out_conv"])
    config.model = "edm_da"   # this script always trains the SDA-Diff model
    seed_packages(config.rdm_seed)
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        config.device = "mps"
    device = torch.device(config.device)
    print(f"device: {device}")

    # ---- data ----
    val_dir = config.val_cache_dir if config.val_cache_dir else config.cache_dir
    tr, va = load_cache(config.cache_dir, "train"), load_cache(val_dir, "val")
    if config.resize:
        for d in (tr, va):
            h = d["X"].shape[-2] // 2 if False else None
            X = torch.from_numpy(d["X"]).float()
            B, T, C, H, W = X.shape
            Xr = F.interpolate(X.reshape(B * T, C, H, W), size=(config.resize, config.resize), mode="bilinear")
            d["X"] = Xr.reshape(B, T, C, config.resize, config.resize).numpy()
            # NOTE: targets are sparse/NaN-heavy -- use NEAREST to preserve gauge pixels
            y = torch.from_numpy(d["y"]).float()
            yr = F.interpolate(y.reshape(B, 1, H, W), size=(config.resize, config.resize), mode="nearest")
            d["y"] = yr.reshape(B, 1, config.resize, config.resize).numpy()
            yg = torch.from_numpy(d["yg"]).float()
            ygr = F.interpolate(yg.reshape(B, 1, H, W), size=(config.resize, config.resize), mode="nearest")
            d["yg"] = ygr.reshape(B, 1, config.resize, config.resize).numpy()
    print(f"train X {tr['X'].shape} | val X {va['X'].shape}")

    tr_loader = torch.utils.data.DataLoader(CompactDS(tr), batch_size=config.batch_size, shuffle=True, num_workers=0)
    va_loader = torch.utils.data.DataLoader(CompactDS(va), batch_size=config.batch_size, shuffle=False, num_workers=0)
    # dense_target needs yg also resized (already handled above since y/yg resized together)

    # ---- model ----
    config.in_dim = tr["X"].shape[2]
    model = get_model(config).to(device)
    ema = get_model(config).to(device)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config.gamma)
    start_epoch = 1
    last_ckpt = os.path.join(config.out, "last.pth.tar")
    if os.path.isfile(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        print(f"resume from epoch {start_epoch} (last.pth.tar)")
    print(f"TOTAL PARAMS: {utils.get_ntrainparams(model)}")

    os.makedirs(config.out, exist_ok=True)
    with open(os.path.join(config.out, "conf.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    def ema_update(decay):
        with torch.no_grad():
            for a, b in zip(ema.parameters(), model.parameters()):
                a.mul_(decay).add_(b.detach(), alpha=1 - decay)

    def train_step(X, y, yg, lead):
        X, y, yg, lead = X.to(device), y.to(device), yg.to(device), lead.to(device)
        x0, obs = dense_target(y, yg)
        # DENSE supervision mask: all valid pixels of the dense target (ocean +
        # gauge pixels), not just the single target-gauge pixel
        mask = (~torch.isnan(x0)).float()
        loss = model.denoise_loss(x0, X, lead, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(config.ema_decay)
        return float(loss.item())

    @torch.no_grad()
    def validate():
        m = ema
        steps = max(config.sample_steps, 25); ens = min(config.ensemble, 2)
        sample_dev = None
        like_mode = "replace"   # first-order SDA guidance (same as official test protocol)
        crps, cov, rmse = [], [], []
        n_val = 0
        for X, y, yg, lead in va_loader:
            if n_val >= 16:   # keep validation sampling cheap on CPU
                break
            n_val += X.shape[0]
            X, y, yg, lead = X.to(device), y.to(device), yg.to(device), lead.to(device)
            mask = (~torch.isnan(y)).float()   # target-gauge scoring mask (unchanged)
            # 12h-WINDOW assimilation: EVERY frame of the past-12h observation
            # series enters the likelihood (per-gauge window mean over valid
            # frames); target-time obs are NEVER used (no future leakage).
            sp = X[:, :, 0]                     # [B,T,H,W] sparse obs series (ch0)
            vm = X[:, :, 1].clamp(0, 1)         # [B,T,H,W] validity (ch1)
            y_win = torch.full_like(y, float("nan"))
            for bi in range(X.shape[0]):
                oy, ox = torch.where(~torch.isnan(y[bi, 0]))
                for (a, b) in zip(oy, ox):
                    vals = sp[bi, :, a, b][vm[bi, :, a, b] > 0]
                    if vals.numel():
                        y_win[bi, 0, a, b] = vals.mean()
            mask_win = (~torch.isnan(y_win)).float()
            out = m.sample_posterior(X, lead, y=y_win, mask=mask_win, R=max(config.obs_noise, 1e-3),
                                     steps=steps, guidance=config.sda_guidance, ensemble=ens,
                                     sigma_max=config.sigma_max, sigma_min=config.sigma_min, seed=0,
                                     device=sample_dev, like_mode=like_mode, sampler="ode")
            mean, samples = out["mean"].cpu().numpy(), out["samples"].cpu().numpy()
            y_np, m_np = y.cpu().numpy(), mask.cpu().numpy()
            for bi in range(mean.shape[0]):
                crps.append(mets.crps_ensemble(samples[bi], y_np[bi], m_np[bi]))
                cov.append(mets.interval_coverage(samples[bi], y_np[bi], m_np[bi], alpha=0.1))
                d = mean[bi, 0][m_np[bi, 0] > 0] - y_np[bi, 0][m_np[bi, 0] > 0]
                rmse.append(float(np.sqrt((d ** 2).mean())) if d.size else float("nan"))
        def _m(a):
            a = np.array([v for v in a if v == v], float); return float(a.mean()) if a.size else float("nan")
        return dict(crps=_m(crps), coverage=_m(cov), rmse=_m(rmse))

    # ---- train ----
    best, log = float("inf"), {}
    for ep in range(start_epoch, config.epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for X, y, yg, lead in tr_loader:
            losses.append(train_step(X, y, yg, lead))
        sched.step()
        msg = f"[EPOCH {ep}/{config.epochs}] loss {np.mean(losses):.5f} | {time.time()-t0:.0f}s"
        log[ep] = {"train_loss": float(np.mean(losses))}
        if ep % config.val_every == 0:
            v = validate()
            msg += f" | CRPS {v['crps']:.4f} | cov {v['coverage']:.3f} | RMSE {v['rmse']:.4f} m"
            log[ep].update(v)
            if v["rmse"] < best:
                best = v["rmse"]
                torch.save({"epoch": ep, "model": model.state_dict(), "ema": ema.state_dict(),
                            "opt": opt.state_dict(), "sched": sched.state_dict(), "metrics": v},
                           os.path.join(config.out, "best_sda.pth.tar"))
        # save a resumable checkpoint every epoch (protects against interruption)
        torch.save({"epoch": ep, "model": model.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "train_loss": float(np.mean(losses))},
                   os.path.join(config.out, "last.pth.tar"))
        print(msg)
        with open(os.path.join(config.out, "trainlog.json"), "w") as f:
            json.dump(log, f, indent=2)
    print(f"done. best val CRPS {best:.4f} -> {config.out}")

if __name__ == "__main__":
    main()
