# -*- coding: utf-8 -*-
"""
FiLM U-TAE (original StormSurgeCastNet best model) trained on the SAME compact
real-data cache as train_sda_cache.py, for a fair head-to-head comparison.

Note: the compact cache holds only the sparse surge target (no GTSM target
channel), so U-TAE is trained with the sparse masked-L1 term only (the main
term of the paper's weighted-l1 loss); out_conv = [1].

Usage:
    python train_utae_cache.py --cache_dir cache_sda_full --resize 128 \
        --epochs 30 --batch_size 4
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
from parse_args import create_parser
from util import utils
from models.utae.utae import UTAE

def seed_packages(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_cache(cache_dir, split):
    path = os.path.join(cache_dir, f"{split}.npz")
    if os.path.isfile(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
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
        self.yg = torch.from_numpy(d["yg"])
        self.T = self.X.shape[1]
        # relative time positions (hours) for the L-TAE positional encoding
        self.pos = torch.arange(self.T, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i], self.yg[i], self.pos

def main():
    parser = create_parser(mode="train")
    parser.add_argument("--resize", type=int, default=0)
    parser.add_argument("--val_cache_dir", default="", help="separate cache dir for validation (default: same as --cache_dir)")
    parser.add_argument("--out", default="results_sda/cache_compare_utae")
    config = utils.str2list(parser.parse_args(), list_args=["encoder_widths", "decoder_widths", "out_conv"])
    seed_packages(config.rdm_seed)
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        config.device = "mps"
    device = torch.device(config.device)
    print(f"device: {device}")

    val_dir = config.val_cache_dir if config.val_cache_dir else config.cache_dir
    tr, va = load_cache(config.cache_dir, "train"), load_cache(val_dir, "val")
    if config.resize:
        for d in (tr, va):
            X = torch.from_numpy(d["X"]).float(); B, T, C, H, W = X.shape
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

    # ---- U-TAE (original default hyperparameters) ----
    model = UTAE(
        input_dim=tr["X"].shape[2],
        encoder_widths=[64, 64, 64, 128], decoder_widths=[64, 64, 64, 128],
        out_conv=[2], out_nonlin_mean=False,
        str_conv_k=4, str_conv_s=2, str_conv_p=1,
        agg_mode="att_group", encoder_norm="group", norm_skip="batch", norm_up="batch",
        decoder_norm="batch", n_head=16, d_model=256, d_k=4,
        encoder=False, return_maps=False, pad_value=0, padding_mode="reflect",
        positional_encoding=True, cond_dim=None, cond_norm_affine=None,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config.gamma)
    print(f"TOTAL PARAMS: {utils.get_ntrainparams(model)}")
    os.makedirs(config.out, exist_ok=True)
    with open(os.path.join(config.out, "conf.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    def masked_l1(pred, y):
        mask = (~torch.isnan(y)).float()
        y = torch.nan_to_num(y, 0.0)
        diff = torch.abs(pred - y) * mask
        denom = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return (diff.sum(dim=(1, 2, 3)) / denom).mean()

    def train_step(X, y, yg, pos):
        X, y, yg = X.to(device), y.to(device), yg.to(device)
        pos = pos.to(device)
        out = model(X, batch_positions=pos)  # [B,1,2,H,W] (sparse, gtsm)
        # same weighted-l1 as the original paper: sparse term + 1/100 GTSM term
        loss = masked_l1(out[:, :, 0], y) + config.weighting * masked_l1(out[:, :, 1], yg)
        opt.zero_grad(); loss.backward(); opt.step()
        return float(loss.item())

    @torch.no_grad()
    def validate():
        rmse, mae = [], []
        for X, y, yg, pos in va_loader:
            X, y = X.to(device), y.to(device)
            out = model(X, batch_positions=pos.to(device))[:, :, 0]  # [B,1,H,W] sparse head
            m = (~torch.isnan(y)).float()
            yc = torch.nan_to_num(y, 0.0)
            for bi in range(X.shape[0]):
                mm = m[bi, 0] > 0
                if mm.any():
                    d = (out[bi, 0][mm] - yc[bi, 0][mm]).float()
                    rmse.append(float(torch.sqrt((d ** 2).mean())))
                    mae.append(float(torch.abs(d).mean()))
        def _m(a):
            return float(np.mean(a)) if a else float("nan")
        return dict(rmse=_m(rmse), mae=_m(mae))

    best, log = float("inf"), {}
    for ep in range(1, config.epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for X, y, yg, pos in tr_loader:
            losses.append(train_step(X, y, yg, pos))
        sched.step()
        msg = f"[EPOCH {ep}/{config.epochs}] loss {np.mean(losses):.4f} | {time.time()-t0:.0f}s"
        log[ep] = {"train_loss": float(np.mean(losses))}
        if ep % config.val_every == 0:
            v = validate()
            msg += f" | RMSE {v['rmse']:.4f} m | MAE {v['mae']:.4f} m"
            log[ep].update(v)
            if v["rmse"] < best:
                best = v["rmse"]
                torch.save({"epoch": ep, "model": model.state_dict()},
                           os.path.join(config.out, "best_utae.pth.tar"))
        print(msg)
        with open(os.path.join(config.out, "trainlog.json"), "w") as f:
            json.dump(log, f, indent=2)
    print(f"done. best val RMSE {best:.4f} m -> {config.out}")

if __name__ == "__main__":
    main()
