# -*- coding: utf-8 -*-
"""
Build a COMPACT, gauge-id-split train/val cache from the 3 GB val.npy cache
(which holds 256x256, T=12 samples with REAL ERA5 -- identical inputs to the
original StormSurgeCastNet pipeline).

Why compact: the original object-array cache explodes memory when unpickled
(3 GB file -> ~286 GB virtual peak); stacking into plain float32 arrays loads
fast and needs little RAM.

Split protocol: gauges are split 70/30 (train/val) by gauge id, so no gauge
appears in both sets (no leakage). Save:
    cache_sda_full/{train,val}.npz
        X    [N, T, 6, H, W]  context (sparse+mask+era5+gtsm, same order as prepare_data)
        y    [N, 1, H, W]     target sparse surge (NaN = unobserved)
        lead [N]
        ids  [N]  gauge ids
        lon, lat [N]
Usage:
    python util/build_compact_cache.py --src Data2/cache/val.npy --out cache_sda_full \
        --val_frac 0.3 --seed 1
"""
import os
import sys
import argparse
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Data2/cache/val.npy")
    ap.add_argument("--out", default="cache_sda_full")
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resize", type=int, default=0,
                    help="if >0, bilinear-resize spatial dims to this size (quick iterations)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print(f"loading {args.src} (may take ~1 min, high virtual memory but OK)...")
    data = np.load(args.src, allow_pickle=True).item()["dataset"]
    print(f"loaded {len(data)} samples")

    # ---- assemble compact tensors (same channel order as train.py prepare_data) ----
    X, Y, YG, lead, ids, lon, lat = [], [], [], [], [], [], []
    for i, s in enumerate(data):
        if s is None:
            continue
        inp, tgt = s["input"], s["target"]
        if args.resize and inp["sparse"].shape[-1] != args.resize:
            def rz(a):
                t_ = torch.from_numpy(a[None]).float()
                t_ = torch.nn.functional.interpolate(t_, size=(args.resize, args.resize), mode="bilinear")
                return t_[0].numpy()
            x = np.concatenate([rz(inp["sparse"]), rz(inp["valid_mask"]),
                                rz(inp["era5"]), rz(inp["gtsm"])], axis=1)   # [T,6,h,w]
            y = rz(tgt["sparse"])                                            # [1,h,w]
            yg = rz(tgt["gtsm"])                                             # [1,h,w] dense
        else:
            x = np.concatenate([inp["sparse"], inp["valid_mask"],
                                inp["era5"], inp["gtsm"]], axis=1)           # [T,6,H,W]
            y = tgt["sparse"]                                                # [1,H,W]
            yg = tgt["gtsm"]                                                 # [1,H,W] dense
        X.append(x); Y.append(y); YG.append(yg)
        lead.append(float(np.asarray(inp["td_lead"]).reshape(-1)[0]))
        ids.append(int(np.asarray(tgt["id"]).reshape(-1)[0]))
        lon.append(float(np.asarray(tgt["lon_gauge"]).reshape(-1)[0]))
        lat.append(float(np.asarray(tgt["lat_gauge"]).reshape(-1)[0]))

    X = np.stack(X).astype(np.float32)          # [N,T,6,H,W]
    Y = np.stack(Y).astype(np.float32)          # [N,1,H,W] sparse (target gauge pixel)
    YG = np.stack(YG).astype(np.float32)        # [N,1,H,W] dense GTSM target
    lead = np.asarray(lead, np.float32)
    ids = np.asarray(ids, np.int64)
    lon = np.asarray(lon, np.float32)
    lat = np.asarray(lat, np.float32)
    print(f"X {X.shape} | Y {Y.shape} | YG {YG.shape} | lead {lead.shape} | unique gauges {len(np.unique(ids))}")

    # ---- gauge-level split ----
    uniq = np.unique(ids)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * args.val_frac))
    val_ids, train_ids = set(uniq[:n_val]), set(uniq[n_val:])
    tr_idx = np.array([i for i, g in enumerate(ids) if g in train_ids])
    va_idx = np.array([i for i, g in enumerate(ids) if g in val_ids])
    print(f"train {len(tr_idx)} samples / {len(train_ids)} gauges | val {len(va_idx)} samples / {len(val_ids)} gauges")

    for name, idx in (("train", tr_idx), ("val", va_idx)):
        path = os.path.join(args.out, f"{name}.npz")
        np.savez(path, X=X[idx], y=Y[idx], yg=YG[idx], lead=lead[idx], ids=ids[idx], lon=lon[idx], lat=lat[idx])
        print("saved", path, os.path.getsize(path) / 1e6, "MB")

if __name__ == "__main__":
    main()
