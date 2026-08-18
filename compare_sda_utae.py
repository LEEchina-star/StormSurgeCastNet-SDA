# -*- coding: utf-8 -*-
"""Summarise SDA-Diff vs FiLM U-TAE runs on the same gauge-level split."""
import os, sys, json, glob, argparse
import numpy as np

def load_log(path):
    with open(path) as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sda", default="results_sda/cmp_sda_128/trainlog.json")
    ap.add_argument("--utae", default="results_sda/cmp_utae_128/trainlog.json")
    ap.add_argument("--sda_best", default="results_sda/cmp_sda_128/best_sda.pth.tar")
    ap.add_argument("--utae_best", default="results_sda/cmp_utae_128/best_utae.pth.tar")
    args = ap.parse_args()

    sda = load_log(args.sda) if os.path.exists(args.sda) else {}
    utae = load_log(args.utae) if os.path.exists(args.utae) else {}

    def last(js, key):
        vals = [v.get(key) for v in js.values() if isinstance(v, dict) and key in v and v[key] is not None]
        vals = [v for v in vals if v == v]
        return float(np.min(vals)) if vals else float("nan")

    print("=" * 62)
    print(f"{'metric':<16}{'SDA-Diff':>16}{'FiLM U-TAE':>16}")
    print("-" * 62)
    for k, lbl in [("rmse", "val RMSE (m)"), ("crps", "val CRPS"), ("coverage", "val cov90"), ("mae", "val MAE (m)")]:
        a = last(sda, k)
        b = last(utae, k)
        print(f"{lbl:<16}{a:>16.4f}{b:>16.4f}")
    print("-" * 62)
    print("checkpoints:", os.path.exists(args.sda_best), os.path.exists(args.utae_best))

if __name__ == "__main__":
    main()
