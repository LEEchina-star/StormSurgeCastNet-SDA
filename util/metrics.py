import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """
    
    def reset(self): pass
    def add(self):   pass
    def value(self): pass


# TODO: evaluating image metrics and aggregation via avg_img_metrics is currently deprecated,
#       proper evaluation needs to NaN mask targets & predictions (but also implement behavior for flags such as --eval_gtsm_pred).
#       For now, call test.py for manual evaluation of model checkpoints on test split, in terms of image metrics.
def img_metrics(target, pred):
    rmse = torch.sqrt(torch.mean(torch.square(target - pred)))
    mae = torch.mean(torch.abs(target - pred))

    metric_dict = {'RMSE': rmse.cpu().numpy().item(),
                   'MAE': mae.cpu().numpy().item()}
    
    return metric_dict

class avg_img_metrics(Metric):
    def __init__(self):
        super().__init__()
        self.n_samples = 0
        self.metrics   = ['RMSE', 'MAE']
        self.metrics  += ['error', 'mean se', 'mean ae']

        self.running_img_metrics = {}
        self.running_nonan_count = {}
        self.reset()

    def reset(self):
        for metric in self.metrics: 
            self.running_nonan_count[metric] = 0
            self.running_img_metrics[metric] = np.nan

    def add(self, metrics_dict):
        for key, val in metrics_dict.items():
            # skip variables not registered
            if key not in self.metrics: continue
            # filter variables not translated to numpy yet
            if torch.is_tensor(val): continue
            if isinstance(val, tuple): val=val[0]

            # only keep a running mean of non-nan values
            if np.isnan(val): continue

            if not self.running_nonan_count[key]: 
                self.running_nonan_count[key] = 1
                self.running_img_metrics[key] = val
            else: 
                self.running_nonan_count[key]+= 1
                self.running_img_metrics[key] = (self.running_nonan_count[key]-1)/self.running_nonan_count[key] * self.running_img_metrics[key] \
                                                + 1/self.running_nonan_count[key] * val

    def value(self):
        return self.running_img_metrics

# =============================================================================
# Probabilistic forecast metrics (SDA-Diff)
# Shape conventions:
#   ens: [N, 1, H, W] or [N, H, W]  (ensemble members first)
#   obs: [1, H, W]   or [H, W]      (observations; NaN = unobserved)
# =============================================================================
import numpy as _np


def _prep(ens, obs, mask=None):
    ens = _np.asarray(ens, dtype=_np.float32)
    obs = _np.asarray(obs, dtype=_np.float32)
    # ens -> [N, H, W]
    if ens.ndim == 5:            # [B, N, 1, H, W] -> take first batch
        ens = ens[0]
    if ens.ndim == 4 and ens.shape[1] == 1:   # [N, 1, H, W]
        ens = ens[:, 0]
    elif ens.ndim == 4 and ens.shape[0] == 1: # [1, N, H, W]
        ens = ens[0]
    # obs -> [H, W]
    if obs.ndim == 4:            # [B, 1, H, W] -> take first batch
        obs = obs[0]
    if obs.ndim == 3 and obs.shape[0] == 1:   # [1, H, W]
        obs = obs[0]
    # mask -> [H, W] (same squeeze rules as obs)
    if mask is not None:
        mask = _np.asarray(mask, dtype=_np.float32)
        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
    return ens, obs, mask


def _valid_flat(obs, mask=None):
    obs = _np.asarray(obs, dtype=_np.float32)
    if mask is None:
        mask = ~_np.isnan(obs)
    else:
        mask = _np.asarray(mask, dtype=bool) & ~_np.isnan(obs)
    return obs[mask], mask


def crps_ensemble(ens, obs, mask=None):
    """Fair CRPS (ensemble version), averaged over valid pixels."""
    ens, obs, mask = _prep(ens, obs, mask)
    o, m = _valid_flat(obs, mask)
    if o.size == 0:
        return float("nan")
    ens_v = ens[..., m]                      # [N, M]
    n = ens_v.shape[0]
    if n < 2:
        return float("nan")
    term1 = _np.abs(ens_v - o[None, :]).mean(axis=0)               # [M]
    term2 = _np.zeros(o.shape)
    for i in range(n):
        term2 += _np.abs(ens_v[i] - ens_v).sum(axis=0)             # [M]
    term2 = term2 / (2.0 * n * n)
    return float((term1 - term2).mean())


def interval_coverage(ens, obs, mask=None, alpha=0.1):
    """Fraction of observations inside the (alpha/2, 1-alpha/2) ensemble interval."""
    ens, obs, mask = _prep(ens, obs, mask)
    o, m = _valid_flat(obs, mask)
    if o.size == 0:
        return float("nan")
    lo = _np.quantile(ens[..., m], alpha / 2, axis=0)
    hi = _np.quantile(ens[..., m], 1 - alpha / 2, axis=0)
    return float(((o >= lo) & (o <= hi)).mean())


def pit_values(ens, obs, mask=None):
    """Probability integral transform u = F_ens(obs) at valid pixels (calibration histogram)."""
    ens, obs, mask = _prep(ens, obs, mask)
    o, m = _valid_flat(obs, mask)
    if o.size == 0:
        return _np.array([])
    n = ens.shape[0]
    return (ens[..., m] < o[None, :]).sum(axis=0) / n


def spread_skill(ens, obs, mask=None):
    """per-valid-pixel ensemble std (spread) and |ensemble-mean| error (skill)."""
    ens, obs, mask = _prep(ens, obs, mask)
    o, m = _valid_flat(obs, mask)
    if o.size == 0:
        return _np.array([]), _np.array([])
    spread = ens[..., m].std(axis=0)
    skill = _np.abs(ens[..., m].mean(axis=0) - o)
    return spread, skill
