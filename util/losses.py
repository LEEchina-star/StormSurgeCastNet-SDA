import numpy as np
import torch

def get_loss(config):
    if config.loss=="weighted-l1":
        # compute sparse loss via nanmean reduction, wrap into additional check (because the nanmean of all NaNs is still NaN)
        criterion1 = lambda pred, targ: torch.nanmean(torch.nn.functional.l1_loss(targ, pred, reduction='none'))
        wrapper    = lambda arg: 0 if torch.isnan(arg) else arg
        # compute coarse loss, then weight both components via a scalar coefficient
        if config.out_conv[-1] == 1: criterion = lambda pred, targ: wrapper(criterion1(pred[:,:,[0],...], targ[:,:,[0],...]))
        elif config.out_conv[-1] == 2:
            criterion   = lambda pred, targ: wrapper(criterion1(pred[:,:,[0],...], targ[:,:,[0],...])) + config.weighting*wrapper(criterion1(pred[:,:,[1],...], targ[:,:,[1],...]))
        else: raise NotImplementedError
    elif config.loss=="l1":
        criterion= lambda pred, targ: torch.nanmean(torch.nn.functional.l1_loss(targ, pred, reduction='none'))
    elif config.loss=="l2":
        criterion = lambda pred, targ: torch.nanmean(torch.nn.functional.mse_loss(targ, pred, reduction='none'))
    else: raise NotImplementedError

    # wrap losses
    loss_wrap = lambda *args: args
    loss = loss_wrap(criterion) 
    return loss if not isinstance(loss, tuple) else loss[0]


def calc_loss(criterion, config, out, y):
    return criterion(out, y)

# see https://github.com/neuralhydrology/neuralhydrology/blob/5436973d7645cf090d0d75b0ffd1d6b7f902cf68/neuralhydrology/evaluation/metrics.py#L52
def nnse(obs, sim):
    denominator = np.nansum(((obs - np.nanmean(obs))**2))
    numerator = np.nansum(((sim - obs)**2))

    value = 1 - numerator / denominator
    normalized_value = 1 / (2 - value)

    return float(normalized_value)

# =============================================================================
# EDM conditional denoising loss helpers (SDA-Diff)
# =============================================================================
import torch as _torch

def sample_sigma(batch, device, p_mean=-1.2, p_std=1.2):
    """Lognormal noise-level sampling (EDM, Karras et al. 2022)."""
    return _torch.exp(_torch.randn(batch, device=device) * p_std + p_mean)


def edm_weighting(sigma, sigma_data=0.5):
    """EDM loss weighting lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma*sigma_data)^2."""
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def edm_denoising_loss(model, x0, c, lead=None, mask=None, sigma=None, residual=None):
    """
    Masked EDM denoising score-matching loss for EDMDataAssimilation.

    model:  EDMDataAssimilation exposing .denoise_loss(x0, c, lead, mask, sigma)
    x0:     [B,1,H,W] clean target (NaN treated as masked)
    c:      [B,T,C,H,W] context
    lead:   [B] lead times
    mask:   [B,1,H,W] 0/1 valid pixels (default from ~isnan)
    sigma:  optional pre-sampled noise levels
    residual: optional GTSM residual target (unused here, kept for interface parity)
    """
    return model.denoise_loss(x0, c, lead=lead, mask=mask, sigma=sigma)
