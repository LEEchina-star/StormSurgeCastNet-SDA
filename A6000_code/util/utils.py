# -*- coding: utf-8 -*-
import torch

# map arg string of written list to list
def str2list(config, list_args):
    for k, v in vars(config).items():
        if k in list_args and v is not None and isinstance(v, str):
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))
    return config


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Pick the best available device: CUDA (RTX A6000) first, then MPS (macOS), then CPU.
    Windows-safe: torch.backends.mps is guarded with getattr (absent on some builds)."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"