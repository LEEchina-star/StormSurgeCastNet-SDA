# -*- coding: utf-8 -*-
"""
Conditional score-based diffusion (EDM) + SDA-style likelihood-guided posterior sampling
for dense, global storm surge forecasting.

Inputs are identical to the FiLM U-TAE pipeline (see util/dataLoader.py):
    context c   : [B, T, C_in, H, W]  sparse + valid_mask + ERA5(msl,u10,v10) + GTSM
    lead    L   : [B]                 lead time (hours)
    target  x0  : [B, 1, H, W]        densified surge at lead time (NaN outside gauge pixels)
    obs     y   : [B, 1, H, W]        sparse gauge observations at inference (masked by A)

Training: EDM denoising score matching with sparse masking (and optional GTSM-residual target).
Inference: SDA posterior sampling -- annealed Langevin dynamics combining the conditional
prior score with a Gaussian likelihood gradient (Tweedie estimate, observation operator
A = validity mask, noise covariance R), yielding a calibrated posterior ensemble.

Reference: Rozet & Louppe, "Score-based Data Assimilation", NeurIPS 2023 (arXiv:2306.10574);
           Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", 2022.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d


# --------------------------------------------------------------------------
# sinusoidal embedding (sigma / lead time)
# --------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.dim() == 0:
            x = x.unsqueeze(0)
        device = x.device
        half = self.dim // 2
        emb = torch.exp(
            torch.arange(half, device=device)
            * (-math.log(10000.0) / half)
        )
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# --------------------------------------------------------------------------
# FiLM modulation
# --------------------------------------------------------------------------
def apply_film(x, scale, shift):
    """channel-wise FiLM: x * (1 + scale) + shift ; scale/shift: [B, C] or [B, C, 1, 1]"""
    if scale.dim() == 2:
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
    return x * (1.0 + scale) + shift


# --------------------------------------------------------------------------
# residual block with time + context FiLM conditioning
# --------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, out_c), in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.norm2 = nn.GroupNorm(min(8, out_c), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # time (sigma + lead) FiLM
        self.time_mlp = nn.Linear(emb_dim, out_c * 2)

    def forward(self, x, t_emb, ctx=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        h = self.drop(h)
        # time FiLM
        s, b = self.time_mlp(F.silu(t_emb)).chunk(2, dim=1)
        h = apply_film(h, s, b)
        # optional context FiLM (scale/shift already per-channel)
        if ctx is not None:
            s_c, b_c = ctx
            h = apply_film(h, s_c, b_c)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """spatial self-attention (channel-last), head dimension 32"""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.reshape(b, self.n_heads, c // self.n_heads, h * w)
        k = k.reshape(b, self.n_heads, c // self.n_heads, h * w)
        v = v.reshape(b, self.n_heads, c // self.n_heads, h * w)
        attn = torch.einsum("b h c n, b h c m -> b h n m", q, k) / math.sqrt(c // self.n_heads)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("b h n m, b h c m -> b h c n", attn, v)
        out = out.reshape(b, c, h, w)
        return self.to_out(out)


# --------------------------------------------------------------------------
# temporal context encoder: [B,T,C,H,W] -> per-scale FiLM parameters
# --------------------------------------------------------------------------
class _CtxBlock(nn.Module):
    """two conv blocks (GroupNorm+SiLU) + stride-2 downsample, no time conditioning"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.b1 = nn.Sequential(nn.GroupNorm(min(8, in_c), in_c), nn.SiLU(),
                                nn.Conv2d(in_c, out_c, 3, 1, 1))
        self.b2 = nn.Sequential(nn.GroupNorm(min(8, out_c), out_c), nn.SiLU(),
                                nn.Conv2d(out_c, out_c, 3, 1, 1))
        self.down = nn.Conv2d(out_c, out_c, 3, 2, 1)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        h = self.b1(x)
        h = self.b2(h)
        return self.down(h + self.skip(x))


class ContextEncoder(nn.Module):
    """
    Shared 2D conv encoder over the time dimension, followed by a lightweight
    temporal attention (learned query) aggregating T -> single frame, producing
    per-UNet-level channel-wise scale/shift for FiLM conditioning.

    NEW (fix): a DEDICATED sparse in-situ path. Channel 0 (rasterised gauges) and
    channel 1 (validity mask) are summarised per frame by max/mean over valid
    pixels, encoded by an MLP and temporally pooled, then injected as an
    additional per-level FiLM term. This prevents the sparse signal (only a few
    dozen non-zero pixels per frame) from being washed out by global average
    pooling in the main conv path.
    """
    def __init__(self, in_c, dim, levels, emb_dim):
        super().__init__()
        self.levels = levels
        self.in_proj = nn.Conv2d(in_c, dim, 3, 1, 1)
        self.enc = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.sparse_head = nn.ModuleList()
        cur = dim
        for lv in range(levels):
            out_c = dim * (2 ** lv)
            self.enc.append(_CtxBlock(cur, out_c))
            # temporal attention weights: learned query over pooled features
            self.attn.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(out_c, out_c), nn.SiLU()
            ))
            # per-scale FiLM params from pooled context
            # input = cat[avg, max]: global mean alone dilutes spatial structure
            # (GTSM/ERA5 surge peaks), so we append the spatial maximum to keep it.
            self.heads.append(nn.Linear(out_c * 2, out_c * 2))
            # sparse-path FiLM for this level
            self.sparse_head.append(nn.Linear(dim, out_c * 2))
            cur = out_c
        # per-level learned temporal query (dim matches level channels)
        self.temporal_query = nn.ParameterList(
            [nn.Parameter(torch.randn(dim * (2 ** lv))) for lv in range(levels)])
        # sparse gauge time-series encoder: per-frame (max, mean) -> dim
        self.sparse_mlp = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        # NEW (spatial conditioning): temporal aggregation of the RAW context
        # field. Global pooling (avg/avg+max) throws away WHERE the surge is;
        # the denoiser must see the GTSM/ERA5/sparse fields spatially, so we
        # learn frame weights and produce a full-res aggregated context map
        # [B, C, H, W] that is concatenated into the denoiser input.
        self.temp_pool = nn.Sequential(nn.Linear(in_c, dim), nn.SiLU(), nn.Linear(dim, 1))

    def _agg_raw(self, c):
        """Temporally aggregate the raw context [B,T,C,H,W] -> [B,C,H,W]
        with learned per-frame weights (softmax over time)."""
        b, t, ch, h, w = c.shape
        pool = c.mean(dim=(3, 4))                     # [B,T,C] per-frame global stats
        scores = self.temp_pool(pool).squeeze(-1)     # [B,T]
        w = scores.softmax(dim=1)                     # [B,T]
        return torch.einsum("b t c h w, b t -> b c h w", c, w)

    def _sparse_film(self, c, lv):
        """Summarise the sparse gauge channel + validity mask -> (s, sh) [B, C_lv]."""
        b, t, ch, h, w = c.shape
        sp = c[:, :, 0:1]                                   # [B,T,1,H,W] gauge values
        vmask = c[:, :, 1:2].clamp(0, 1)                    # [B,T,1,H,W] validity
        spm = sp * vmask
        mx = spm.amax(dim=(3, 4))                           # max over space (keeps gauge peaks)
        cnt = vmask.sum(dim=(3, 4)).clamp(min=1.0)
        mn = spm.sum(dim=(3, 4)) / cnt                      # mean over valid pixels
        feat = torch.cat([mx, mn], dim=-1)                  # [B,T,2]
        feat = self.sparse_mlp(feat)                        # [B,T,dim]
        agg = feat.mean(dim=1)                              # [B,dim] temporal mean
        s, sh = self.sparse_head[lv](agg).chunk(2, dim=1)   # [B, C_lv]
        return s, sh

    def forward(self, c, t_emb=None):
        """
        c: [B, T, C, H, W] -> (list over levels of (scale, shift) [B, C_lv],
                               raw aggregated context map [B, C, H, W])
        """
        b, t, ch, h, w = c.shape
        x = c.reshape(b * t, ch, h, w)
        x = self.in_proj(x)
        ctxs = []
        for lv in range(self.levels):
            q = self.temporal_query[lv]
            x = self.enc[lv](x)                   # [B*T, C_lv, H_lv, W_lv]
            pool = self.attn[lv](x)               # [B*T, C_lv]
            scores = torch.einsum("bc,c->b", pool, q)
            scores = scores.reshape(b, t).softmax(dim=1)     # [B, T]
            # temporal weighted average (per-channel, spatial broadcast)
            feat = x.reshape(b, t, *x.shape[1:])             # [B,T,C,H,W]
            agg = torch.einsum("b t c h w, b t -> b c h w", feat, scores)
            # cat[avg, max] pooling: preserves spatial peaks (GTSM surge structure,
            # pressure troughs) that global mean averaging washes out.
            pooled = torch.cat([agg.mean(dim=(2, 3)),
                                agg.amax(dim=(2, 3))], dim=1)          # [B, 2*C_lv]
            s, sh = self.heads[lv](pooled).chunk(2, dim=1)             # [B, C_lv]
            s_sp, sh_sp = self._sparse_film(c, lv)           # sparse in-situ contribution
            ctxs.append((s + s_sp, sh + sh_sp))
        return ctxs, self._agg_raw(c)


# --------------------------------------------------------------------------
# conditional U-Net denoiser (EDM preconditioned, x0-prediction)
# --------------------------------------------------------------------------
class CondUNet(nn.Module):
    def __init__(self, in_ch, out_ch, dim=64, levels=3, emb_dim=256, attn_level=2, dropout=0.0):
        super().__init__()
        self.levels = levels
        self.in_proj = nn.Conv2d(in_ch, dim, 3, 1, 1)
        # time embedding: sigma (ln) + lead
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim), nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.lead_proj = nn.Linear(1, emb_dim)

        # encoder
        self.enc = nn.ModuleList()
        cur = dim
        for lv in range(levels):
            out_c = dim * (2 ** lv)
            blk = nn.ModuleList([
                ResBlock(cur, out_c, emb_dim, dropout),
                ResBlock(out_c, out_c, emb_dim, dropout),
            ])
            if lv == attn_level:
                blk.append(AttentionBlock(out_c))
            self.enc.append(blk)
            cur = out_c

        # mid
        self.mid = nn.ModuleList([
            ResBlock(cur, cur, emb_dim, dropout),
            AttentionBlock(cur),
            ResBlock(cur, cur, emb_dim, dropout),
        ])

        # decoder
        self.dec = nn.ModuleList()
        for lv in reversed(range(levels)):
            in_c = cur + dim * (2 ** lv)
            out_c = dim * (2 ** lv)
            blk = nn.ModuleList([
                ResBlock(in_c, out_c, emb_dim, dropout),
                ResBlock(out_c, out_c, emb_dim, dropout),
            ])
            if lv == attn_level:
                blk.append(AttentionBlock(out_c))
            self.dec.append(blk)
            cur = out_c

        self.out_norm = nn.GroupNorm(min(8, cur), cur)
        self.out_conv = nn.Conv2d(cur, out_ch, 1)

    def forward(self, x, sigma, lead=None, ctx=None):
        """
        x:     [B, C_in, H, W] noisy input (already c_in preconditioned by caller or raw)
        sigma: [B] noise levels
        lead:  [B] lead times (hours)
        ctx:   list of (scale, shift) per level from ContextEncoder
        """
        emb = self.time_mlp(sigma)                          # [B, emb_dim]
        if exists(lead):
            emb = emb + self.lead_proj(lead.unsqueeze(-1).float())
        x0 = self.in_proj(x)

        skips = []
        for lv, blk in enumerate(self.enc):
            for layer in blk:
                if isinstance(layer, ResBlock):
                    x0 = layer(x0, emb, ctx[lv] if ctx is not None else None)
                else:
                    x0 = layer(x0)
            skips.append(x0)
            if lv < self.levels - 1:
                x0 = F.avg_pool2d(x0, 2)

        for layer in self.mid:
            if isinstance(layer, ResBlock):
                x0 = layer(x0, emb)
            else:
                x0 = layer(x0)

        for lv, blk in enumerate(self.dec):
            skip = skips[self.levels - 1 - lv]
            x0 = F.interpolate(x0, size=skip.shape[-2:], mode="nearest")
            x0 = torch.cat([x0, skip], dim=1)
            for layer in blk:
                if isinstance(layer, ResBlock):
                    x0 = layer(x0, emb, ctx[self.levels - 1 - lv] if ctx is not None else None)
                else:
                    x0 = layer(x0)

        return self.out_conv(F.silu(self.out_norm(x0)))


# --------------------------------------------------------------------------
# full model: conditional EDM + SDA posterior sampling
# --------------------------------------------------------------------------
class EDMDataAssimilation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cond_in = getattr(config, "in_dim", 6)
        # denoiser input = noisy surge + RAW time-aggregated context field
        # (spatial conditioning: GTSM/ERA5/sparse structure stays visible)
        in_ch   = getattr(config, "diff_in_ch", 1) + cond_in
        out_ch  = getattr(config, "diff_out_ch", 1)
        dim     = getattr(config, "diff_dim", 64)
        levels  = getattr(config, "diff_levels", 3)
        emb_dim = getattr(config, "diff_emb_dim", 256)
        attn_level = getattr(config, "diff_attn_level", 2)
        dropout = getattr(config, "diff_dropout", 0.0)
        self.sigma_data = getattr(config, "sigma_data", 0.5)
        self.P_mean     = getattr(config, "p_mean", -1.2)
        self.P_std      = getattr(config, "p_std", 1.2)

        cond_in = getattr(config, "in_dim", 6)
        self.context_encoder = ContextEncoder(cond_in, dim, levels, emb_dim)
        self.denoiser = CondUNet(in_ch, out_ch, dim, levels, emb_dim, attn_level, dropout)

    # ---------------------------------------------------------- conditioning
    def _ctx(self, c, lead):
        """c: [B,T,C,H,W] -> (per-level FiLM params, raw aggregated context map)"""
        emb = torch.zeros(c.shape[0], device=c.device)
        return self.context_encoder(c, emb)

    # ------------------------------------------------------- EDM preconditioning
    def denoise(self, x_t, sigma, c, lead=None):
        """
        x_t: [B, C, H, W] noisy target
        returns D_theta (estimate of clean x0), [B, C, H, W]
        """
        ctxs, c_agg = self._ctx(c, lead)
        c_in = 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        # EDM preconditioning applies to the noisy image ONLY; the aggregated
        # context is clean conditioning and enters the denoiser unscaled.
        x_in = torch.cat([x_t * c_in.view(-1, 1, 1, 1), c_agg], dim=1)
        # official EDM time conditioning: c_noise = log(sigma)/4 (Karras et al. 2022)
        F_in = self.denoiser(x_in, sigma.log() / 4, lead, ctxs)
        return c_skip.view(-1, 1, 1, 1) * x_t + c_out.view(-1, 1, 1, 1) * F_in

    # ---------------------------------------------------------- training loss
    def denoise_loss(self, x0, c, lead, mask=None, sigma=None):
        """
        x0:   [B, 1, H, W] clean target (NaN -> treated as masked)
        c:    [B, T, C, H, W] context
        lead: [B]
        mask: [B, 1, H, W] float 0/1 valid pixels (defaults to ~isnan(x0))
        """
        x0 = x0.float()
        if mask is None:
            mask = (~torch.isnan(x0)).float()
        x0 = torch.nan_to_num(x0, 0.0)

        b = x0.shape[0]
        if sigma is None:
            sigma = torch.exp(torch.randn(b, device=x0.device) * self.P_std + self.P_mean)
        noise = torch.randn_like(x0)
        x_t = x0 + noise * sigma.view(-1, 1, 1, 1)

        pred = self.denoise(x_t, sigma, c, lead)          # [B,1,H,W]

        # EDM loss weighting
        w = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        diff = (pred - x0) ** 2
        if mask is not None:
            diff = diff * mask
            denom = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
            per_sample = diff.sum(dim=(1, 2, 3)) / denom
        else:
            per_sample = diff.mean(dim=(1, 2, 3))
        return (w * per_sample).mean()

    # ---------------------------------------------------------- SDA posterior sampling
    @torch.no_grad()
    def sample_posterior(self, c, lead, y=None, mask=None, R=0.1,
                         steps=50, guidance=1.0, ensemble=8,
                         sigma_max=None, sigma_min=0.002, seed=None,
                         device=None, like_mode='autograd', sampler='ode'):
        """
        Annealed Langevin dynamics with likelihood guidance (SDA, Rozet & Louppe 2023).

        c:     [B, T, C, H, W]
        lead:  [B]
        y:     [B, 1, H, W] sparse observations at target time (NaN where unavailable)
        mask:  [B, 1, H, W] observation operator A (valid pixel = 1)
        R:     observation noise std (scalar) -> covariance R = R^2 I
        guidance: weight of the likelihood gradient
        device: optional sampling device (e.g. 'cpu' to dodge flaky MPS autograd);
                model & inputs are moved there for the duration of sampling.
        returns: dict(mean, q05, q50, q95, samples)  (on `device` if given, else input device)
        """
        b, _, c_, h, w = c.shape
        device = c.device if device is None else torch.device(device)
        if sigma_max is None:
            sigma_max = getattr(self.config, "sigma_max", 1.5)
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        # optionally relocate model + inputs for stable sampling
        params_dev = next(self.parameters()).device
        moved = params_dev != device
        if moved:
            self.to(device)
            c = c.to(device); lead = lead.to(device)
            if y is not None: y = y.to(device)
            if mask is not None: mask = mask.to(device)

        obs_mask = mask
        if y is not None:
            y = y.float()
            if obs_mask is None:
                obs_mask = (~torch.isnan(y)).float()
            y = torch.nan_to_num(y, 0.0)
            y_obs = y * obs_mask

        # EDM power schedule (Karras et al. 2022, Algorithm 2) with t_N = 0 appended
        rho = 7.0
        inv = torch.linspace(0, 1, steps, device=device)
        sigmas = (sigma_max ** (1 / rho) + inv * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
        samples = []
        for n in range(ensemble):
            x = torch.randn(b, 1, h, w, device=device) * sigma_max
            for i in range(steps):
                sig = sigmas[i].expand(b)
                if y is not None:
                    # annealed likelihood: gradient of 0.5*||(y - A x0_hat)||^2 / (R^2 + sigma^2)
                    # -> weak guidance at large sigma (stable), strong at small sigma (exact DA)
                    inv_var = 1.0 / (R ** 2 + sig ** 2)
                    if like_mode == 'replace':
                        # MPS-stable first-order gradient (pure tensor ops):
                        #   g = A^T (y - A x0_hat) / (R^2 + sigma^2)
                        with torch.no_grad():
                            x0_hat = self.denoise(x, sig, c, lead)
                        g = (y_obs - x0_hat * obs_mask) * obs_mask * inv_var.view(-1, 1, 1, 1)
                    else:
                        # exact gradient through the Tweedie estimate (SDA Eq. 10)
                        with torch.enable_grad():
                            xg = x.detach().requires_grad_(True)   # leaf var for exact SDA gradient
                            x0_hat = self.denoise(xg, sig, c, lead)
                            resid = (y_obs - x0_hat * obs_mask)
                            g = torch.autograd.grad(
                                0.5 * ((resid ** 2) * inv_var.view(-1, 1, 1, 1)).mean(), xg)[0]
                            x0_hat = x0_hat.detach()
                else:
                    with torch.no_grad():
                        x0_hat = self.denoise(x, sig, c, lead)
                    g = torch.zeros_like(x)
                # EDM Euler step + guidance (Karras et al. 2022, Algorithm 2)
                # note: dt_ < 0 (sigma decreases); likelihood guidance must pull x
                # towards the observations, hence the MINUS sign on g = (y - A x0_hat)
                d_cur = (x - x0_hat) / sig.view(-1, 1, 1, 1)
                sigma_next = sigmas[i + 1]
                dt_ = sigma_next - sigmas[i]
                x_next = x + dt_ * d_cur - guidance * dt_ * sigmas[i] * g
                # 2nd-order Heun correction (skipped on the last step, t_N = 0)
                if i < steps - 1:
                    sig2 = sigma_next.expand(b)
                    if y is not None:
                        if like_mode == 'replace':
                            with torch.no_grad():
                                x0_hat2 = self.denoise(x_next, sig2, c, lead)
                            inv_var2 = 1.0 / (R ** 2 + sig2 ** 2)
                            g2 = (y_obs - x0_hat2 * obs_mask) * obs_mask * inv_var2.view(-1, 1, 1, 1)
                        else:
                            with torch.enable_grad():
                                x0_hat2 = self.denoise(x_next.detach().requires_grad_(True), sig2, c, lead)
                                resid2 = (y_obs - x0_hat2 * obs_mask)
                                g2 = torch.autograd.grad(
                                    0.5 * ((resid2 ** 2) * (1.0 / (R ** 2 + sig2 ** 2)).view(-1, 1, 1, 1)).mean(),
                                    x_next)[0]
                                x0_hat2 = x0_hat2.detach()
                        g = g2
                    else:
                        with torch.no_grad():
                            x0_hat2 = self.denoise(x_next, sig2, c, lead)
                    d_prime = (x_next - x0_hat2) / sig2.view(-1, 1, 1, 1)
                    x_next = x + dt_ * (0.5 * d_cur + 0.5 * d_prime) - guidance * dt_ * sigmas[i] * g
                x = x_next
            samples.append(x)
        samples = torch.stack(samples, dim=0)              # [N, B, 1, H, W]
        samples = samples.permute(1, 0, 2, 3, 4)           # [B, N, 1, H, W]
        mean = samples.mean(dim=1)
        q05 = torch.quantile(samples, 0.05, dim=1)
        q50 = torch.quantile(samples, 0.50, dim=1)
        q95 = torch.quantile(samples, 0.95, dim=1)
        if moved:
            self.to(params_dev)
        return dict(mean=mean, q05=q05, q50=q50, q95=q95, samples=samples)
