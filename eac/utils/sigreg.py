"""SIGReg: characteristic-function regularizer for latent embeddings.

Adapted (in spirit) from the LeWM line of work (arXiv:2603.19312). The goal is
to prevent latent collapse / degeneracy in the world-model latent path by
matching, in 1-D random projections, the empirical characteristic function of
the latent to that of a standard Gaussian.

Implementation notes:
  * Fully differentiable, AMP-safe (computed in float32 internally).
  * Active samples < 8 -> returns 0 (with grad-preserving zero so callers can
    still backprop without branching).
  * Random projections are drawn once per call from the latent's device; this
    avoids stale buffers across years / model copies.

Usage:
    from utils.sigreg import sigreg_loss
    loss = sigreg_loss(z, mask=valid_mask)  # z: [..., D]; mask: [...]
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _flatten_active(z: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Reshape `z` to `[M_active, D]`; drop rows where mask==0 (if given)."""
    if z.dim() < 2:
        return z.unsqueeze(0)
    D = z.shape[-1]
    flat = z.reshape(-1, D)
    if mask is not None:
        m = mask.reshape(-1)
        if m.shape[0] != flat.shape[0]:
            # Mask is shaped over a subset of `z` axes (e.g. [B,H,N]) — broadcast.
            try:
                m = mask.expand(z.shape[:-1]).reshape(-1)
            except RuntimeError:
                m = None  # fall back to using all rows
        if m is not None:
            m = m.bool()
            flat = flat[m]
    # Drop non-finite rows so backward never sees NaNs.
    finite = torch.isfinite(flat).all(dim=-1)
    flat = flat[finite]
    return flat


def sigreg_loss(
    z: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_projections: int = 128,
    num_knots: int = 16,
    max_samples: int = 4096,
    t_min: float = 0.2,
    t_max: float = 4.0,
    weight_type: str = "none",   # "none" | "exp"
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable SIGReg-style regularizer.

    Args:
        z: latent tensor, last dim is feature dim D (e.g. [B,H,N,D], [B,N,D]).
        mask: optional active/finite mask over the non-feature axes of `z`.
        num_projections: number of 1-D random unit projections u ~ S^{D-1}.
        num_knots: number of frequencies t sampled in [t_min, t_max].
        max_samples: subsample if active rows exceed this many.
        weight_type: knot weighting. "exp" emphasizes low-frequency matching.

    Returns:
        scalar loss in float (same dtype as `z` for AMP, internal math is fp32).
    """
    if z is None or z.numel() == 0:
        return z.new_zeros(())

    orig_dtype = z.dtype
    # Run in float32 for numerical stability under AMP.
    z32 = z.float()
    flat = _flatten_active(z32, mask)
    M, D = flat.shape if flat.dim() == 2 else (0, z32.shape[-1])
    if M < 8 or D < 1:
        # Grad-preserving zero (depends on `z` so .backward() is well-defined).
        return (z32.sum() * 0.0).to(orig_dtype)

    # Per-feature standardization so SIGReg is comparable to N(0,1) regardless
    # of latent's raw scale. This makes the loss easier to weight (lambda).
    mu = flat.mean(dim=0, keepdim=True)
    sd = flat.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    flat = (flat - mu) / sd

    # Subsample if we have too many rows (memory + speed).
    if M > max_samples:
        idx = torch.randperm(M, device=flat.device)[:max_samples]
        flat = flat[idx]
        M = max_samples

    # Random unit directions u: [P, D].
    P = max(1, int(num_projections))
    u = torch.randn(P, D, device=flat.device, dtype=flat.dtype)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(eps)

    # h: [M, P] = projections of each sample on each direction.
    h = flat @ u.t()

    # Frequencies t: [K].
    K = max(1, int(num_knots))
    t = torch.linspace(t_min, t_max, K, device=flat.device, dtype=flat.dtype)

    # ECF: phi(t) = E[exp(i*t*h)]. Real / imag parts averaged over M.
    # th: [K, M, P] via outer-style broadcast.
    th = t.view(K, 1, 1) * h.unsqueeze(0)  # [K, M, P]
    cos_th = th.cos().mean(dim=1)          # [K, P]
    sin_th = th.sin().mean(dim=1)          # [K, P]

    # Target: phi_N(0,1)(t) = exp(-t^2/2). Real only.
    phi0 = torch.exp(-0.5 * t * t).view(K, 1)  # [K, 1]

    # MSE per knot, then weighted average over knots and projections.
    err = (cos_th - phi0) ** 2 + sin_th ** 2   # [K, P]

    if weight_type == "exp":
        w = torch.exp(-0.5 * t * t).view(K, 1)
        loss = (err * w).sum() / (w.sum() * P)
    else:
        loss = err.mean()

    if not torch.isfinite(loss):
        return (z32.sum() * 0.0).to(orig_dtype)
    return loss.to(orig_dtype)


def sigreg_on_dict(
    latents: dict,
    flags: dict,
    mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Sum SIGReg over named latent tensors gated by `flags`.

    Example:
        sigreg_on_dict(
            {"ctx": z_ctx, "tgt": z_tgt, "pred": z_pred},
            {"ctx": True,   "tgt": True,  "pred": False},
            mask=valid_mask,
        )
    """
    total = None
    for name, z in latents.items():
        if z is None:
            continue
        if not flags.get(name, False):
            continue
        l = sigreg_loss(z, mask=mask, **kwargs)
        total = l if total is None else total + l
    if total is None:
        # Pick any tensor to anchor the dtype/device.
        any_z = next((v for v in latents.values() if v is not None), None)
        return any_z.new_zeros(()) if any_z is not None else torch.zeros(())
    return total
