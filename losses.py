"""Training losses for the UrbanFormer field-prediction work packages.

WP1 (UrbanFormer-Mid, U-Net baseline) trains against :func:`masked_mse`: plain
mean squared error restricted to fluid cells, so solid (building) cells never
enter the gradient. This is the exact loss that produced the WP1 headline
numbers.

The richer WP3 field loss (masked MSE + gradient + optional spectral + tail
weighting, ``masked_field_loss`` with its ``make_radial_bins`` / ``_radial_psd``
helpers) is coupled to the WP3 training config and is ported alongside WP3.
"""

from __future__ import annotations


def masked_mse(pred, target, mask):
    """Mean squared error over fluid cells only.

    Solid (building) cells are excluded from both the numerator and the
    denominator, so predictions there contribute nothing to the loss.

    Parameters
    ----------
    pred, target : torch.Tensor
        Predicted and ground-truth fields of identical shape, e.g.
        ``(B, Ny, Nx)``.
    mask : torch.Tensor
        Fluid mask broadcastable to ``pred``; ``1.0`` on fluid cells and
        ``0.0`` on solid cells (the ``fluid_mask_mid`` convention).

    Returns
    -------
    torch.Tensor
        Scalar: the sum of squared error over fluid cells divided by the number
        of fluid cells (``mask.sum()``).
    """
    return (mask * (pred - target) ** 2).sum() / mask.sum()


# ===========================================================================
# WP3 field loss: masked tail-MSE + gradient + radial-spectral
# ===========================================================================
import torch as _torch  # noqa: E402

# --- WP3 loss configuration (the UF-F defaults that produced the flagship) ---
TAIL_ALPHA    = 0.3     # tail up-weighting exponent (0 disables)
LAMBDA_GRAD   = 0.5     # weight on the finite-difference gradient term
SPECTRAL_LOSS = True    # include the radial-PSD term
LAMBDA_SPEC   = 0.1     # weight on the spectral term
NBIN          = 24      # radial PSD bins


def _radial_psd(field, fluid, rbin, nbin):
    """Radially-averaged power spectrum of a masked 2D field. field: (B, Ny, Nx)."""
    f = (field * fluid)
    F2 = _torch.fft.rfft2(f)
    p = (F2.real ** 2 + F2.imag ** 2).reshape(field.shape[0], -1)      # (B, Ny*(Nx//2+1))
    out = _torch.zeros(field.shape[0], nbin, device=field.device)
    out.index_add_(1, rbin, p)
    cnt = _torch.bincount(rbin, minlength=nbin).clamp_min(1).float()
    return out / cnt[None]


def make_radial_bins(Ny, Nx, nbin=NBIN, device="cpu"):
    """Radial-frequency bin index for each rfft2 coefficient of an (Ny, Nx) grid."""
    fy = _torch.fft.fftfreq(Ny)[:, None]
    fx = _torch.fft.rfftfreq(Nx)[None, :]
    r = _torch.sqrt(fy ** 2 + fx ** 2)
    rbin = _torch.clamp((r / r.max() * (nbin - 1)).round().long().reshape(-1), 0, nbin - 1)
    return rbin.to(device)


def masked_field_loss(pred, target, fluid, rbin=None, nbin=NBIN):
    """UF-F training loss over fluid cells (WP3).

    pred / target / fluid: (B, Ny, Nx). Combines a tail-weighted masked MSE, a
    finite-difference gradient term, and an optional radial-PSD (spectral) term
    that penalises the missing high-frequency energy directly. Returns
    ``(loss, parts)`` where ``parts`` breaks out the mse / grad / spec scalars.
    """
    m = fluid
    denom = m.sum().clamp_min(1.0)
    se = (pred - target) ** 2
    if TAIL_ALPHA > 0:
        with _torch.no_grad():
            t = target[m > 0]
            mu, sd = t.mean(), t.std().clamp_min(1e-6)
            w = ((target - mu).abs() / sd).clamp_min(1e-3) ** TAIL_ALPHA
            w = w / (w[m > 0].mean())
        mse = (w * se * m).sum() / denom
    else:
        mse = (se * m).sum() / denom

    # gradient (finite diff), masked by the intersection of valid neighbours
    gx_p = pred[:, :, 1:] - pred[:, :, :-1]; gx_t = target[:, :, 1:] - target[:, :, :-1]
    gy_p = pred[:, 1:, :] - pred[:, :-1, :]; gy_t = target[:, 1:, :] - target[:, :-1, :]
    mx = (m[:, :, 1:] * m[:, :, :-1]); my = (m[:, 1:, :] * m[:, :-1, :])
    grad = (((gx_p - gx_t) ** 2 * mx).sum() / mx.sum().clamp_min(1.0)
            + ((gy_p - gy_t) ** 2 * my).sum() / my.sum().clamp_min(1.0))

    loss = mse + LAMBDA_GRAD * grad
    spec = _torch.tensor(0.0, device=pred.device)
    if SPECTRAL_LOSS and rbin is not None:
        pp = _radial_psd(pred, m, rbin, nbin)
        pt = _radial_psd(target, m, rbin, nbin)
        spec = (_torch.log1p(pp) - _torch.log1p(pt)).abs().mean()
        loss = loss + LAMBDA_SPEC * spec

    return loss, dict(mse=float(mse.detach()), grad=float(grad.detach()), spec=float(spec.detach()))
