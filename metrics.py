"""Field-quality metrics for the UrbanFormer work packages (WP5 scope).

Pure metric functions over stacked fields ``(N, Ny, Nx)`` with a fluid mask:
only fluid cells (``M > 0``) count, so building cells never enter any score.

* :func:`field_metrics`  -- global RMSE / MAE / R^2 / relative-L2 over fluid cells.
* :func:`per_case_rmse`  -- one fluid-only RMSE per case (feeds the §5 analysis).
* :func:`spatial_corr`   -- mean per-case Pearson correlation over fluid cells.

The physics-oriented metrics (wake / canyon / deficit errors) read WP5 threshold
constants (``WAKE_D``, ``CAN_D``, ``LOW_THR``, ``HI_THR``) and are ported with the
WP5 evaluation notebook rather than here.
"""

from __future__ import annotations

import numpy as np
import torch


def field_metrics(P, T, M):
    """Global field metrics over fluid cells.

    Parameters
    ----------
    P, T : torch.Tensor
        Predicted and target fields, shape ``(N, Ny, Nx)``.
    M : torch.Tensor
        Fluid mask, ``> 0`` on fluid cells; broadcastable to ``P``.

    Returns
    -------
    dict
        ``RMSE``, ``MAE``, ``R2``, ``relL2`` as Python floats. ``SS_tot`` and the
        target energy are floored at ``1e-12`` so a constant or all-zero target
        does not divide by zero.
    """
    fluid = M > 0
    fp, ft = P[fluid], T[fluid]
    SS_res = ((fp - ft) ** 2).sum()
    SS_tot = ((ft - ft.mean()) ** 2).sum().clamp_min(1e-12)
    return dict(
        RMSE=torch.sqrt(SS_res / ft.numel()).item(),
        MAE=(fp - ft).abs().mean().item(),
        R2=(1 - SS_res / SS_tot).item(),
        relL2=(torch.sqrt(SS_res) / torch.sqrt((ft ** 2).sum().clamp_min(1e-12))).item(),
    )


def per_case_rmse(P, T, M):
    """Fluid-only RMSE for each case.

    Returns a ``(N,)`` array; a case with no fluid cells is ``nan``.
    """
    out = np.zeros(P.shape[0])
    for i in range(P.shape[0]):
        m = M[i] > 0
        d = P[i][m] - T[i][m]
        out[i] = torch.sqrt((d ** 2).mean()).item() if m.any() else np.nan
    return out


def spatial_corr(P, T, M):
    """Mean per-case Pearson correlation over fluid cells.

    Cases with fewer than two fluid cells, or with a degenerate (near-constant)
    prediction or target, are skipped. Returns ``nan`` if nothing qualifies.
    """
    cs = []
    for i in range(P.shape[0]):
        m = M[i] > 0
        p, t = P[i][m].numpy(), T[i][m].numpy()
        if p.size < 2 or p.std() < 1e-8 or t.std() < 1e-8:
            continue
        cs.append(float(np.corrcoef(p, t)[0, 1]))
    return float(np.mean(cs)) if cs else np.nan


# ===========================================================================
# WP5 physics-oriented metrics
# ===========================================================================
# region thresholds (u / U_ref) and depths (cells)
WAKE_D, CAN_D = 6, 4
LOW_THR, HI_THR = 0.5, 1.5     # low-speed / high-speed-channel thresholds


def region_masks(solid, fluid):
    """(wake_mask, canyon_mask) over fluid cells, derived from a solid mask.

    Wake: cells within WAKE_D downstream (+x) of a building. Canyon: fluid cells
    flanked by buildings within CAN_D on both spanwise sides.
    """
    wake = np.zeros_like(solid, bool)
    for d in range(1, WAKE_D + 1):
        wake[:, d:] |= solid[:, :-d]                       # building d cells upstream (-x)
    left = np.zeros_like(solid, bool); right = np.zeros_like(solid, bool)
    for d in range(1, CAN_D + 1):
        left[d:, :] |= solid[:-d, :]; right[:-d, :] |= solid[d:, :]
    f = fluid > 0
    return (wake & f), (left & right & f)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else np.nan


def physics_metrics(P, T, M):
    """Aggregate physics-oriented errors over a split (means over cases).

    P, T, M : torch.Tensor of shape (N, Ny, Nx). Returns plane-averaged velocity
    error, wake / canyon / deficit-region RMSE, and low/high-speed-area errors.
    """
    plane, wake, can, defi, loA, hiA = [], [], [], [], [], []
    for i in range(P.shape[0]):
        p, t = P[i].numpy(), T[i].numpy()
        f = (M[i] > 0).numpy()
        solid = (M[i] == 0).numpy()
        if f.sum() == 0:
            continue
        pf, tf = p[f], t[f]
        plane.append(abs(pf.mean() - tf.mean()))                       # plane-averaged velocity error
        wm, cm = region_masks(solid, f)
        if wm.any():
            wake.append(_rmse(p[wm], t[wm]))
        if cm.any():
            can.append(_rmse(p[cm], t[cm]))
        dm = f & (t < LOW_THR)                                          # velocity-deficit region
        if dm.any():
            defi.append(_rmse(p[dm], t[dm]))
        A = f.sum()
        loA.append(abs((f & (p < LOW_THR)).sum() - (f & (t < LOW_THR)).sum()) / A)
        hiA.append(abs((f & (p > HI_THR)).sum() - (f & (t > HI_THR)).sum()) / A)
    def nanmean(v):
        return float(np.nanmean(v)) if len(v) else np.nan
    return dict(plane_avg_err=nanmean(plane), wake_rmse=nanmean(wake),
                canyon_rmse=nanmean(can), deficit_rmse=nanmean(defi),
                low_area_err=nanmean(loA), high_area_err=nanmean(hiA))
