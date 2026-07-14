"""Regression tests for :mod:`urbanformer.metrics` (WP5).

Required invariant: ``R2(y, y) == 1.0`` -- a perfect prediction scores exactly 1.
The suite also pins that solid cells are excluded from every metric and that the
reductions match hand computation.
"""

import numpy as np
import torch

from urbanformer.metrics import field_metrics, per_case_rmse, spatial_corr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fixture(seed=0, N=4, Ny=8, Nx=8):
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(N, Ny, Nx, generator=g)
    T = torch.randn(N, Ny, Nx, generator=g)
    M = (torch.rand(N, Ny, Nx, generator=g) > 0.4).float()
    M.view(N, -1)[:, 0] = 1.0  # guarantee >=1 fluid cell per case
    return P, T, M


# ---------------------------------------------------------------------------
# R2(y, y) == 1 (required invariant)
# ---------------------------------------------------------------------------
def test_perfect_prediction_scores_r2_one():
    _, T, M = _fixture()
    m = field_metrics(T, T, M)
    assert m["R2"] == 1.0
    assert m["RMSE"] == 0.0
    assert m["MAE"] == 0.0
    assert m["relL2"] == 0.0


def test_per_case_rmse_of_perfect_prediction_is_zero():
    _, T, M = _fixture()
    assert np.allclose(per_case_rmse(T, T, M), 0.0)


def test_spatial_corr_of_perfect_prediction_is_one():
    _, T, M = _fixture()
    # T is random, so each case is non-degenerate and correlates perfectly with itself.
    assert np.isclose(spatial_corr(T, T, M), 1.0)


# ---------------------------------------------------------------------------
# solid cells excluded
# ---------------------------------------------------------------------------
def test_solid_cells_do_not_affect_field_metrics():
    P, T, M = _fixture(seed=1)
    base = field_metrics(P, T, M)
    solid = M == 0
    P2 = P.clone()
    P2[solid] += 1e3 * torch.randn(int(solid.sum()))
    after = field_metrics(P2, T, M)
    assert base == after


def test_no_fluid_case_is_nan_per_case():
    P, T, M = _fixture(seed=2, N=2)
    M[0] = 0.0  # first case has no fluid cells
    out = per_case_rmse(P, T, M)
    assert np.isnan(out[0])
    assert not np.isnan(out[1])


# ---------------------------------------------------------------------------
# reductions match hand computation
# ---------------------------------------------------------------------------
def test_r2_matches_closed_form():
    P, T, M = _fixture(seed=3)
    fluid = M > 0
    fp, ft = P[fluid], T[fluid]
    ss_res = ((fp - ft) ** 2).sum()
    ss_tot = ((ft - ft.mean()) ** 2).sum().clamp_min(1e-12)
    expected = (1 - ss_res / ss_tot).item()
    assert field_metrics(P, T, M)["R2"] == expected


# ---------------------------------------------------------------------------
# WP5 physics metrics
# ---------------------------------------------------------------------------
def test_physics_metrics_zero_on_perfect_prediction():
    from urbanformer.metrics import physics_metrics
    _, T, M = _fixture(seed=7, N=3, Ny=20, Nx=20)
    out = physics_metrics(T.clone(), T, M)
    # every error term is 0 for a perfect prediction (nan only if a region is absent)
    for k, v in out.items():
        assert v == 0.0 or np.isnan(v), (k, v)


def test_region_masks_are_within_fluid():
    from urbanformer.metrics import region_masks
    rng = np.random.default_rng(0)
    solid = rng.random((30, 30)) > 0.8
    fluid = (~solid).astype(float)
    wake, canyon = region_masks(solid, fluid)
    assert wake.dtype == bool and canyon.dtype == bool
    assert not (wake & (fluid == 0)).any()      # never marks solid cells
    assert not (canyon & (fluid == 0)).any()



if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
