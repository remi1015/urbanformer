"""Regression tests for :mod:`urbanformer.losses` (WP1).

Required invariant: the masked MSE ignores solid cells. The first two tests pin
it directly (perturbing solid cells cannot move the loss; a prediction that is
perfect on fluid cells is exactly zero even when it is wrong on solid cells).
The remaining tests are structural sanity checks on the reduction.
"""

import torch

from urbanformer.losses import masked_mse


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fixture(seed=0):
    """A small (B, Ny, Nx) case with a mixed fluid / solid mask."""
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(2, 8, 8, generator=g)
    target = torch.randn(2, 8, 8, generator=g)
    mask = (torch.rand(2, 8, 8, generator=g) > 0.4).float()
    mask.view(-1)[0] = 1.0  # guarantee at least one fluid cell
    return pred, target, mask


# ---------------------------------------------------------------------------
# masked MSE ignores solid cells (required invariant)
# ---------------------------------------------------------------------------
def test_solid_cells_do_not_affect_loss():
    """Arbitrary changes to pred on solid cells (mask == 0) leave the loss fixed."""
    pred, target, mask = _fixture()
    base = masked_mse(pred, target, mask)

    solid = mask == 0
    perturbed = pred.clone()
    perturbed[solid] += 1e3 * torch.randn(int(solid.sum()))  # wreck the solid cells

    assert torch.equal(masked_mse(perturbed, target, mask), base)


def test_perfect_on_fluid_is_zero_despite_solid_error():
    """Matching target on every fluid cell yields exactly 0, even if solid cells are wrong."""
    pred, target, mask = _fixture(seed=1)
    fluid = mask > 0
    pred = target.clone()
    pred[~fluid] = target[~fluid] + 7.0  # deliberately wrong where it does not count

    assert masked_mse(pred, target, mask).item() == 0.0


# ---------------------------------------------------------------------------
# reduction: structural sanity
# ---------------------------------------------------------------------------
def test_matches_manual_fluid_only_mse():
    """Equals plain MSE computed over the fluid cells alone."""
    pred, target, mask = _fixture(seed=2)
    fluid = mask > 0
    manual = ((pred[fluid] - target[fluid]) ** 2).mean()
    assert torch.allclose(masked_mse(pred, target, mask), manual)


def test_matches_closed_form_reduction():
    """Equals sum(mask * se) / sum(mask) exactly, the definition in wp1.py."""
    pred, target, mask = _fixture(seed=3)
    expected = (mask * (pred - target) ** 2).sum() / mask.sum()
    assert torch.equal(masked_mse(pred, target, mask), expected)


def test_all_fluid_reduces_to_plain_mse():
    """With an all-ones mask the loss is the ordinary MSE."""
    pred, target, mask = _fixture(seed=4)
    ones = torch.ones_like(mask)
    assert torch.allclose(masked_mse(pred, target, ones),
                          ((pred - target) ** 2).mean())



# ---------------------------------------------------------------------------
# WP3 field loss: masked, solid-invariant, zero on a perfect prediction
# ---------------------------------------------------------------------------
def _field_fixture(seed=0, B=2, Ny=16, Nx=16):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(B, Ny, Nx, generator=g)
    target = torch.randn(B, Ny, Nx, generator=g)
    fluid = (torch.rand(B, Ny, Nx, generator=g) > 0.3).float()
    fluid.view(B, -1)[:, 0] = 1.0
    return pred, target, fluid


def test_field_loss_ignores_solid_cells():
    from urbanformer.losses import masked_field_loss, make_radial_bins
    pred, target, fluid = _field_fixture()
    rbin = make_radial_bins(16, 16)
    base, _ = masked_field_loss(pred, target, fluid, rbin)
    solid = fluid == 0
    pred2 = pred.clone(); pred2[solid] += 1e3 * torch.randn(int(solid.sum()))
    after, _ = masked_field_loss(pred2, target, fluid, rbin)
    assert torch.allclose(base, after)


def test_field_loss_zero_on_perfect_prediction():
    from urbanformer.losses import masked_field_loss, make_radial_bins
    _, target, fluid = _field_fixture(seed=1)
    rbin = make_radial_bins(16, 16)
    loss, parts = masked_field_loss(target.clone(), target, fluid, rbin)
    assert abs(float(loss)) < 1e-6
    assert parts["mse"] == 0.0 and parts["grad"] == 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
