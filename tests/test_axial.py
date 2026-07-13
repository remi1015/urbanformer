"""
Regression tests for `uff-axial-fix`.

The bug: `AxialSelfAttention`'s column branch reshaped (B, Ny, Nx, D) straight to
(B * Nx, Ny, D) without permuting to (B, Nx, Ny, D) first. Each "column" sequence
was in fact a row. `reshape` did not raise, because B * Ny * Nx * D factors
identically either way, on any grid, square or not.

`test_column_gather_strides_by_nx` is the single test that would have caught it.
Use a NON-SQUARE grid: on a square grid the buggy gather still produces Ny values
per sequence and a length check passes.
"""
from __future__ import annotations

import pytest
import torch

from urbanformer.models.axial import (
    AxialSelfAttention,
    from_column_sequences,
    to_column_sequences,
    to_row_sequences,
)


NY, NX, D = 4, 16, 3  # deliberately non-square


def _index_grid(B: int = 1) -> torch.Tensor:
    """(B, Ny * Nx, D) where every channel holds the flat row-major index."""
    idx = torch.arange(NY * NX, dtype=torch.float32)
    return idx.view(1, NY * NX, 1).expand(B, NY * NX, D).contiguous()


def test_column_gather_strides_by_nx():
    """Column 0 must be [0, Nx, 2*Nx, ...], not [0, 1, 2, ...].

    Under the bug this returns [0, 1, 2, 3] and the test fails.
    """
    x = _index_grid()
    cols = to_column_sequences(x, NY, NX)  # (Nx, Ny, D)

    assert cols.shape == (NX, NY, D)
    expected_col0 = [0.0, 16.0, 32.0, 48.0]
    assert cols[0, :, 0].tolist() == expected_col0

    # and column j is the same list shifted by j
    for j in range(NX):
        assert cols[j, :, 0].tolist() == [v + j for v in expected_col0]


def test_row_gather_is_contiguous():
    """Sanity check on the branch that was always correct."""
    x = _index_grid()
    rows = to_row_sequences(x, NY, NX)  # (Ny, Nx, D)
    assert rows.shape == (NY, NX, D)
    assert rows[0, :, 0].tolist() == list(range(NX))
    assert rows[1, :, 0].tolist() == list(range(NX, 2 * NX))


def test_column_roundtrip_is_identity():
    """`from_column_sequences` must invert `to_column_sequences` exactly.

    Note: this test PASSES under the bug. Forward and inverse are wrong
    symmetrically, so they still compose to the identity. It is kept as a guard
    against an asymmetric edit, not as a detector for `uff-axial-fix`. The only
    tests that catch the bug are `test_column_gather_strides_by_nx` and
    `test_column_attention_actually_couples_across_rows`.
    """
    B = 2
    x = _index_grid(B) + torch.randn(B, NY * NX, D)
    back = from_column_sequences(to_column_sequences(x, NY, NX), B, NY, NX)
    assert torch.equal(x, back)


def test_row_and_column_gathers_differ_on_nonsquare_grid():
    """The two branches must not see the same sequences.

    Also passes under the bug: the buggy gather still returns shape (Nx, Ny, D)
    and its first sequence still differs from the first row. Kept only to pin the
    shape contract. Ordering, not shape, is what distinguishes the two variants,
    which is precisely why the bug survived every shape check for so long.
    """
    x = _index_grid()
    rows = to_row_sequences(x, NY, NX)
    cols = to_column_sequences(x, NY, NX)
    assert rows.shape == (NY, NX, D)
    assert cols.shape == (NX, NY, D)


def test_column_attention_actually_couples_across_rows():
    """A perturbation at (0, 0) must reach (1, 0), which shares its column.

    Row-only coupling leaves (1, 0) untouched. This is a behavioural test on the
    full module rather than on the gather helpers, so it holds even if someone
    reimplements the reshape.
    """
    torch.manual_seed(0)
    d_model = 8
    attn = AxialSelfAttention(d_model=d_model, n_heads=2, max_grid=64, axial_pos=False).eval()

    base = torch.randn(1, NY * NX, d_model)
    perturbed = base.clone()
    # Perturb ONE grid position, non-uniformly across channels. A constant
    # perturbation would be annihilated by the LayerNorm before the attention.
    perturbed[0, 0, :] += torch.linspace(1.0, -1.0, d_model) * 5.0

    with torch.no_grad():
        out_base = attn(base, NY, NX)
        out_pert = attn(perturbed, NY, NX)

    delta = (out_pert - out_base).abs().view(NY, NX, d_model).sum(-1)

    # Control: the row branch alone already reaches (0, 1), so a passing value
    # here proves nothing on its own.
    same_row_other_col = delta[0, 1].item()
    assert same_row_other_col > 1e-6, "row branch is not attending over rows"

    # The assertion that matters. Under the bug this is exactly 0.
    same_column_other_row = delta[1, 0].item()  # (row 1, col 0)
    assert same_column_other_row > 1e-6, (
        "perturbation did not propagate down the column: "
        "the column branch is not attending over columns"
    )


@pytest.mark.parametrize("ny,nx", [(4, 16), (16, 4), (9, 9), (1, 7), (7, 1)])
def test_shapes_survive_arbitrary_grids(ny, nx):
    attn = AxialSelfAttention(d_model=D, n_heads=1, max_grid=64, axial_pos=True)
    x = torch.randn(2, ny * nx, D)
    assert attn(x, ny, nx).shape == x.shape
