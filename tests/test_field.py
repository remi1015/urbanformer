"""Regression tests for :mod:`urbanformer.models.field` (WP3, flagship UF-F).

Pins the architectural fingerprint (1,633,969 params -- the count behind the
flagship R2 = 0.8461), the forward contract, the attention-return path, and the
set-invariance of the token encoder.
"""

import numpy as np
import torch

from urbanformer.models.field import (
    NUM_QUERY_FEATS,
    PATCH_FEAT_DIM,
    RESIDUAL_DEPTH,
    TOKEN_DIM,
    UrbanFormerField,
)


def _grid_xy(Ny, Nx):
    gx, gy = np.meshgrid((np.arange(Nx) + 0.5) / Nx, (np.arange(Ny) + 0.5) / Ny)
    return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)


def _inputs(B=2, N=10, Ny=16, Nx=16):
    tok = torch.rand(B, N, TOKEN_DIM)
    pad = torch.zeros(B, N, dtype=torch.bool)
    qxy = torch.from_numpy(_grid_xy(Ny, Nx))[None].repeat(B, 1, 1)
    qf = torch.rand(B, Ny * Nx, NUM_QUERY_FEATS)
    pa = torch.rand(B, Ny * Nx, PATCH_FEAT_DIM)
    return tok, pad, qxy, qf, pa, Ny, Nx


# ---------------------------------------------------------------------------
# architectural fingerprint
# ---------------------------------------------------------------------------
def test_param_count_matches_flagship():
    assert sum(p.numel() for p in UrbanFormerField().parameters()) == 1_633_969


def test_forward_shape():
    tok, pad, qxy, qf, pa, Ny, Nx = _inputs()
    out = UrbanFormerField().eval()(tok, pad, qxy, qf, pa, Ny, Nx)
    assert tuple(out.shape) == (2, Ny * Nx)


def test_return_attn_gives_one_map_per_block():
    tok, pad, qxy, qf, pa, Ny, Nx = _inputs()
    m = UrbanFormerField().eval()
    memory, pad2 = m.encode(tok, pad)
    pred, attns = m.decode(memory, pad2, qxy, tok[..., :2], qf, pa, Ny, Nx, return_attn=True)
    assert tuple(pred.shape) == (2, Ny * Nx)
    assert len(attns) == RESIDUAL_DEPTH


# ---------------------------------------------------------------------------
# set-invariance: token order must not change the field
# ---------------------------------------------------------------------------
def test_token_permutation_invariance():
    torch.manual_seed(0)
    m = UrbanFormerField().eval()
    tok, pad, qxy, qf, pa, Ny, Nx = _inputs(B=1, N=8)
    with torch.no_grad():
        out_ref = m(tok, pad, qxy, qf, pa, Ny, Nx)
        perm = torch.randperm(tok.shape[1])
        out_perm = m(tok[:, perm], pad[:, perm], qxy, qf, pa, Ny, Nx)
    assert torch.allclose(out_ref, out_perm, atol=1e-4)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
