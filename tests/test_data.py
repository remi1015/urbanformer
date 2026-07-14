"""Regression tests for :mod:`urbanformer.data` (WP0).

Two invariants anchor this suite:

1. Every token entry lies in ``[0, 1]`` (per-axis / global-height normalization).
2. No case appears in two splits, and the three splits partition all cases;
   the split is deterministic and reproduces the WP0 notebook exactly.
"""

import numpy as np
import torch
import pytest

from urbanformer import data as D
from urbanformer.morphology import SOLID_CODE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _random_height_map(rng, Ny, Nx, max_buildings=20):
    hm = np.zeros((Ny, Nx), dtype=np.float32)
    for _ in range(int(rng.integers(1, max_buildings))):
        h = int(rng.integers(2, 20))
        w = int(rng.integers(2, 10))
        l = int(rng.integers(2, 10))
        y0 = int(rng.integers(0, Ny - l))
        x0 = int(rng.integers(0, Nx - w))
        hm[y0:y0 + l, x0:x0 + w] = np.maximum(hm[y0:y0 + l, x0:x0 + w], h)
    return hm


# ---------------------------------------------------------------------------
# build_tokens: normalization (required invariant #1)
# ---------------------------------------------------------------------------
def test_tokens_lie_in_unit_range():
    rng = np.random.default_rng(1)
    for _ in range(100):
        Ny = int(rng.integers(30, 80))
        Nx = int(rng.integers(30, 80))
        hm = _random_height_map(rng, Ny, Nx)
        h_ref = float(hm.max()) if hm.max() > 0 else 1.0
        tokens = D.build_tokens(hm, h_ref)
        assert tokens.shape[1] == 5
        assert tokens.dtype == np.float32
        if tokens.size:
            assert tokens.min() >= 0.0
            assert tokens.max() <= 1.0 + 1e-6


def test_token_normalization_axes():
    """One rectangular building: check each token entry against hand math."""
    Ny, Nx = 50, 40
    hm = np.zeros((Ny, Nx), dtype=np.float32)
    hm[10:20, 8:16] = 12.0                 # rows 10..19 (l_y=10), cols 8..15 (l_x=8)
    h_ref = 24.0
    tok = D.build_tokens(hm, h_ref)
    assert tok.shape == (1, 5)
    x_c, y_c, lx, ly, h = tok[0]
    assert np.isclose(x_c, ((8 + 15) / 2) / Nx)
    assert np.isclose(y_c, ((10 + 19) / 2) / Ny)
    assert np.isclose(lx, 8 / Nx)
    assert np.isclose(ly, 10 / Ny)
    assert np.isclose(h, 12.0 / h_ref)


def test_empty_map_yields_zero_by_five_tokens():
    hm = np.zeros((30, 30), dtype=np.float32)
    tok = D.build_tokens(hm, h_ref=1.0)
    assert tok.shape == (0, 5)


# ---------------------------------------------------------------------------
# case_fields
# ---------------------------------------------------------------------------
def test_case_fields_masks_and_velocity():
    Ny, Nx = 16, 16
    hm = np.zeros((Ny, Nx), dtype=np.float32)
    hm[4:8, 4:8] = 5.0
    geom = np.where(hm > 0, SOLID_CODE, 0).astype(int)
    y_vel = np.full((Ny, Nx), 2.0, dtype=np.float32)
    utau = 0.5
    f = D.case_fields(hm, geom, y_vel, utau)

    assert np.allclose(f['U_mid'], 2.0 / 0.5)                       # y / utau
    assert np.array_equal(f['footprint_mask'], (hm > 0).astype(np.float32))
    assert np.array_equal(f['fluid_mask_mid'], (geom != SOLID_CODE).astype(np.float32))
    # solid cells are excluded from the fluid mask
    assert f['fluid_mask_mid'][5, 5] == 0.0
    assert f['fluid_mask_mid'][0, 0] == 1.0
    for v in f.values():
        assert v.dtype == np.float32


# ---------------------------------------------------------------------------
# make_splits: no leakage (required invariant #2)
# ---------------------------------------------------------------------------
def test_splits_are_a_disjoint_partition():
    n = 5225
    s = D.make_splits(n)
    train, val, test = set(s['train']), set(s['val']), set(s['test'])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(range(n))         # covers every case exactly once


def test_split_sizes_follow_fractions():
    n = 5225
    s = D.make_splits(n, fractions=(0.70, 0.15, 0.15))
    assert len(s['train']) == int(0.70 * n)            # 3657
    assert len(s['val']) == int(0.15 * n)              # 783
    assert len(s['train']) + len(s['val']) + len(s['test']) == n


def test_split_is_deterministic():
    a = D.make_splits(2000, seed=42)
    b = D.make_splits(2000, seed=42)
    for k in ('train', 'val', 'test'):
        assert np.array_equal(a[k], b[k])
    c = D.make_splits(2000, seed=7)
    assert not np.array_equal(a['train'], c['train'])   # seed actually matters


# ---------------------------------------------------------------------------
# UNetMidDataset: tensor contract (WP1)
# ---------------------------------------------------------------------------
def _write_case(case_dir, Ny, Nx, rng):
    case_dir.mkdir(parents=True)
    hm = (rng.random((Ny, Nx)) * 10).astype(np.float64)
    np.save(case_dir / "height_map.npy", hm)
    np.save(case_dir / "footprint_mask.npy", (hm > 5).astype(np.int64))
    np.save(case_dir / "U_mid.npy", rng.standard_normal((Ny, Nx)))
    np.save(case_dir / "fluid_mask_mid.npy", (hm <= 5).astype(np.int64))


def test_unet_dataset_item_contract(tmp_path):
    rng = np.random.default_rng(0)
    Ny, Nx = 40, 50
    _write_case(tmp_path / "case_0", Ny, Nx, rng)
    ds = D.UNetMidDataset([tmp_path / "case_0"])

    assert len(ds) == 1
    x, y, mask = ds[0]
    assert tuple(x.shape) == (4, Ny, Nx)
    assert tuple(y.shape) == (Ny, Nx)
    assert tuple(mask.shape) == (Ny, Nx)
    assert x.dtype == y.dtype == mask.dtype == torch.float32
    # fluid mask is binary
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_unet_dataset_coordinate_channels_are_cell_centered(tmp_path):
    rng = np.random.default_rng(1)
    Ny, Nx = 10, 8
    _write_case(tmp_path / "case_0", Ny, Nx, rng)
    x, _, _ = D.UNetMidDataset([tmp_path / "case_0"])[0]
    x_grid, y_grid = x[2], x[3]
    # channel 2 varies along columns as (col + 0.5) / Nx, constant down rows
    assert torch.allclose(x_grid[0], (torch.arange(Nx, dtype=torch.float32) + 0.5) / Nx)
    assert torch.allclose(x_grid[0], x_grid[-1])
    # channel 3 varies along rows as (row + 0.5) / Ny, constant across columns
    assert torch.allclose(y_grid[:, 0], (torch.arange(Ny, dtype=torch.float32) + 0.5) / Ny)
    assert torch.allclose(y_grid[:, 0], y_grid[:, -1])


# ---------------------------------------------------------------------------
# TokenDataset + collate_fn (WP2)
# ---------------------------------------------------------------------------
def _write_token_case(case_dir, Ny, Nx, n_buildings, rng):
    case_dir.mkdir(parents=True)
    tokens = rng.random((n_buildings, 5)).astype(np.float32)
    np.save(case_dir / "building_tokens.npy", tokens)
    np.save(case_dir / "U_mid.npy", rng.standard_normal((Ny, Nx)).astype(np.float32))
    fluid = (rng.random((Ny, Nx)) > 0.3).astype(np.float32)
    np.save(case_dir / "fluid_mask_mid.npy", fluid)
    return int(fluid.sum()), n_buildings


def test_token_dataset_train_mode_contract(tmp_path):
    rng = np.random.default_rng(0)
    n_fluid, n_b = _write_token_case(tmp_path / "case_0", 40, 40, 12, rng)
    ds = D.TokenDataset([tmp_path / "case_0"], mode="train", K_pts=200)
    tok, qxy, tgt = ds[0]
    assert tuple(tok.shape) == (n_b, 5)
    k = min(200, n_fluid)
    assert tuple(qxy.shape) == (k, 2)
    assert tuple(tgt.shape) == (k,)
    assert tok.dtype == qxy.dtype == tgt.dtype == torch.float32


def test_token_dataset_eval_mode_returns_all_fluid_points(tmp_path):
    rng = np.random.default_rng(1)
    n_fluid, _ = _write_token_case(tmp_path / "case_0", 30, 30, 8, rng)
    ds = D.TokenDataset([tmp_path / "case_0"], mode="eval", K_pts=200)
    _, qxy, tgt = ds[0]
    assert qxy.shape[0] == n_fluid          # all fluid points, no subsampling
    assert tgt.shape[0] == n_fluid


def test_collate_pads_and_builds_padding_mask():
    # three cases with different token counts
    batch = [
        (torch.ones(3, 5), torch.rand(10, 2), torch.rand(10)),
        (torch.ones(5, 5), torch.rand(10, 2), torch.rand(10)),
        (torch.ones(1, 5), torch.rand(10, 2), torch.rand(10)),
    ]
    tok_pad, pmask, qxy, tgt = D.collate_fn(batch)
    assert tuple(tok_pad.shape) == (3, 5, 5)     # padded to max_nb = 5
    assert tuple(pmask.shape) == (3, 5) and pmask.dtype == torch.bool
    # real-token slots are False, padding slots True
    assert pmask[0].tolist() == [False, False, False, True, True]
    assert pmask[1].tolist() == [False] * 5
    assert pmask[2].tolist() == [False, True, True, True, True]
    assert tuple(qxy.shape) == (3, 10, 2)
    assert tuple(tgt.shape) == (3, 10)



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
