"""Regression tests for :mod:`urbanformer.morphology` (WP0).

Two invariants anchor this suite:

1. ``row_canyons`` treats the streamwise row as periodic, so an air run that
   wraps the x = 0 / x = Nx seam is counted once, not split in two.
2. The modified alignedness satisfies ``gamma_m* >= gamma_s`` by construction,
   because every penetrating row is credited with the layout's strongest
   sheltered-canyon ratio (>= its own contribution to gamma_s).
"""

import numpy as np
import pytest
from scipy.ndimage import label

from urbanformer import morphology as M


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _random_height_map(rng, Ny, Nx, max_buildings=25):
    """Random non-negative height map of stacked rectangular buildings."""
    hm = np.zeros((Ny, Nx), dtype=np.float32)
    for _ in range(int(rng.integers(3, max_buildings))):
        h = int(rng.integers(2, 20))
        w = int(rng.integers(2, 12))
        l = int(rng.integers(2, 12))
        y0 = int(rng.integers(0, Ny - l))
        x0 = int(rng.integers(0, Nx - w))
        hm[y0:y0 + l, x0:x0 + w] = np.maximum(hm[y0:y0 + l, x0:x0 + w], h)
    return hm


# ---------------------------------------------------------------------------
# row_canyons: periodic seam (required invariant #1)
# ---------------------------------------------------------------------------
def test_row_canyons_wraps_the_seam():
    """An air run spanning the x=0/x=Nx boundary is a single canyon.

    Naive (non-periodic) labeling would report two separate runs; the periodic
    rotation must collapse them into one run of the summed length.
    """
    Nx = 10
    air = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)   # air wraps 8,9,0,1
    height = np.array([0, 0, 5, 5, 5, 5, 5, 5, 0, 0], dtype=float)

    assert label(air)[1] == 2                      # naive baseline: two runs
    canyons = M.row_canyons(air, height, Nx)
    assert len(canyons) == 1                        # periodic: one canyon
    run_len, h_ahead = canyons[0]
    assert run_len == 4                             # 8,9 + 0,1
    assert h_ahead == 5.0                           # building at the seam


def test_row_canyons_open_row_is_penetrating():
    Nx = 12
    air = np.ones(Nx, dtype=bool)
    height = np.zeros(Nx, dtype=float)
    assert M.row_canyons(air, height, Nx) == [(Nx, 0.0)]


def test_row_canyons_solid_row_has_no_canyon():
    Nx = 12
    air = np.zeros(Nx, dtype=bool)
    height = np.full(Nx, 5.0)
    assert M.row_canyons(air, height, Nx) == []


def test_row_canyons_interior_run_not_wrapped():
    """A run bounded by buildings on both sides, away from the seam."""
    Nx = 10
    air = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    height = np.array([7, 7, 0, 0, 0, 3, 3, 3, 3, 3], dtype=float)
    canyons = M.row_canyons(air, height, Nx)
    assert canyons == [(3, 3.0)]                    # length 3, building of height 3 ahead


# ---------------------------------------------------------------------------
# alignedness: gamma_m* >= gamma_s (required invariant #2)
# ---------------------------------------------------------------------------
def test_gamma_mstar_ge_gamma_s_random():
    rng = np.random.default_rng(1)
    for _ in range(200):
        Ny = int(rng.integers(30, 80))
        Nx = int(rng.integers(30, 80))
        hm = _random_height_map(rng, Ny, Nx)
        fp = hm > 0
        lambda_p = float(fp.sum() / hm.size)
        a = M.compute_alignedness(fp, hm, lambda_p)
        assert a['gamma_m_star'] >= a['gamma_s'] - 1e-9


def test_gamma_mstar_ge_gamma_s_with_penetrating_streets():
    """Force penetrating rows: gamma_m* must still dominate gamma_s.

    Rows 0-2 are fully open (penetrating). The rest hold a bounded canyon so
    gamma_s is strictly positive; gamma_m* credits the open rows with the
    layout's max C/H, keeping the inequality strict-or-equal.
    """
    Ny, Nx = 20, 40
    hm = np.zeros((Ny, Nx), dtype=np.float32)
    # bounded canyons in lower rows: two buildings with air between them
    for r in range(5, Ny):
        hm[r, 5:10] = 6.0
        hm[r, 25:30] = 6.0
    fp = hm > 0
    lambda_p = float(fp.sum() / hm.size)
    a = M.compute_alignedness(fp, hm, lambda_p)
    assert a['gamma_s'] > 0.0                       # bounded canyons contribute
    assert a['gamma_m_star'] >= a['gamma_s'] - 1e-9


# ---------------------------------------------------------------------------
# building extraction and descriptor packaging
# ---------------------------------------------------------------------------
def test_extract_buildings_counts_and_geometry():
    hm = np.zeros((20, 20), dtype=np.float32)
    hm[2:5, 2:6] = 4.0      # one building: l_y=3 (rows), l_x=4 (cols), h=4
    hm[10:12, 14:15] = 9.0  # another:      l_y=2,         l_x=1,      h=9
    buildings, footprint, n = M.extract_buildings(hm)
    assert n == 2
    assert int(footprint.sum()) == 3 * 4 + 2 * 1
    heights = sorted(b['h'] for b in buildings)
    assert heights == [4.0, 9.0]
    by_h = {b['h']: b for b in buildings}
    assert (by_h[4.0]['l_x'], by_h[4.0]['l_y']) == (4.0, 3.0)
    assert (by_h[9.0]['l_x'], by_h[9.0]['l_y']) == (1.0, 2.0)


def test_empty_map_density_and_height_block_is_zero():
    """A void has no buildings, so every density/height descriptor is zero.

    Alignedness is deliberately NOT asserted zero here: with no buildings every
    row is one unobstructed penetrating street, so gamma_m == gamma_p == 1 by
    definition (see the companion test below). Only the density/height block and
    the building count collapse to zero.
    """
    hm = np.zeros((30, 30), dtype=np.float32)
    desc, align, stats = M.compute_global_descriptors(hm)
    assert desc.shape == (8,) and align.shape == (5,)
    assert stats['n_buildings'] == 0
    # lambda_p, lambda_f, h_m, h_rms, h_skew, h_kurt, h_max  (all but gamma_m@idx6)
    density_height = np.delete(desc, M.MORPHOLOGY_KEYS.index('gamma_m'))
    assert np.allclose(density_height, 0.0)


def test_empty_map_reads_as_maximally_aligned():
    """Documents the void degenerate case: gamma_m == gamma_p == 1.

    Every row is a full-width penetrating street, so the profile gamma(y) is 1
    everywhere. gamma_s / gamma_c (bounded canyons only) and gamma_m* (credited
    from gamma_s, which is 0) stay zero.
    """
    hm = np.zeros((30, 30), dtype=np.float32)
    _, align, _ = M.compute_global_descriptors(hm)
    a = dict(zip(M.ALIGNEDNESS_KEYS, align))
    assert np.isclose(a['gamma_m'], 1.0)
    assert np.isclose(a['gamma_p'], 1.0)
    assert np.isclose(a['gamma_s'], 0.0)
    assert np.isclose(a['gamma_c'], 0.0)
    assert np.isclose(a['gamma_m_star'], 0.0)


def test_descriptor_vectors_have_canonical_order():
    rng = np.random.default_rng(7)
    hm = _random_height_map(rng, 78, 78)
    desc, align, stats = M.compute_global_descriptors(hm)
    assert desc.dtype == np.float32 and align.dtype == np.float32
    assert len(M.MORPHOLOGY_KEYS) == desc.size == 8
    assert len(M.ALIGNEDNESS_KEYS) == align.size == 5
    # gamma_m appears in both vectors; the shared entry must agree
    gamma_m_from_desc = desc[M.MORPHOLOGY_KEYS.index('gamma_m')]
    gamma_m_from_align = align[M.ALIGNEDNESS_KEYS.index('gamma_m')]
    assert np.isclose(gamma_m_from_desc, gamma_m_from_align)


def test_lambda_p_matches_footprint_fraction():
    hm = np.zeros((10, 10), dtype=np.float32)
    hm[0:5, 0:4] = 3.0       # 20 solid cells out of 100
    desc, _, _ = M.compute_global_descriptors(hm)
    assert np.isclose(desc[M.MORPHOLOGY_KEYS.index('lambda_p')], 0.20)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
