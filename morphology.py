"""WP0 building extraction and morphological descriptors.

Ported faithfully from the WP0 preprocessing notebook. Every function here is
pure: it takes arrays and returns values, with no file I/O. The per-case
orchestration that loads the raw ``.npz`` and writes per-case ``.npy`` lives in
the preprocessing script, not in this module.

Descriptor families
-------------------
Canonical 8-vector written to ``global_descriptors.npy`` (roadmap order)::

    lambda_p, lambda_f, h_m, h_rms, h_skew, h_kurt, gamma_m, h_max

Full alignedness family written to ``alignedness_variants.npy``
(Lu et al. 2023)::

    gamma_m, gamma_m_star, gamma_s, gamma_p, gamma_c

References
----------
Alas et al.        -- per-building height statistics and plan/frontal densities.
Lu et al. (2023)   -- alignedness family; equation numbers are cited inline.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis

# --- configuration ---------------------------------------------------------
SOLID_CODE = 8          # geom value marking a solid (building) cell at the mid-plane
DX = 1.0                # horizontal cell size in the SAME length unit as heights.
                        # The LBM grid is isotropic in lattice units, so DX = 1.
                        # If heights are in metres while x,y are in cells, set
                        # DX = cell size (m).

# canonical morphology vector saved to global_descriptors.npy (roadmap order)
MORPHOLOGY_KEYS = ['lambda_p', 'lambda_f', 'h_m', 'h_rms',
                   'h_skew', 'h_kurt', 'gamma_m', 'h_max']
# full alignedness family saved separately for ablation / analysis
ALIGNEDNESS_KEYS = ['gamma_m', 'gamma_m_star', 'gamma_s', 'gamma_p', 'gamma_c']


def extract_buildings(height_map):
    """Connected-component buildings (footprint = height > 0).

    Returns ``(buildings, footprint, n_buildings)`` where each building is a
    dict with pixel-space geometry and the raw roof height.
    """
    footprint = height_map > 0
    labeled, n = label(footprint)
    buildings = []
    for b in range(1, n + 1):
        mask = labeled == b
        rows, cols = np.where(mask)
        buildings.append({
            'x_center': float(cols.mean()),
            'y_center': float(rows.mean()),
            'l_x': float(cols.max() - cols.min() + 1),   # streamwise (x) extent, px
            'l_y': float(rows.max() - rows.min() + 1),   # spanwise   (y) extent, px
            'h':   float(height_map[mask].max()),        # raw roof height
        })
    return buildings, footprint, n


def safe_skew(values):
    v = np.asarray(values, dtype=float)
    if v.size < 3 or np.allclose(v.var(), 0.0):
        return 0.0
    return float(sp_skew(v))


def safe_kurtosis(values, fisher=True):
    # fisher=True -> excess kurtosis (normal == 0); set fisher=False for Pearson.
    v = np.asarray(values, dtype=float)
    if v.size < 4 or np.allclose(v.var(), 0.0):
        return 0.0
    return float(sp_kurtosis(v, fisher=fisher))


def row_canyons(air_row, height_row, Nx):
    """Air canyons in a single streamwise row, under PERIODIC boundaries.

    Wind is along +x (axis 1). Returns a list of ``(run_length_px, H_ahead)``:

      * ``run_length_px`` : streamwise length of the uninterrupted air run (C)
      * ``H_ahead``       : roof height of the building immediately downstream
                            (H); ``H_ahead == 0`` marks a fully penetrating
                            street.

    Periodicity is handled by rotating the row so a building sits at index 0,
    which prevents any air run from being split across the x = 0 / x = Nx seam.
    """
    if air_row.all():
        return [(Nx, 0.0)]            # one penetrating street, no building ahead
    if not air_row.any():
        return []                     # solid row, no canyon

    shift   = int(np.where(~air_row)[0][0])     # first building cell
    air_rot = np.roll(air_row, -shift)          # building -> index 0 => no wrap runs
    h_rot   = np.roll(height_row, -shift)

    labeled, n = label(air_rot)
    out = []
    for s in range(1, n + 1):
        cols = np.where(labeled == s)[0]
        nxt  = (cols[-1] + 1) % Nx              # downstream cell (a building)
        out.append((len(cols), float(h_rot[nxt])))
    return out


def compute_alignedness(footprint, height_map, lambda_p, dx=DX):
    """Five alignedness descriptors (Lu et al. 2023), for wind along +x."""
    Ny, Nx = height_map.shape
    air = ~footprint

    gm_y = np.zeros(Ny)       # gamma(y) profile, Eq. 1a  (normalized by Lx)
    gc_y = np.zeros(Ny)       # canyon profile,  Eq. A3  (non-penetrating only)
    gs_y = np.zeros(Ny)       # sheltering C/H,  Eq. A1  (non-penetrating only)
    penetrating = np.zeros(Ny, dtype=bool)

    for r in range(Ny):
        canyons = row_canyons(air[r], height_map[r], Nx)
        if not canyons:
            continue
        gm_best = gc_best = gs_best = 0.0
        for run_len, H_ahead in canyons:
            run_norm = run_len / Nx
            gm_best = max(gm_best, run_norm)                       # incl. penetrating
            if H_ahead > 0:                                        # building-bounded canyon
                gc_best = max(gc_best, run_norm)
                gs_best = max(gs_best, (run_len * dx) / H_ahead)   # aspect ratio C/H
            else:
                penetrating[r] = True                             # penetrating street
        gm_y[r] = gm_best
        gc_y[r] = gc_best
        gs_y[r] = gs_best

    gamma_m = float(gm_y.mean())                                  # Eq. 1b
    gamma_c = float(gc_y.mean())                                  # Eq. A3
    gamma_s = float(gs_y.mean())                                  # Eq. A1 (penetrating excluded)

    # Eq. 2: modified alignedness includes penetrating streets. C/H is undefined
    # as H -> 0, so a penetrating row is credited with the strongest sheltered-
    # canyon ratio found in the layout. This keeps it finite and guarantees
    # gamma_m* >= gamma_s.
    ref = float(gs_y.max())
    gstar_y = np.where(penetrating, ref, gs_y)
    gamma_m_star = float(gstar_y.mean())                         # Eq. 2b

    # Eq. A2: principal alignedness = mean of the upper lambda_p fraction of gamma(y).
    n_top = max(1, int(np.ceil(lambda_p * Ny)))
    gamma_p = float(np.sort(gm_y)[-n_top:].mean())

    return {
        'gamma_m': gamma_m, 'gamma_m_star': gamma_m_star,
        'gamma_s': gamma_s, 'gamma_p': gamma_p, 'gamma_c': gamma_c,
    }


def compute_global_descriptors(height_map, dx=DX):
    """Canonical 8-vector and alignedness 5-vector for one height map.

    Mirrors exactly what the WP0 loop writes to disk per case, minus the
    externally supplied fields (``case_idx``, ``U_ref``).

    Returns
    -------
    descriptors : (8,) float32   -- MORPHOLOGY_KEYS order
    alignedness : (5,) float32   -- ALIGNEDNESS_KEYS order
    stats : dict                 -- named scalars + n_buildings, Ny, Nx (metadata)
    """
    Ny, Nx = height_map.shape
    buildings, footprint, n_bld = extract_buildings(height_map)
    heights = np.array([b['h'] for b in buildings], dtype=float)
    N = len(buildings)

    # --- densities -------------------------------------------------------
    # plan area density: pixel-exact footprint fraction (== sum l_x*l_y / A for
    # non-overlapping rectangular footprints)
    lambda_p = float(footprint.sum() / (Nx * Ny))
    # frontal area density: gross sum of per-building (height * spanwise width) / A
    lambda_f = float(sum(b['h'] * b['l_y'] for b in buildings) / (Nx * Ny))

    # --- per-building height statistics (Alas et al.) --------------------
    h_m    = float(heights.mean()) if N else 0.0     # mean building height
    h_rms  = float(heights.std())  if N else 0.0     # std about the mean = sigma_H
    h_skew = safe_skew(heights)
    h_kurt = safe_kurtosis(heights, fisher=True)
    h_max  = float(heights.max())  if N else 0.0

    # --- alignedness family (Lu et al. 2023) ----------------------------
    align = compute_alignedness(footprint, height_map, lambda_p, dx=dx)

    # --- canonical descriptor vector (roadmap order) --------------------
    descriptors = np.array([
        lambda_p, lambda_f, h_m, h_rms,
        h_skew, h_kurt, align['gamma_m'], h_max,
    ], dtype=np.float32)

    # --- full alignedness family (for ablation / analysis) --------------
    alignedness = np.array([align[k] for k in ALIGNEDNESS_KEYS], dtype=np.float32)

    stats = {
        'h_m': h_m, 'h_rms': h_rms, 'h_max': h_max,
        'n_buildings': int(n_bld), 'Ny': int(Ny), 'Nx': int(Nx),
    }
    return descriptors, alignedness, stats
