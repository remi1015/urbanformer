"""Data layer: building tokens, raw per-case fields, and layout splits.

WP0 scope. This module holds the pure, testable data-preparation functions the
WP0 preprocessing notebook calls:

* :func:`build_tokens`  -- per-building tokens normalized to ``[0, 1]``.
* :func:`case_fields`   -- the raw per-case fields that need no morphology
                           (height map, normalized mid-plane velocity, footprint
                           and fluid masks).
* :func:`make_splits`   -- 70 / 15 / 15 split by full urban layout, no
                           grid-point leakage, reproducible from a seed.

The train-time ``torch`` datasets are ported here as they are reached:

* :class:`UNetMidDataset` (WP1) -- raster inputs for the U-Net baseline.
* :class:`TokenDataset` + :func:`collate_fn` (WP2) -- building tokens plus
  sampled fluid query points, padded/collated for the pooled Transformers.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from urbanformer.morphology import SOLID_CODE, extract_buildings

# token layout, in order (see build_tokens)
TOKEN_KEYS = ['x_center', 'y_center', 'l_x', 'l_y', 'h']
TOKEN_DIM = len(TOKEN_KEYS)


def build_tokens(height_map, h_ref):
    """Per-building tokens ``[x_center, y_center, l_x, l_y, h]``, all in ``[0, 1]``.

    x / width are normalized by ``Nx`` (Lx), y / length by ``Ny`` (Ly), and the
    roof height by the global ``h_ref``. Tokens are returned unshuffled; the
    train-time ``Dataset`` shuffles them for set-invariance.

    Returns
    -------
    tokens : (N_buildings, 5) float32   -- empty maps yield shape (0, 5).
    """
    Ny, Nx = height_map.shape
    buildings, _, _ = extract_buildings(height_map)
    tokens = np.array([
        [b['x_center'] / Nx,
         b['y_center'] / Ny,
         b['l_x'] / Nx,
         b['l_y'] / Ny,
         b['h'] / h_ref]
        for b in buildings
    ], dtype=np.float32).reshape(-1, 5)
    return tokens


def case_fields(height_map, geom, y_vel, utau):
    """Raw per-case fields that need no morphology.

    Parameters
    ----------
    height_map : (Ny, Nx)   raw height map.
    geom       : (Ny, Nx)   cell-type codes (``SOLID_CODE`` marks a building).
    y_vel      : (Ny, Nx)   raw mid-plane streamwise velocity.
    utau       : float      friction velocity, used as ``U_ref``.

    Returns
    -------
    dict with float32 arrays:
        ``height_map``, ``U_mid`` (= y_vel / utau), ``footprint_mask``
        (height > 0), ``fluid_mask_mid`` (geom != SOLID_CODE).
    """
    utau = float(utau)
    return {
        'height_map':     height_map.astype(np.float32),
        'U_mid':          (y_vel / utau).astype(np.float32),
        'footprint_mask': (height_map > 0).astype(np.float32),
        'fluid_mask_mid': (geom != SOLID_CODE).astype(np.float32),
    }


def make_splits(n_cases, seed=42, fractions=(0.70, 0.15, 0.15)):
    """Split case indices 70 / 15 / 15 by full urban layout.

    Splitting on whole layouts (not grid points) prevents leakage: no case ever
    lands in two splits. Reproducible from ``seed`` and identical to the WP0
    notebook's split so already-generated split files stay valid.

    The test fraction is the remainder, so ``fractions[2]`` is advisory; train
    and val counts use ``floor(fraction * n_cases)`` exactly as the notebook does.

    Returns
    -------
    dict[str, np.ndarray]  -- integer index arrays keyed 'train', 'val', 'test'.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_cases)
    rng.shuffle(indices)

    n_train = int(fractions[0] * n_cases)
    n_val   = int(fractions[1] * n_cases)

    return {
        'train': indices[:n_train],
        'val':   indices[n_train:n_train + n_val],
        'test':  indices[n_train + n_val:],
    }


class UNetMidDataset(Dataset):
    """One case -> (x, y, mask) for the WP1 U-Net baseline.

    x    : (4, Ny, Nx) float32 -- [height_map, footprint_mask, x_grid, y_grid]
    y    : (Ny, Nx)    float32 -- U_mid (u / u_ref)
    mask : (Ny, Nx)    float32 -- fluid_mask_mid (1 = fluid, 0 = building)

    Coordinate channels use the cell-centered convention
    ``(col + 0.5) / Nx`` and ``(row + 0.5) / Ny``. Each ``case_dir`` holds the
    per-case ``.npy`` fields written by WP0.
    """

    def __init__(self, case_dirs):
        self.case_dirs = case_dirs

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]

        # Force float32 so the tensors match the (float32) model weights.
        hmap      = np.load(case_dir / "height_map.npy").astype(np.float32)
        footprint = np.load(case_dir / "footprint_mask.npy").astype(np.float32)
        U_mid     = np.load(case_dir / "U_mid.npy").astype(np.float32)
        fluid     = np.load(case_dir / "fluid_mask_mid.npy").astype(np.float32)

        Ny, Nx = hmap.shape

        # Cell-centered normalized coordinate channels.
        x_ch = (np.arange(Nx, dtype=np.float32) + 0.5) / Nx
        y_ch = (np.arange(Ny, dtype=np.float32) + 0.5) / Ny
        x_grid = np.broadcast_to(x_ch[np.newaxis, :], (Ny, Nx))
        y_grid = np.broadcast_to(y_ch[:, np.newaxis], (Ny, Nx))

        x = np.stack([hmap, footprint, x_grid, y_grid], axis=0).astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(U_mid),
            torch.from_numpy(fluid),
        )


class TokenDataset(Dataset):
    """Building-token dataset for the pooled Transformer (WP2/WP3).

    Returns ``(tokens [N_b, 5], query_xy [K_pts, 2], target [K_pts])``.

    Training mode
    -------------
    tokens   : (N_b, 5) float32 -- shuffled [x_c, y_c, w, l, h] in [0, 1]
    query_xy : (K_pts, 2) float32 -- sampled fluid cell centres [x/Lx, y/Ly]
    target   : (K_pts,) float32 -- U_mid / U_ref at those points

    Eval mode returns the same tuple but with ALL fluid points (used by a
    full-map prediction pass, not a DataLoader). Token order is shuffled and
    fluid points are subsampled only in ``'train'`` mode, giving set-invariance
    over buildings and stochastic query coverage.
    """

    def __init__(self, case_dirs, mode='train', K_pts=2000):
        assert mode in ('train', 'eval')
        self.case_dirs = case_dirs
        self.mode = mode
        self.K_pts = K_pts

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        tokens = np.load(case_dir / 'building_tokens.npy').astype(np.float32)  # [N_b, 5]
        U_mid  = np.load(case_dir / 'U_mid.npy').astype(np.float32)            # [Ny, Nx]
        fluid  = np.load(case_dir / 'fluid_mask_mid.npy').astype(np.float32)   # [Ny, Nx]
        Ny, Nx = U_mid.shape

        # Cell-centred coordinates (same convention as WP1).
        x_ch = (np.arange(Nx, dtype=np.float32) + 0.5) / Nx
        y_ch = (np.arange(Ny, dtype=np.float32) + 0.5) / Ny
        yy, xx = np.meshgrid(y_ch, x_ch, indexing='ij')                       # [Ny, Nx] each

        # Flatten and keep only fluid cells.
        fluid_flat = fluid.flatten() > 0
        xx_f = xx.flatten()[fluid_flat]                                       # [N_fluid]
        yy_f = yy.flatten()[fluid_flat]
        U_f  = U_mid.flatten()[fluid_flat]

        if self.mode == 'train':
            n_fluid = xx_f.shape[0]
            k = min(self.K_pts, n_fluid)
            sel = np.random.choice(n_fluid, size=k, replace=False)
            xx_f = xx_f[sel]
            yy_f = yy_f[sel]
            U_f = U_f[sel]
            # Shuffle token order: buildings are a set, not a sequence.
            perm = np.random.permutation(tokens.shape[0])
            tokens = tokens[perm]

        query_xy = np.stack([xx_f, yy_f], axis=-1)                            # [K_pts, 2] or [N_fluid, 2]
        return (
            torch.from_numpy(tokens),      # variable length: [N_b, 5]
            torch.from_numpy(query_xy),    # [K_pts, 2]
            torch.from_numpy(U_f),         # [K_pts]
        )


def collate_fn(batch):
    """Pad variable-length token sequences and build the key_padding_mask.

    Returns
    -------
    tokens_pad   : (B, max_nb, TOKEN_DIM) float32 -- zero-padded
    padding_mask : (B, max_nb) bool -- True = padding slot (PyTorch
                   ``src_key_padding_mask`` convention)
    query_xy     : (B, K_pts, 2) float32
    target       : (B, K_pts) float32
    """
    tokens_list, query_list, target_list = zip(*batch)
    B = len(tokens_list)
    max_nb = max(t.shape[0] for t in tokens_list)
    max_nb = max(max_nb, 1)                          # guard against degenerate edge case

    tokens_pad = torch.zeros(B, max_nb, TOKEN_DIM, dtype=torch.float32)
    padding_mask = torch.ones(B, max_nb, dtype=torch.bool)   # True = padding
    for i, t in enumerate(tokens_list):
        n = t.shape[0]
        tokens_pad[i, :n] = t
        padding_mask[i, :n] = False                  # real tokens

    query_xy = torch.stack(query_list, dim=0)        # (B, K_pts, 2)
    target = torch.stack(target_list, dim=0)         # (B, K_pts)
    return tokens_pad, padding_mask, query_xy, target


# ===========================================================================
# WP3 full-grid dataset for UrbanFormer-Field
# ===========================================================================
import torch.nn.functional as _F  # noqa: E402

from urbanformer.models.field import PATCH, QUERY_PATCH  # noqa: E402

# augmentation (train only). Spanwise reflection is exact for +x wind; periodic
# translation is enabled here matching the flagship run.
ENABLE_AUG    = True
AUG_REFLECT_Y = True
AUG_TRANSLATE = True


def compute_query_feats(tokens, height_map, xq, yq):
    """(K, 4): [h_local, d_nearest, h_nearest, d_upstream] per query point."""
    Ny, Nx = height_map.shape
    c = np.clip(np.round(xq * Nx - 0.5).astype(int), 0, Nx - 1)
    r = np.clip(np.round(yq * Ny - 0.5).astype(int), 0, Ny - 1)
    bx, by, bh = tokens[:, 0], tokens[:, 1], tokens[:, 4]
    ddx = xq[:, None] - bx[None, :]
    ddy = yq[:, None] - by[None, :]
    d2 = ddx ** 2 + ddy ** 2
    j = d2.argmin(1)
    d_near = np.sqrt(d2[np.arange(len(j)), j])
    h_near = bh[j]
    h_local = height_map[r, c]
    up = np.where(ddx > 0.0, ddx, np.inf)            # x_q - x_c > 0 -> upstream (wind +x)
    d_up = up.min(1)
    d_up[~np.isfinite(d_up)] = 1.0
    return np.stack([h_local, d_near, h_near, d_up], axis=1).astype(np.float32)


def extract_patches(height_map, P=PATCH):
    """(Ny*Nx, P*P) local height window per grid cell, zero-padded at borders."""
    hm = torch.from_numpy(height_map)[None, None]                       # (1, 1, Ny, Nx)
    patches = _F.unfold(hm, kernel_size=P, padding=P // 2)              # (1, P*P, Ny*Nx)
    return patches.squeeze(0).transpose(0, 1).contiguous().numpy().astype(np.float32)


def grid_query_xy(Ny, Nx):
    """Row-major (cols fastest) normalized cell-center coordinates -> (Ny*Nx, 2)."""
    gx, gy = np.meshgrid((np.arange(Nx) + 0.5) / Nx, (np.arange(Ny) + 0.5) / Ny)  # 'xy'
    return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)


class TokenFieldDataset(Dataset):
    """One case -> (tokens, query_xy, qfeats, patches, target_map, fluid). Full grid.

    UF-F decodes the whole grid per case so the axial query self-attention and the
    structural loss operate on a coherent field. Train mode applies spanwise
    reflection and periodic translation.
    """

    def __init__(self, case_dirs, train=False):
        self.case_dirs = case_dirs
        self.train = train

    def _augment(self, tokens, U, fluid, hm):
        Ny, Nx = U.shape
        if AUG_REFLECT_Y and np.random.rand() < 0.5:
            U, fluid, hm = U[::-1].copy(), fluid[::-1].copy(), hm[::-1].copy()
            tokens = tokens.copy(); tokens[:, 1] = 1.0 - tokens[:, 1]
        if AUG_TRANSLATE:
            sy, sx = np.random.randint(Ny), np.random.randint(Nx)
            U = np.roll(U, (sy, sx), axis=(0, 1))
            fluid = np.roll(fluid, (sy, sx), axis=(0, 1))
            hm = np.roll(hm, (sy, sx), axis=(0, 1))
            tokens = tokens.copy()
            tokens[:, 0] = (tokens[:, 0] + sx / Nx) % 1.0
            tokens[:, 1] = (tokens[:, 1] + sy / Ny) % 1.0
        return tokens, U, fluid, hm

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        cd = self.case_dirs[idx]
        tokens = np.load(cd / "building_tokens.npy").astype(np.float32)
        U = np.load(cd / "U_mid.npy").astype(np.float32)
        fluid = np.load(cd / "fluid_mask_mid.npy").astype(np.float32)
        hm = np.load(cd / "height_map.npy").astype(np.float32)
        if self.train and ENABLE_AUG:
            tokens, U, fluid, hm = self._augment(tokens, U, fluid, hm)

        Ny, Nx = U.shape
        qxy = grid_query_xy(Ny, Nx)
        qf = compute_query_feats(tokens, hm, qxy[:, 0], qxy[:, 1])
        pa = (extract_patches(hm) if QUERY_PATCH
              else np.zeros((Ny * Nx, 0), np.float32))
        return (torch.from_numpy(tokens), torch.from_numpy(qxy), torch.from_numpy(qf),
                torch.from_numpy(pa), torch.from_numpy(U), torch.from_numpy(fluid))


def collate_field(batch):
    """Pad token sets to batch max; stack same-shape fields. pad mask: True = pad."""
    tok_l, q_l, f_l, p_l, t_l, fl_l = zip(*batch)
    Bn = len(batch); N_max = max(t.shape[0] for t in tok_l)
    tokens = torch.zeros(Bn, N_max, TOKEN_DIM)
    pad = torch.ones(Bn, N_max, dtype=torch.bool)
    for i, t in enumerate(tok_l):
        n = t.shape[0]; tokens[i, :n] = t; pad[i, :n] = False
    return (tokens, pad, torch.stack(q_l), torch.stack(f_l),
            torch.stack(p_l), torch.stack(t_l), torch.stack(fl_l))
