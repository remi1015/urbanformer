"""
Axial self-attention over the query grid (WP3, UrbanFormer-Field).

This is the load-bearing lever of UF-F. Query-to-building cross-attention gives
each query point the right buildings but no awareness of its neighbours, so the
predicted field collapses into a narrow band and wakes and street channels blur.
Axial self-attention restores spatial coherence at O(Nx + Ny) per query instead
of the O(Nx * Ny) a full grid attention would cost.

Architecture revision `uff-axial-fix`
-------------------------------------
The column branch MUST permute (B, Ny, Nx, D) -> (B, Nx, Ny, D) before reshaping
to (B * Nx, Ny, D). Reshaping straight through gathers ROWS into each "column"
sequence and then writes row i's attention output into column i via the
transposed view. The result is row attention applied twice, the second time
scattered transposed. There is no column attention at all.

`reshape` never raises on the buggy path, because B * Ny * Nx * D factors
identically either way, on any grid, square or not. No shape check catches it and
no provenance guard catches it, because the two variants differ in no config key
and no tensor shape. Only `tests/test_axial.py` catches it.

Every UF-F number produced before this fix, including the original headline
R2 = 0.8284, came from a model with streamwise coupling only. Weights do not
transfer across the fix.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _axial_pos(pos: nn.Parameter, n: int) -> torch.Tensor:
    """Slice or interpolate a learned positional table to length `n`."""
    if pos.shape[1] == n:
        return pos
    return torch.nn.functional.interpolate(
        pos.transpose(1, 2), size=n, mode="linear", align_corners=False
    ).transpose(1, 2)


def to_column_sequences(x: torch.Tensor, Ny: int, Nx: int) -> torch.Tensor:
    """(B, Ny * Nx, D) -> (B * Nx, Ny, D), one sequence per column.

    Factored out of `AxialSelfAttention` so it can be tested in isolation. The
    permute on the second line is the entire fix.
    """
    B, _, D = x.shape
    xc = x.view(B, Ny, Nx, D)
    xc = xc.permute(0, 2, 1, 3).contiguous()  # (B, Nx, Ny, D)  <-- the fix
    return xc.reshape(B * Nx, Ny, D)


def from_column_sequences(xc: torch.Tensor, B: int, Ny: int, Nx: int) -> torch.Tensor:
    """Inverse of `to_column_sequences`: (B * Nx, Ny, D) -> (B, Ny * Nx, D)."""
    D = xc.shape[-1]
    xc = xc.view(B, Nx, Ny, D)
    xc = xc.permute(0, 2, 1, 3).contiguous()  # (B, Ny, Nx, D)
    return xc.reshape(B, Ny * Nx, D)


def to_row_sequences(x: torch.Tensor, Ny: int, Nx: int) -> torch.Tensor:
    """(B, Ny * Nx, D) -> (B * Ny, Nx, D), one sequence per row. No permute needed."""
    B, _, D = x.shape
    return x.view(B, Ny, Nx, D).reshape(B * Ny, Nx, D)


def from_row_sequences(xr: torch.Tensor, B: int, Ny: int, Nx: int) -> torch.Tensor:
    D = xr.shape[-1]
    return xr.view(B, Ny, Nx, D).reshape(B, Ny * Nx, D)


class AxialSelfAttention(nn.Module):
    """Row attention then column attention over a (Ny, Nx) query grid.

    Input and output are flattened as (B, Ny * Nx, D) in row-major order, which
    is what the cross-attention stage produces.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        max_grid: int = 128,
        axial_pos: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.axial_pos = axial_pos

        self.attn_row = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_col = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_row = nn.LayerNorm(d_model)
        self.norm_col = nn.LayerNorm(d_model)

        if axial_pos:
            self.pos_row = nn.Parameter(torch.zeros(1, max_grid, d_model))
            self.pos_col = nn.Parameter(torch.zeros(1, max_grid, d_model))
            nn.init.trunc_normal_(self.pos_row, std=0.02)
            nn.init.trunc_normal_(self.pos_col, std=0.02)

    def forward(self, x: torch.Tensor, Ny: int, Nx: int) -> torch.Tensor:
        B = x.shape[0]
        assert x.shape[1] == Ny * Nx, f"expected {Ny * Nx} queries, got {x.shape[1]}"

        # --- row branch: attend along +x (streamwise) ---
        h = self.norm_row(x)
        xr = to_row_sequences(h, Ny, Nx)
        if self.axial_pos:
            xr = xr + _axial_pos(self.pos_row, Nx)
        xr, _ = self.attn_row(xr, xr, xr, need_weights=False)
        x = x + from_row_sequences(xr, B, Ny, Nx)

        # --- column branch: attend along +y (spanwise) ---
        h = self.norm_col(x)
        xc = to_column_sequences(h, Ny, Nx)
        if self.axial_pos:
            xc = xc + _axial_pos(self.pos_col, Ny)
        xc, _ = self.attn_col(xc, xc, xc, need_weights=False)
        x = x + from_column_sequences(xc, B, Ny, Nx)

        return x
