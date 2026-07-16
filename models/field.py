"""UrbanFormer-Field: the flagship geometry-conditioned field surrogate (WP3).

Encoder (permutation-invariant set Transformer over building tokens) -> memory.
Decoder = ``RESIDUAL_DEPTH`` residual blocks of {relative-geometry cross-attention
to buildings -> axial query self-attention over the grid -> FFN}. The
cross-attention restores conditional geometry (anisotropic streamwise/spanwise
scales plus a linear streamwise term for upstream/downstream asymmetry); the
axial self-attention restores spatial coherence.

The axial stage imports :class:`urbanformer.models.axial.AxialSelfAttention`,
the corrected column-permute implementation (architecture revision
``uff-axial-fix``). The column branch permutes ``(B, Ny, Nx, D) -> (B, Nx, Ny, D)``
before reshaping, so it attends down columns rather than re-attending rows; this
is the fix behind the flagship R2 = 0.8461. The module has 1,633,969 parameters
in the default configuration.

The module-level flags are the ablation levers; all default to the full UF-F
configuration. Toggle one at a time to reproduce the WP3 ablation matrix.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from urbanformer.models.axial import AxialSelfAttention

# --- token / model dims (carried from WP2/WP3 unchanged) ---
TOKEN_DIM     = 5          # [x_c, y_c, w, l, h]
D_MODEL       = 128
N_HEADS       = 4
N_ENC_LAYERS  = 3
ENC_DIM_FF    = 256
DEC_DIM_FF    = 256
NUM_FREQS     = 64         # Fourier features -> 2 * NUM_FREQS
FOURIER_SCALE = 10.0
DROPOUT       = 0.1

# --- UF-F ablation levers (default = full UF-F) ---
QUERY_SELFATTN  = True     # (A) axial self-attention over the query grid (core)
REL_COORD       = True     # (B) anisotropic, upstream/downstream-asymmetric relative bias
QUERY_PATCH     = True     # (C) local PATCH x PATCH height window per query
PATCH           = 9
QUERY_KNN       = True      # (C) scalar nearest-building feats [h_local, d_near, h_near, d_up]
NUM_QUERY_FEATS = 4
RESIDUAL_DEPTH  = 4        # (F) number of decoder blocks (coarse -> fine)
MULTISCALE      = False    # (E) prepend a global morphology token (WP4 vector plugs in here)
GEO_BIAS        = True     # isotropic distance bias, used only when REL_COORD=False
AXIAL_POS       = True     # learned row/col positional embeddings inside axial attn

PATCH_FEAT_DIM = (PATCH * PATCH) if QUERY_PATCH else 0
EXTRA_FEAT_DIM = (NUM_QUERY_FEATS if QUERY_KNN else 0)
QUERY_IN_DIM   = 2 * NUM_FREQS + EXTRA_FEAT_DIM + PATCH_FEAT_DIM


class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, num_freqs=NUM_FREQS, scale=FOURIER_SCALE):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, num_freqs) * scale)

    def forward(self, xy):
        proj = 2 * np.pi * xy @ self.B
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class TokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(TOKEN_DIM, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=ENC_DIM_FF,
            dropout=DROPOUT, activation="gelu", norm_first=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_ENC_LAYERS)
        if MULTISCALE:
            self.morph = nn.Sequential(nn.Linear(2 * TOKEN_DIM, D_MODEL), nn.GELU(),
                                       nn.Linear(D_MODEL, D_MODEL))

    def forward(self, tokens, padding_mask):
        h = self.embed(tokens)
        if MULTISCALE:
            # self-contained global morphology token from per-set stats (WP4 vector plugs in here)
            real = (~padding_mask).float().unsqueeze(-1)                # (B, N, 1)
            n = real.sum(1).clamp_min(1.0)
            mean = (tokens * real).sum(1) / n
            var = ((tokens - mean.unsqueeze(1)) ** 2 * real).sum(1) / n
            g = self.morph(torch.cat([mean, var.sqrt()], dim=-1)).unsqueeze(1)   # (B, 1, D)
            h = torch.cat([g, h], dim=1)
            padding_mask = torch.cat([torch.zeros_like(padding_mask[:, :1]), padding_mask], dim=1)
        return self.encoder(h, src_key_padding_mask=padding_mask), padding_mask


class RelCrossAttention(nn.Module):
    """Query -> building cross-attention with a relative-geometry bias on the logits.

    REL_COORD: anisotropic + streamwise-asymmetric bias
        bias_h = -softplus(g_s_h) * Ds^2 - softplus(g_n_h) * Dn^2 + beta_h * Ds
    (separate streamwise/spanwise scales give wind-aligned anisotropy; the linear
    Ds term breaks upstream/downstream symmetry). Falls back to GEO_BIAS
    isotropic ``-gamma * d^2``.
    """

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        if REL_COORD:
            self.g_s = nn.Parameter(torch.zeros(n_heads))
            self.g_n = nn.Parameter(torch.zeros(n_heads))
            self.beta = nn.Parameter(torch.zeros(n_heads))
        elif GEO_BIAS:
            self.log_gamma = nn.Parameter(torch.zeros(n_heads))

    def forward(self, q, memory, query_xy, build_xy, key_padding_mask, return_attn=False):
        B, Q, _ = q.shape
        N = memory.shape[1]
        Qh = self.q_proj(q).view(B, Q, self.h, self.dk).transpose(1, 2)
        Kh = self.k_proj(memory).view(B, N, self.h, self.dk).transpose(1, 2)
        Vh = self.v_proj(memory).view(B, N, self.h, self.dk).transpose(1, 2)
        logits = (Qh @ Kh.transpose(-2, -1)) / (self.dk ** 0.5)         # (B, h, Q, N)
        Nb = build_xy.shape[1]
        ds = query_xy[:, :, None, 0] - build_xy[:, None, :, 0]          # (B, Q, Nb) streamwise (+x)
        dn = query_xy[:, :, None, 1] - build_xy[:, None, :, 1]          # spanwise
        if REL_COORD:
            gs = F.softplus(self.g_s).view(1, self.h, 1, 1)
            gn = F.softplus(self.g_n).view(1, self.h, 1, 1)
            bt = self.beta.view(1, self.h, 1, 1)
            bias = -gs * (ds ** 2)[:, None] - gn * (dn ** 2)[:, None] + bt * ds[:, None]
        elif GEO_BIAS:
            gamma = F.softplus(self.log_gamma).view(1, self.h, 1, 1)
            bias = -gamma * (ds ** 2 + dn ** 2)[:, None]
        else:
            bias = None
        if bias is not None:
            if N == Nb + 1:                                            # global token prepended -> no geo bias
                bias = torch.cat([torch.zeros_like(bias[..., :1]), bias], dim=-1)
            logits = logits + bias
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = self.drop(logits.softmax(dim=-1))
        out = self.out_proj((attn @ Vh).transpose(1, 2).reshape(B, Q, self.h * self.dk))
        return (out, attn) if return_attn else (out, None)


class UFFieldBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_q = nn.LayerNorm(D_MODEL)
        self.cross = RelCrossAttention()
        if QUERY_SELFATTN:
            # corrected column-permute axial attention (architecture revision uff-axial-fix)
            self.axial = AxialSelfAttention(d_model=D_MODEL, n_heads=N_HEADS,
                                            dropout=DROPOUT, axial_pos=AXIAL_POS)
        self.norm_ff = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, DEC_DIM_FF), nn.GELU(),
                                nn.Dropout(DROPOUT), nn.Linear(DEC_DIM_FF, D_MODEL))

    def forward(self, q, memory, query_xy, build_xy, padding_mask, Ny, Nx, return_attn=False):
        attn_out, attn_w = self.cross(self.norm_q(q), memory, query_xy, build_xy,
                                      padding_mask, return_attn)
        q = q + attn_out
        if QUERY_SELFATTN:
            q = self.axial(q, Ny, Nx)
        q = q + self.ff(self.norm_ff(q))
        return q, attn_w


class UFFieldDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fourier = FourierFeatures()
        self.query_proj = nn.Linear(QUERY_IN_DIM, D_MODEL)
        self.layers = nn.ModuleList([UFFieldBlock() for _ in range(RESIDUAL_DEPTH)])
        self.head = nn.Sequential(nn.LayerNorm(D_MODEL),
                                  nn.Linear(D_MODEL, D_MODEL), nn.GELU(),
                                  nn.Linear(D_MODEL, 1))

    def forward(self, memory, padding_mask, query_xy, build_xy, qfeats, patches, Ny, Nx,
                return_attn=False):
        qe = self.fourier(query_xy)
        if QUERY_KNN:
            qe = torch.cat([qe, qfeats], dim=-1)
        if QUERY_PATCH:
            qe = torch.cat([qe, patches], dim=-1)
        q = self.query_proj(qe)
        attns = []
        for blk in self.layers:
            q, a = blk(q, memory, query_xy, build_xy, padding_mask, Ny, Nx, return_attn)
            attns.append(a)
        pred = self.head(q).squeeze(-1)
        return (pred, attns) if return_attn else pred


class UrbanFormerField(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TokenEncoder()
        self.decoder = UFFieldDecoder()

    def encode(self, tokens, padding_mask):
        return self.encoder(tokens, padding_mask)        # -> (memory, padding_mask')

    def decode(self, memory, padding_mask, query_xy, build_xy, qfeats, patches, Ny, Nx,
               return_attn=False):
        return self.decoder(memory, padding_mask, query_xy, build_xy, qfeats, patches, Ny, Nx,
                            return_attn)

    def forward(self, tokens, padding_mask, query_xy, qfeats, patches, Ny, Nx):
        memory, pad2 = self.encode(tokens, padding_mask)
        build_xy = tokens[..., :2]
        return self.decode(memory, pad2, query_xy, build_xy, qfeats, patches, Ny, Nx)
