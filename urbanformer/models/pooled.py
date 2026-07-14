"""Pooled building-token Transformers (WP2, base vs Fourier/FiLM).

Two models that share a token encoder and differ only in the decoder, isolating
the decoder as the variable in the WP2 comparison:

* :class:`PooledTransformer` -- base. Building tokens -> Transformer encoder ->
  masked mean-pool to a single geometry vector ``z_geom`` -> plain MLP decoder on
  ``[z_geom, query_xy]``. 464,769 parameters.
* :class:`PooledTransformerFiLM` -- same encoder and mean-pool, but the decoder
  lifts the query coordinate through fixed random Fourier features and modulates
  two FiLM blocks with ``z_geom`` (Tancik et al. 2020; Perez et al. 2018).
  695,169 parameters.

The single pooled ``z_geom`` is the deliberate bottleneck: one latent vector
cannot carry per-location geometry, which is what motivates the per-query
cross-attention of WP3.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# --- shared encoder hyperparameters ---------------------------------------
TOKEN_DIM       = 5      # [x_center, y_center, l_x, l_y, h]
D_MODEL         = 128
N_HEAD          = 4
NUM_LAYERS      = 3
DIM_FEEDFORWARD = 256
DROPOUT         = 0.1

# --- decoder hyperparameters ----------------------------------------------
DEC_HIDDEN      = 256    # MLP / FiLM hidden width (both decoders)
NUM_FOURIER     = 64     # number of random Fourier frequencies (FiLM model)
FOURIER_SCALE   = 10.0   # std of the Gaussian frequency matrix B
FOURIER_SEED    = 0      # fixes B so the feature map is reproducible


class PooledTransformer(nn.Module):
    """WP2 base: pooled building-token Transformer + plain MLP coordinate decoder."""

    def __init__(self, token_dim=TOKEN_DIM, d_model=D_MODEL, nhead=N_HEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 dropout=DROPOUT, dec_hidden=DEC_HIDDEN):
        super().__init__()
        # --- token embedding ---
        self.token_embed = nn.Linear(token_dim, d_model)
        # --- transformer encoder (Pre-LN) ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,  # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)
        # --- MLP decoder: (d_model + 2) -> dec_hidden -> dec_hidden//2 -> 1 ---
        self.decoder = nn.Sequential(
            nn.Linear(d_model + 2, dec_hidden),
            nn.GELU(),
            nn.Linear(dec_hidden, dec_hidden // 2),
            nn.GELU(),
            nn.Linear(dec_hidden // 2, 1),
        )

    def encode(self, tokens, padding_mask):
        """tokens (B, N, 5), padding_mask (B, N) True=pad -> z_geom (B, d_model)."""
        x = self.token_embed(tokens)                                   # (B, N, d_model)
        x = self.transformer(x, src_key_padding_mask=padding_mask)     # (B, N, d_model)
        real = (~padding_mask).float().unsqueeze(-1)                   # (B, N, 1)
        z = (x * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)       # (B, d_model)
        return z

    def decode(self, z, query_xy):
        """z (B, d_model), query_xy (B, K, 2) -> pred (B, K)."""
        K = query_xy.shape[1]
        z_exp = z.unsqueeze(1).expand(-1, K, -1)                       # (B, K, d_model)
        dec_in = torch.cat([z_exp, query_xy], dim=-1)                  # (B, K, d_model+2)
        return self.decoder(dec_in).squeeze(-1)                        # (B, K)

    def forward(self, tokens, padding_mask, query_xy):
        return self.decode(self.encode(tokens, padding_mask), query_xy)


class FourierFeatures(nn.Module):
    """Fixed Gaussian random Fourier features for 2D coordinates.

    ``gamma(v) = [sin(2*pi * B v), cos(2*pi * B v)]`` with ``B ~ N(0, scale^2)``
    fixed at construction. Output dimension is ``2 * num_freqs``. ``B`` is a
    registered buffer, so it is saved in the checkpoint and moves with ``.to()``.
    """

    def __init__(self, num_freqs=NUM_FOURIER, scale=FOURIER_SCALE, seed=FOURIER_SEED):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(num_freqs, 2, generator=g) * scale            # (num_freqs, 2)
        self.register_buffer("B", B)
        self.out_dim = 2 * num_freqs

    def forward(self, xy):
        # xy: (..., 2) -> (..., 2*num_freqs)
        proj = 2.0 * np.pi * xy @ self.B.t()                          # (..., num_freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FiLMBlock(nn.Module):
    """Linear -> FiLM(gamma, beta from cond) -> GELU.

    ``h <- (1 + gamma) * h + beta``, with the ``(gamma, beta)`` generator
    zero-initialised so the block is the identity modulation at the start of
    training.
    """

    def __init__(self, in_dim, out_dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * out_dim)
        self.act = nn.GELU()
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, h, cond):
        # h (B, K, in_dim); cond (B, cond_dim) -> (B, K, out_dim)
        h = self.linear(h)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)       # each (B, out_dim)
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)        # FiLM modulation
        return self.act(h)


class PooledTransformerFiLM(nn.Module):
    """WP2-FiLM: same token encoder + mean-pool as WP2-base, but the decoder uses
    Fourier-feature query embedding and FiLM conditioning on ``z_geom``."""

    def __init__(self, token_dim=TOKEN_DIM, d_model=D_MODEL, nhead=N_HEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 dropout=DROPOUT, dec_hidden=DEC_HIDDEN,
                 num_freqs=NUM_FOURIER, fourier_scale=FOURIER_SCALE):
        super().__init__()
        # --- token embedding + encoder (identical to WP2-base) ---
        self.token_embed = nn.Linear(token_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)
        # --- Fourier query embedding + FiLM-modulated decoder ---
        self.fourier = FourierFeatures(num_freqs=num_freqs, scale=fourier_scale)
        self.input_proj = nn.Linear(self.fourier.out_dim, dec_hidden)
        self.film1 = FiLMBlock(dec_hidden, dec_hidden, d_model)
        self.film2 = FiLMBlock(dec_hidden, dec_hidden, d_model)
        self.head = nn.Linear(dec_hidden, 1)

    def encode(self, tokens, padding_mask):
        """Identical to WP2-base."""
        x = self.token_embed(tokens)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        real = (~padding_mask).float().unsqueeze(-1)
        z = (x * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        return z

    def decode(self, z, query_xy):
        """z (B, d_model), query_xy (B, K, 2) -> pred (B, K)."""
        h = self.input_proj(self.fourier(query_xy))                   # (B, K, dec_hidden)
        h = self.film1(h, z)
        h = self.film2(h, z)
        return self.head(h).squeeze(-1)                               # (B, K)

    def forward(self, tokens, padding_mask, query_xy):
        return self.decode(self.encode(tokens, padding_mask), query_xy)
