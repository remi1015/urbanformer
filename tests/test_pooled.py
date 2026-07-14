"""Regression tests for :mod:`urbanformer.models.pooled` (WP2).

Required invariant: the two models have exactly 464,769 (base) and 695,169
(FiLM) parameters. These counts are the architectural fingerprint the WP2 report
quotes, so a change to any layer width is caught here. The suite also pins the
forward contract, that mean-pooling ignores padded tokens, and that FiLM starts
as identity modulation.
"""

import torch

from urbanformer.models.pooled import (
    FiLMBlock,
    FourierFeatures,
    PooledTransformer,
    PooledTransformerFiLM,
)


# ---------------------------------------------------------------------------
# parameter counts (required invariant)
# ---------------------------------------------------------------------------
def test_base_param_count():
    assert sum(p.numel() for p in PooledTransformer().parameters()) == 464_769


def test_film_param_count():
    assert sum(p.numel() for p in PooledTransformerFiLM().parameters()) == 695_169


# ---------------------------------------------------------------------------
# forward contract
# ---------------------------------------------------------------------------
def test_forward_shapes_base_and_film():
    tok = torch.zeros(2, 30, 5)
    pmask = torch.zeros(2, 30, dtype=torch.bool)
    qxy = torch.rand(2, 2000, 2)
    for M in (PooledTransformer, PooledTransformerFiLM):
        m = M().eval()
        assert tuple(m(tok, pmask, qxy).shape) == (2, 2000)
        assert tuple(m.encode(tok, pmask).shape) == (2, 128)


# ---------------------------------------------------------------------------
# masked mean-pool ignores padded tokens
# ---------------------------------------------------------------------------
def test_encode_ignores_padded_tokens():
    torch.manual_seed(0)
    m = PooledTransformer().eval()

    tokens = torch.randn(1, 5, 5)
    pmask = torch.zeros(1, 5, dtype=torch.bool)
    z_ref = m.encode(tokens, pmask)

    # append two padded tokens with arbitrary content; z must be unchanged
    extra = torch.randn(1, 2, 5) * 100.0
    tokens2 = torch.cat([tokens, extra], dim=1)
    pmask2 = torch.cat([pmask, torch.ones(1, 2, dtype=torch.bool)], dim=1)
    z_pad = m.encode(tokens2, pmask2)

    assert torch.allclose(z_ref, z_pad, atol=1e-5)


# ---------------------------------------------------------------------------
# FiLM starts at identity modulation
# ---------------------------------------------------------------------------
def test_film_block_is_zero_initialised():
    blk = FiLMBlock(16, 16, 8)
    assert torch.equal(blk.to_gamma_beta.weight, torch.zeros_like(blk.to_gamma_beta.weight))
    assert torch.equal(blk.to_gamma_beta.bias, torch.zeros_like(blk.to_gamma_beta.bias))
    # at init, gamma = beta = 0, so the block is exactly Linear -> GELU
    h = torch.randn(2, 7, 16)
    cond = torch.randn(2, 8)
    expected = blk.act(blk.linear(h))
    assert torch.allclose(blk(h, cond), expected)


def test_fourier_features_are_deterministic_and_shaped():
    f1 = FourierFeatures()
    f2 = FourierFeatures()
    assert torch.equal(f1.B, f2.B)          # seeded -> reproducible buffer
    xy = torch.rand(3, 2000, 2)
    out = f1(xy)
    assert tuple(out.shape) == (3, 2000, 2 * 64)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
