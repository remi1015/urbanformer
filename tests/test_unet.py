"""Regression tests for :mod:`urbanformer.models.unet` (WP1).

Required invariant: the odd 78 -> 39 -> 19 -> 9 encoder pyramid round-trips, so a
(B, 4, 78, 78) input yields a (B, 1, 78, 78) prediction. These are structural
forward-pass checks; no training happens here.
"""

import torch

from urbanformer.models.unet import DoubleConv, UNetMid


# ---------------------------------------------------------------------------
# 78 -> 39 -> 19 -> 9 round-trip (required invariant)
# ---------------------------------------------------------------------------
def test_encoder_pyramid_is_78_39_19_9():
    net = UNetMid().eval()
    x = torch.zeros(2, 4, 78, 78)
    s1 = net.enc1(x)
    s2 = net.enc2(net.pool(s1))
    s3 = net.enc3(net.pool(s2))
    sizes = [x.shape[-1], net.pool(s1).shape[-1], net.pool(s2).shape[-1], net.pool(s3).shape[-1]]
    assert sizes == [78, 39, 19, 9]


def test_forward_preserves_spatial_shape():
    net = UNetMid().eval()
    for b in (1, 2, 5):
        out = net(torch.zeros(b, 4, 78, 78))
        assert tuple(out.shape) == (b, 1, 78, 78)


# ---------------------------------------------------------------------------
# structural sanity
# ---------------------------------------------------------------------------
def test_parameter_count_is_stable():
    """Pin the parameter count so an accidental channel-width edit is caught."""
    net = UNetMid()
    assert sum(p.numel() for p in net.parameters()) == 1_927_297


def test_double_conv_preserves_spatial_size_and_channels():
    dc = DoubleConv(4, 16).eval()
    out = dc(torch.zeros(1, 4, 78, 78))
    assert tuple(out.shape) == (1, 16, 78, 78)


def test_custom_channel_config():
    net = UNetMid(in_channels=3, out_channels=2).eval()
    out = net(torch.zeros(1, 3, 78, 78))
    assert tuple(out.shape) == (1, 2, 78, 78)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
