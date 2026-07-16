"""U-Net baseline for mid-canopy velocity (WP1, UrbanFormer-Mid).

A 3-level U-Net mapping the 4-channel raster input
(``height_map``, ``footprint_mask``, ``x_grid``, ``y_grid``) to the single-channel
mid-plane streamwise velocity ``U_mid`` on a fixed 78x78 grid. This is the
rasterized CNN baseline the object-based models are measured against.

The decoder resamples each up-conv back onto its skip connection with
``F.interpolate``, so the odd 78 -> 39 -> 19 -> 9 encoder pyramid round-trips
cleanly instead of drifting by a pixel at each level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2, spatial size preserved (padding=1)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetMid(nn.Module):
    """3-level U-Net. F.interpolate after each up-conv realigns spatial dims,
    which lets the odd 78 -> 39 -> 19 -> 9 pyramid round-trip cleanly."""

    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        # encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        # bottleneck + downsample
        self.bottleneck = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        # decoder (dec*-in = up-channels + skip-channels)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)   # 128 (up) + 128 (skip)
        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)    # 64 + 64
        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)     # 32 + 32
        # output
        self.output_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def _up(self, x, skip, up, dec):
        """Up-conv -> resize to skip -> concat skip -> double-conv."""
        x = up(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return dec(x)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x  = self.bottleneck(self.pool(s3))
        x  = self._up(x, s3, self.up3, self.dec3)
        x  = self._up(x, s2, self.up2, self.dec2)
        x  = self._up(x, s1, self.up1, self.dec1)
        return self.output_conv(x)
