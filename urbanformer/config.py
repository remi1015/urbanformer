"""Per-work-package configuration registry and the model factory.

Single source of truth for the mapping ``WP -> (model, loss, dataset, tag)`` that
both :mod:`urbanformer.train` and :mod:`urbanformer.eval` dispatch on. Keeping it
here (rather than in the notebooks) is what lets the CLI reproduce a run with one
command while the notebooks stay thin drivers over the same package.

The four comparison tags match the WP5 provenance guard
(:mod:`urbanformer.provenance`): ``WP1-unet``, ``WP2-pool``, ``WP3-UFF``,
``WP4-morph``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import urbanformer.models.field as _field
from urbanformer.models.field import UrbanFormerField
from urbanformer.models.pooled import PooledTransformer, PooledTransformerFiLM
from urbanformer.models.unet import UNetMid


@dataclass
class WPConfig:
    """Everything needed to build, train, and score one work package."""

    wp: int
    tag: str
    kind: str                       # "unet" | "pooled" | "field"
    morph_mode: str = "none"        # provenance lever; "token" for WP4
    epochs: int = 60
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 5e-4
    extra: Dict[str, Any] = field(default_factory=dict)


# The canonical run of each work package. `variant` selects within a WP where a
# WP ships more than one model (WP2 base vs FiLM).
WP_CONFIGS: Dict[int, WPConfig] = {
    1: WPConfig(wp=1, tag="WP1-unet", kind="unet", lr=1e-3, weight_decay=0.0,
                extra={"in_channels": 4}),
    2: WPConfig(wp=2, tag="WP2-pool", kind="pooled", lr=1e-3,
                extra={"variant": "film", "k_points": 2000}),
    3: WPConfig(wp=3, tag="WP3-UFF", kind="field", morph_mode="none"),
    4: WPConfig(wp=4, tag="WP4-morph", kind="field", morph_mode="token",
                weight_decay=5e-4, extra={"dropout": 0.15}),
}


def get_config(wp: int) -> WPConfig:
    if wp not in WP_CONFIGS:
        raise ValueError(f"unknown work package {wp!r}; choose from {sorted(WP_CONFIGS)}")
    return WP_CONFIGS[wp]


def build_model(wp: int, variant: str | None = None):
    """Instantiate the canonical model for a work package.

    WP4 flips the module-level ``MULTISCALE`` lever in ``models.field`` on before
    construction so the global-morphology token is wired in; every other WP reads
    the shipped defaults. Returns an ``nn.Module`` on CPU in eval-agnostic state.
    """
    cfg = get_config(wp)
    if cfg.kind == "unet":
        return UNetMid(in_channels=cfg.extra.get("in_channels", 4))
    if cfg.kind == "pooled":
        v = variant or cfg.extra.get("variant", "film")
        return PooledTransformerFiLM() if v == "film" else PooledTransformer()
    if cfg.kind == "field":
        want_morph = cfg.morph_mode == "token"
        prev = _field.MULTISCALE
        _field.MULTISCALE = want_morph          # WP4 bridge: global morphology token
        try:
            model = UrbanFormerField()
        finally:
            _field.MULTISCALE = prev            # never leak the lever across builds
        return model
    raise ValueError(f"unhandled model kind {cfg.kind!r}")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
