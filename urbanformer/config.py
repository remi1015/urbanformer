"""Model registry and factory: one readable entry per model in the study.

Single source of truth for the mapping ``stage -> (model, loss, dataset)`` that
both :mod:`urbanformer.train` and :mod:`urbanformer.eval` dispatch on. Keeping it
here (rather than in the notebooks) is what lets the CLI reproduce a run with one
command while the notebooks stay thin drivers over the same package.

The study is built in numbered **stages** (the code and notebooks historically
call these "work packages", WP0-WP5). Stages 1-4 are the four models compared
here; each carries a human-readable ``name`` and one-line ``desc``:

===== =============================== ===========================================
stage name                            what it is
===== =============================== ===========================================
1     U-Net baseline                  raster CNN over a height map (image-to-image)
2     Pooled-token Transformer        set encoder -> one vector -> Fourier/FiLM decoder
3     UrbanFormer-Field               flagship: per-query cross-attention + axial grid
4     UrbanFormer-Field + morphology  stage 3 plus a global morphology token
===== =============================== ===========================================

``tag`` is the short, stable identifier written into a checkpoint's metadata and
checked by the provenance guard (:mod:`urbanformer.provenance`); it is an internal
label, not a display string. Use ``name``/``desc`` for anything a human reads.
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
    """Everything needed to build, train, and score one model (one study stage)."""

    wp: int                         # stage number (1-4 here; 0=data, 5=evaluation)
    tag: str                        # stable checkpoint/provenance label (internal)
    name: str                       # human-readable model name (for display)
    desc: str                       # one-line description of the model
    kind: str                       # "unet" | "pooled" | "field"
    morph_mode: str = "none"        # provenance lever; "token" for stage 4
    epochs: int = 60
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 5e-4
    extra: Dict[str, Any] = field(default_factory=dict)


# The canonical run of each stage. `variant` selects within a stage when it ships
# more than one model (stage 2 base vs FiLM).
WP_CONFIGS: Dict[int, WPConfig] = {
    1: WPConfig(wp=1, tag="WP1-unet", name="U-Net baseline",
                desc="raster CNN over a height map (image-to-image)",
                kind="unet", lr=1e-3, weight_decay=0.0,
                extra={"in_channels": 4}),
    2: WPConfig(wp=2, tag="WP2-pool", name="Pooled-token Transformer",
                desc="set encoder pooled to one vector + Fourier/FiLM decoder",
                kind="pooled", lr=1e-3,
                extra={"variant": "film", "k_points": 2000}),
    3: WPConfig(wp=3, tag="WP3-UFF", name="UrbanFormer-Field",
                desc="flagship: per-query cross-attention to buildings + axial grid",
                kind="field", morph_mode="none"),
    4: WPConfig(wp=4, tag="WP4-morph", name="UrbanFormer-Field + morphology",
                desc="stage 3 plus a global morphology token",
                kind="field", morph_mode="token",
                weight_decay=5e-4, extra={"dropout": 0.15}),
}


def get_config(wp: int) -> WPConfig:
    if wp not in WP_CONFIGS:
        raise ValueError(f"unknown stage {wp!r}; choose from {sorted(WP_CONFIGS)}")
    return WP_CONFIGS[wp]


def build_model(wp: int, variant: str | None = None):
    """Instantiate the canonical model for a study stage.

    Stage 4 (UrbanFormer-Field + morphology) flips the module-level ``MULTISCALE``
    lever in ``models.field`` on before construction so the global-morphology token
    is wired in; every other stage reads the shipped defaults. Returns an
    ``nn.Module`` on CPU in eval-agnostic state.
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
        _field.MULTISCALE = want_morph          # stage-4 bridge: global morphology token
        try:
            model = UrbanFormerField()
        finally:
            _field.MULTISCALE = prev            # never leak the lever across builds
        return model
    raise ValueError(f"unhandled model kind {cfg.kind!r}")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
