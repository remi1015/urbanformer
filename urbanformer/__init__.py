"""UrbanFormer — a geometry-conditioned surrogate for urban canopy flow.

What this package is, in one paragraph for a machine-learning reader
-------------------------------------------------------------------
UrbanFormer learns a fast **surrogate** (emulator) for an expensive fluid
solver. Given the *geometry* of a city block — a variable-length **set** of
building boxes plus a few global scalars — it predicts a 2-D **scalar field**:
the time-averaged wind speed at every point of a horizontal plane inside the
"urban canopy" (the layer of air between the buildings). Formally it learns a
map ``set -> function`` and evaluates that function at arbitrary ``(x, y)``
query coordinates, so it is a *conditional neural field* / *operator-learning*
model, not a fixed grid-to-grid image translator. The ground truth comes from
thousands of Lattice-Boltzmann (LBM) CFD simulations; once trained, the model
replaces the solver for any new layout in the same family — no per-layout
retraining, no new simulation.

The four models compared in the study (see :mod:`urbanformer.config`):

* Stage 1 — **U-Net baseline**: raster CNN over a height map (image-to-image).
* Stage 2 — **Pooled-token Transformer**: set encoder pooled to one vector + FiLM.
* Stage 3 — **UrbanFormer-Field** (the flagship): per-query cross-attention to
  building tokens plus axial self-attention over the query grid.
* Stage 4 — **UrbanFormer-Field + morphology**: stage 3 plus a global morphology
  token (a pre-registered null result).

Public API (import-friendly re-exports)
---------------------------------------
Models    : :class:`UNetMid`, :class:`PooledTransformer`,
            :class:`PooledTransformerFiLM`, :class:`UrbanFormerField`
Config    : :func:`get_config`, :func:`build_model`, :data:`WP_CONFIGS`
Losses    : :func:`masked_mse`, :func:`masked_field_loss`, :func:`make_radial_bins`
Metrics   : :func:`field_metrics`, :func:`per_case_rmse`, :func:`physics_metrics`
Guardrail : :func:`check_morph_provenance`, :class:`ProvenanceError`
Submodules: ``data``, ``morphology`` (kept as modules; import what you need)

Everything is CPU-friendly and testable on synthetic tensors; the dataset is not
required to import, build, or unit-test any of it. See the top-level ``README.md``
and ``docs/glossary.md`` for the fluid-mechanics vocabulary translated into ML
terms.
"""

from __future__ import annotations

from urbanformer import data, morphology
from urbanformer.config import WP_CONFIGS, build_model, count_params, get_config
from urbanformer.losses import make_radial_bins, masked_field_loss, masked_mse
from urbanformer.metrics import field_metrics, per_case_rmse, physics_metrics
from urbanformer.models.field import UrbanFormerField
from urbanformer.models.pooled import PooledTransformer, PooledTransformerFiLM
from urbanformer.models.unet import UNetMid
from urbanformer.provenance import ProvenanceError, check_morph_provenance

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # submodules worth surfacing
    "data",
    "morphology",
    # models
    "UNetMid",
    "PooledTransformer",
    "PooledTransformerFiLM",
    "UrbanFormerField",
    # config / model factory
    "WP_CONFIGS",
    "get_config",
    "build_model",
    "count_params",
    # losses
    "masked_mse",
    "masked_field_loss",
    "make_radial_bins",
    # metrics
    "field_metrics",
    "per_case_rmse",
    "physics_metrics",
    # provenance guard
    "check_morph_provenance",
    "ProvenanceError",
]
