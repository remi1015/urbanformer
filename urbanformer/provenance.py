"""Checkpoint provenance and strict-loading guards (WP5).

The WP5 comparison scores four checkpoints side by side. A mislabeled
checkpoint (for example a ``MORPH_MODE="none"`` control saved under the
``WP4-morph`` treatment tag) would silently corrupt the morphology conclusion,
so every checkpoint is verified before it is trusted:

* :func:`check_morph_provenance` -- a UF-F checkpoint's declared ``MORPH_MODE``
  must match its tag (``WP3-UFF`` -> ``"none"``, ``WP4-morph`` -> ``"token"``).
* :func:`extract_state_dict` -- unwrap raw / ``{"model": ...}`` checkpoints.
* :func:`strict_load` / :func:`positional_remap` -- key-exact load, with a
  shape-checked positional fallback for the reconstructed WP2 class.
"""

from __future__ import annotations

import glob
import os

import torch

# The morphology lever each UF-F tag is required to carry.
EXPECTED_MORPH = {"WP3-UFF": "none", "WP4-morph": "token"}


class ProvenanceError(RuntimeError):
    """Raised when a checkpoint's contents contradict its claimed identity."""


def check_morph_provenance(tag, cfg):
    """Verify a UF-F checkpoint's ``MORPH_MODE`` matches its tag.

    A missing ``MORPH_MODE`` is inferred from the tag (lenient), but an explicit
    value that contradicts the tag is rejected: e.g. a ``"none"`` control saved
    under ``WP4-morph`` raises :class:`ProvenanceError`. Non-UF-F tags pass
    through untouched.
    """
    if tag not in EXPECTED_MORPH:
        return
    expected = EXPECTED_MORPH[tag]
    got = cfg.get("MORPH_MODE", expected)
    if got != expected:
        raise ProvenanceError(
            f"[{tag}] checkpoint MORPH_MODE={got!r} but tag requires {expected!r} "
            f"-- mislabeled checkpoint, refusing to score it as {tag}."
        )


def find_checkpoint(directory):
    """First ``*.pt`` / ``*.pth`` found under ``directory`` (recursive)."""
    pts = (sorted(glob.glob(os.path.join(directory, "**", "*.pt"), recursive=True))
           + sorted(glob.glob(os.path.join(directory, "**", "*.pth"), recursive=True)))
    if not pts:
        raise FileNotFoundError(f"no .pt/.pth checkpoint under {directory}")
    return pts[0]


def extract_state_dict(obj):
    """Return ``(state_dict, config_or_None)`` from a loaded checkpoint object."""
    if isinstance(obj, dict):
        for k in ("model", "state_dict", "model_state_dict", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k], obj.get("config")
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj, None
    raise ProvenanceError("unrecognized checkpoint format")


def strict_load(model, sd, tag):
    """Load with an exact key match. Returns True on a clean load, else False."""
    res = model.load_state_dict(sd, strict=False)
    miss, unexp = list(res.missing_keys), list(res.unexpected_keys)
    if not miss and not unexp:
        return True
    return False


def positional_remap(model, sd, tag):
    """Fallback for a reconstructed class: assign checkpoint tensors to model
    parameters in registration order, asserting identical shapes pairwise."""
    msd = model.state_dict()
    mk, ck = list(msd.keys()), list(sd.keys())
    if len(mk) != len(ck):
        raise ProvenanceError(
            f"[{tag}] param count differs: model {len(mk)} vs ckpt {len(ck)}."
        )
    new = {}
    for a, b in zip(mk, ck):
        if msd[a].shape != sd[b].shape:
            raise ProvenanceError(
                f"[{tag}] shape mismatch {a}{tuple(msd[a].shape)} <- {b}{tuple(sd[b].shape)}."
            )
        new[a] = sd[b]
    model.load_state_dict(new, strict=True)
