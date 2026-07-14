"""Regression tests for :mod:`urbanformer.provenance` (WP5).

Required invariant: a MORPH_MODE="none" checkpoint tagged "WP4-morph" is
rejected. The morphology conclusion depends on the treatment checkpoint really
being the treatment, so a mislabeled control must not load silently.
"""

import pytest
import torch

from urbanformer.provenance import (
    ProvenanceError,
    check_morph_provenance,
    extract_state_dict,
)


def test_rejects_none_checkpoint_tagged_wp4_morph():
    with pytest.raises(ProvenanceError):
        check_morph_provenance("WP4-morph", {"MORPH_MODE": "none"})


def test_accepts_correctly_tagged_checkpoints():
    check_morph_provenance("WP4-morph", {"MORPH_MODE": "token"})   # no raise
    check_morph_provenance("WP3-UFF", {"MORPH_MODE": "none"})      # no raise


def test_missing_morph_mode_is_inferred_from_tag():
    check_morph_provenance("WP4-morph", {})   # lenient: inferred as "token"
    check_morph_provenance("WP3-UFF", {})


def test_non_uff_tag_passes_through():
    check_morph_provenance("U-Net", {"MORPH_MODE": "whatever"})    # not a UF-F tag


def test_extract_state_dict_unwraps_model_key():
    sd = {"w": torch.zeros(3)}
    got_sd, cfg = extract_state_dict({"model": sd, "config": {"MORPH_MODE": "token"}})
    assert got_sd is sd and cfg == {"MORPH_MODE": "token"}
    raw_sd, raw_cfg = extract_state_dict({"w": torch.zeros(2)})
    assert "w" in raw_sd and raw_cfg is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
