"""CLI + config wiring, covered on synthetic tensors (no dataset, no GPU).

These tests pin the parts of ``urbanformer.train`` / ``urbanformer.eval`` /
``urbanformer.config`` that do not need the LBM dataset:

* :func:`urbanformer.train.smoke_step` runs one forward+backward for every work
  package, so the model / loss / optimizer plumbing is self-verifying on CPU.
* :func:`urbanformer.config.build_model` returns the right architecture (checked
  by parameter-count fingerprint) and the morphology token genuinely adds
  parameters (stage 4 vs stage 3).
* the pure ``eval`` helpers (grid subsampling, markdown table formatting) behave.
"""
from __future__ import annotations

import math

import pytest

from urbanformer.config import build_model, count_params, get_config
from urbanformer.eval import format_metrics_table, subsample_grid_indices
from urbanformer.train import smoke_step


# --- training plumbing (one forward+backward per stage) --------------------
@pytest.mark.parametrize("wp", [1, 2, 3, 4])
def test_smoke_step_runs_for_each_wp(wp):
    loss = smoke_step(wp)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss >= 0.0            # every WP loss is an MSE-family (non-negative) scalar


# --- model factory ---------------------------------------------------------
def test_build_model_param_fingerprints():
    """The factory must build the documented architectures, not lookalikes."""
    assert count_params(build_model(1)) == 1_927_297     # stage 1, U-Net
    assert count_params(build_model(3)) == 1_633_969     # stage 3, UrbanFormer-Field


def test_morph_token_adds_parameters_and_restores_the_lever():
    """Stage 4 (global morphology token) has strictly more params than stage 3, and
    the module-level MULTISCALE lever is not leaked across builds."""
    import urbanformer.models.field as field

    assert field.MULTISCALE is False
    n3 = count_params(build_model(3))
    n4 = count_params(build_model(4))
    assert n4 > n3
    assert field.MULTISCALE is False   # config.build_model restores it in a finally


def test_config_tags_match_provenance_expectations():
    from urbanformer.provenance import EXPECTED_MORPH

    # `tag` is the stable internal provenance identifier; it must stay in sync
    # with the guard even though humans now see `name` instead.
    assert get_config(3).tag == "WP3-UFF"
    assert get_config(4).tag == "WP4-morph"
    assert get_config(3).morph_mode == EXPECTED_MORPH["WP3-UFF"]
    assert get_config(4).morph_mode == EXPECTED_MORPH["WP4-morph"]


def test_config_exposes_readable_names():
    assert get_config(1).name == "U-Net baseline"
    assert get_config(3).name == "UrbanFormer-Field"
    # every stage carries a non-empty one-line description
    for wp in (1, 2, 3, 4):
        cfg = get_config(wp)
        assert cfg.name and cfg.desc


def test_get_config_rejects_unknown_wp():
    with pytest.raises(ValueError):
        get_config(99)


# --- pure eval helpers (no data) -------------------------------------------
def test_subsample_grid_indices_stride1_is_identity():
    idx = subsample_grid_indices(5, 7, stride=1)
    assert idx.tolist() == list(range(5 * 7))


def test_subsample_grid_indices_stride2_picks_every_other_cell():
    # 4x4 grid, stride 2 -> rows {0,2} x cols {0,2} in row-major flat indices.
    idx = subsample_grid_indices(4, 4, stride=2)
    assert idx.tolist() == [0, 2, 8, 10]


def test_subsample_grid_indices_rejects_bad_stride():
    with pytest.raises(ValueError):
        subsample_grid_indices(8, 8, stride=0)


def test_format_metrics_table_is_valid_markdown():
    rows = {"UF-F": {"RMSE": 0.6192, "R2": 0.8461}}
    table = format_metrics_table(rows, ["RMSE", "R2"])
    lines = table.splitlines()
    assert lines[0] == "| model | RMSE | R2 |"
    assert set(lines[1]) <= set("|-:")               # the separator rule row
    assert lines[2] == "| UF-F | 0.6192 | 0.8461 |"  # 4-decimal formatting
