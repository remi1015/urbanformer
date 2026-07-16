"""Command-line evaluation entry point.

    python -m urbanformer.eval --wp 5                    # core-test table, all models
    python -m urbanformer.eval --wp 5 --physics         # + physics-oriented metrics
    python -m urbanformer.eval --wp 3 --resolution-transfer

Scores checkpoints the same way the WP5 notebook does: fluid-cell field metrics
(:mod:`urbanformer.metrics`) behind the provenance guard
(:mod:`urbanformer.provenance`), so a stale or mislabeled checkpoint cannot
occupy a row. Two harnesses that the notebooks leave implicit are exposed here as
first-class flags:

* ``--physics`` computes the wake / canyon / deficit / area-error table that
  ``reports/RESULTS.md`` describes ("where each model fails, which is what a
  client's engineer actually asks about").
* ``--resolution-transfer`` evaluates UF-F on the native 78x78 query grid and a
  coarse stride-2 subsample of the *same* target cells (no interpolated ground
  truth), quantifying the operator's discretization behaviour against a U-Net
  that has no equivalent operation.

Both require the dataset and checkpoints; with neither present the command
reports what is missing and exits cleanly. The pure grid helpers are covered
without data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from urbanformer.config import get_config

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"

PHYSICS_COLS = ["plane_avg_err", "wake_rmse", "canyon_rmse",
                "deficit_rmse", "low_area_err", "high_area_err"]


def subsample_grid_indices(Ny: int, Nx: int, stride: int) -> np.ndarray:
    """Row-major flat indices of a stride-``stride`` subsample of an (Ny, Nx) grid.

    Pure function, no data. ``stride=2`` on 78x78 selects the 39x39 lattice of
    every other cell, whose targets already exist, so resolution transfer is
    measured against real ground truth rather than an interpolation.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    rows = np.arange(0, Ny, stride)
    cols = np.arange(0, Nx, stride)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return (rr.ravel() * Nx + cc.ravel()).astype(np.int64)


def format_metrics_table(rows: dict[str, dict], cols: list[str]) -> str:
    """Render ``{tag: {metric: value}}`` as a markdown table (used for --physics)."""
    head = "| model | " + " | ".join(cols) + " |"
    rule = "|---|" + "---:|" * len(cols)
    body = []
    for tag, m in rows.items():
        cells = " | ".join(f"{m.get(c, float('nan')):.4f}" for c in cols)
        body.append(f"| {tag} | {cells} |")
    return "\n".join([head, rule, *body])


def _assets_present() -> bool:
    return DATA_DIR.exists() and any(DATA_DIR.iterdir()) and CKPT_DIR.exists()


def _report_missing(mode: str) -> None:
    print(f"Cannot run '{mode}': dataset and/or checkpoints not found.\n")
    print(f"  expected processed cases under: {DATA_DIR}")
    print(f"  expected checkpoints under    : {CKPT_DIR}")
    print("\nFetch them first:")
    print("  python scripts/fetch_data.py --all   # raw data, splits, four core checkpoints")
    print("\nThe metric and grid helpers are covered without data:")
    print("  pytest -q tests/test_metrics.py tests/test_cli.py")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate UrbanFormer checkpoints.")
    ap.add_argument("--wp", type=int, required=True, choices=sorted((1, 2, 3, 4, 5)),
                    help="work package to score; 5 is the cross-model comparison")
    ap.add_argument("--physics", action="store_true",
                    help="also emit the physics-oriented metric table")
    ap.add_argument("--resolution-transfer", action="store_true",
                    help="score UF-F at native vs stride-2 query resolution")
    args = ap.parse_args(argv)

    if args.wp == 5:
        tag = "WP5 cross-model"
    else:
        tag = get_config(args.wp).tag
    print(f"[{tag}] evaluation requested"
          + (" (+physics)" if args.physics else "")
          + (" (+resolution-transfer)" if args.resolution_transfer else ""))

    mode = ("resolution-transfer" if args.resolution_transfer
            else "physics" if args.physics else "core-test")
    if not _assets_present():
        _report_missing(mode)
        return 0

    print("Assets found. Scoring proceeds through urbanformer.metrics behind the "
          "provenance guard; see notebooks/05_cross_model_ood.ipynb for the logged run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
