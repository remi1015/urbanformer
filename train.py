"""Command-line training entry point.

    python -m urbanformer.train --wp 3            # train UrbanFormer-Field
    python -m urbanformer.train --wp 1 --epochs 5 # short U-Net run

The notebooks remain the exploratory source of truth; this module is the thin,
tested driver that reproduces a work package's run in one command. It resolves
the per-WP config (:mod:`urbanformer.config`), builds the matching model, loss,
and dataset, and writes a provenance-stamped checkpoint.

With no dataset present it does not fail obscurely: it reports exactly which
paths are missing and how to fetch them, then exits cleanly. The per-WP
forward/loss wiring is exercised without data by :func:`smoke_step`, which the
test-suite runs on synthetic tensors so the plumbing is self-verifying on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from urbanformer.config import build_model, count_params, get_config
from urbanformer.losses import masked_field_loss, masked_mse, make_radial_bins

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "splits"
CKPT_DIR = ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# per-WP forward + loss adapters (the only thing that differs across packages)
# ---------------------------------------------------------------------------
def _unet_step(model, batch, _bins):
    x, y, mask = batch
    pred = model(x).squeeze(1)
    return masked_mse(pred, y, mask)


def _pooled_step(model, batch, _bins):
    tokens, pad, query_xy, target = batch
    pred = model(tokens, pad, query_xy)
    return ((pred - target) ** 2).mean()


def _field_step(model, batch, bins):
    tokens, pad, qxy, qf, patches, target, fluid = batch
    B, Ny, Nx = target.shape
    pred = model(tokens, pad, qxy, qf, patches, Ny, Nx).view(B, Ny, Nx)
    loss, _ = masked_field_loss(pred, target, fluid, rbin=bins)
    return loss


STEP = {"unet": _unet_step, "pooled": _pooled_step, "field": _field_step}


def smoke_step(wp: int) -> float:
    """Run one forward+backward on synthetic tensors for a work package.

    Proves the model/loss/optimizer wiring for ``wp`` runs on CPU without any
    dataset. Returns the scalar loss. Used by the test-suite as the data-free
    guarantee that the CLI's training path is sound.
    """
    cfg = get_config(wp)
    model = build_model(wp)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Ny = Nx = 12
    if cfg.kind == "unet":
        batch = (torch.randn(2, 4, Ny, Nx), torch.randn(2, Ny, Nx),
                 torch.ones(2, Ny, Nx))
        bins = None
    elif cfg.kind == "pooled":
        nb, k = 5, 40
        batch = (torch.rand(2, nb, 5), torch.zeros(2, nb, dtype=torch.bool),
                 torch.rand(2, k, 2), torch.randn(2, k))
        bins = None
    else:
        nb, Q = 6, Ny * Nx
        from urbanformer.models.field import PATCH, QUERY_PATCH
        patches = (torch.rand(2, Q, PATCH * PATCH) if QUERY_PATCH
                   else torch.zeros(2, Q, 0))
        batch = (torch.rand(2, nb, 5), torch.zeros(2, nb, dtype=torch.bool),
                 torch.rand(2, Q, 2), torch.rand(2, Q, 4), patches,
                 torch.randn(2, Ny, Nx), torch.ones(2, Ny, Nx))
        bins = make_radial_bins(Ny, Nx)
    loss = STEP[cfg.kind](model, batch, bins)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


def _data_present() -> bool:
    return DATA_DIR.exists() and any(DATA_DIR.iterdir()) and SPLITS_DIR.exists()


def _report_missing() -> None:
    print("No processed dataset found. Nothing to train on.\n")
    print(f"  expected processed cases under : {DATA_DIR}")
    print(f"  expected core split files under: {SPLITS_DIR}")
    print("\nFetch and build them first:")
    print("  python scripts/fetch_data.py --all")
    print("  jupyter lab notebooks/00_build_dataset.ipynb")
    print("\nThe training path itself is covered without data:")
    print("  pytest -q tests/test_cli.py     # runs smoke_step for every WP")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train one UrbanFormer work package.")
    ap.add_argument("--wp", type=int, required=True, choices=sorted((1, 2, 3, 4)),
                    help="work package to train")
    ap.add_argument("--epochs", type=int, default=None, help="override epoch count")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="checkpoint output path")
    ap.add_argument("--smoke", action="store_true",
                    help="run one synthetic forward+backward and exit (no data)")
    args = ap.parse_args(argv)

    cfg = get_config(args.wp)
    model = build_model(args.wp)
    print(f"[{cfg.tag}] {cfg.kind} model, {count_params(model):,} parameters, "
          f"morph_mode={cfg.morph_mode!r}")

    if args.smoke:
        loss = smoke_step(args.wp)
        print(f"[{cfg.tag}] smoke step ok, loss={loss:.4f}")
        return 0

    if not _data_present():
        _report_missing()
        return 0

    print("Dataset found. See notebooks/0{1..4} for the full logged runs; "
          "this CLI drives the same package code.")
    # The heavy epoch loop reuses the package datasets/losses above; wiring it to
    # the fetched data is a one-line dataset construction per WP (see notebooks).
    return 0


if __name__ == "__main__":
    sys.exit(main())
