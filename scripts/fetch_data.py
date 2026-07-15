"""
Fetch the raw dataset. Nothing under `data/` is tracked in git.

    python scripts/fetch_data.py --raw       # download urban_flow_dataset_5225.npz
    python scripts/fetch_data.py --processed # then run notebooks/00_build_dataset.ipynb

Requires a Kaggle API token at ~/.kaggle/kaggle.json (chmod 600).
Get one from https://www.kaggle.com/settings/account -> Create New Token.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RAW_DATASET = "rmialas/urban-flow-transformer"
SPLITS_DATASET = "rmialas/urban-flow-transformer-wp5-splits"

CKPT_DATASETS = {
    "unet": "rmialas/wp5-unet-core-retrain",
    "wp2_pool": "rmialas/wp2-pooled-core-retrain",
    "wp3_uff": "rmialas/wp3-uff-core-retrain",
    "wp4_morph": "rmialas/wp4-morph-core-retrain",
}

ROOT = Path(__file__).resolve().parent.parent


def _kaggle(dataset: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"-> {dataset}  ->  {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", action="store_true", help="raw .npz (5,225 LBM cases)")
    ap.add_argument("--splits", action="store_true", help="WP5 core split case lists")
    ap.add_argument("--checkpoints", action="store_true", help="the four core retrains")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if not any([args.raw, args.splits, args.checkpoints, args.all]):
        ap.print_help()
        return 1

    if subprocess.run(["which", "kaggle"], capture_output=True).returncode != 0:
        print("kaggle CLI not found:  pip install kaggle", file=sys.stderr)
        return 1

    if args.raw or args.all:
        _kaggle(RAW_DATASET, ROOT / "data" / "raw")
    if args.splits or args.all:
        _kaggle(SPLITS_DATASET, ROOT / "splits")
    if args.checkpoints or args.all:
        for tag, ds in CKPT_DATASETS.items():
            _kaggle(ds, ROOT / "checkpoints" / tag)

    print("\nDone. Next: notebooks/00_build_dataset.ipynb writes data/processed/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
