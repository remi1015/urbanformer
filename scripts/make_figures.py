"""
Regenerate the summary result figures from the numbers in reports/RESULTS.md.

These two figures are data-free: they render the WP5 core-test and OOD tables and
are committed so the README is not promising images that are not in the tree.

    python scripts/make_figures.py

The qualitative field galleries (predicted vs truth per model) are produced inside
notebooks/05_cross_model_ood.ipynb, which needs the dataset and the four core-split
checkpoints (see scripts/fetch_data.py). This script deliberately does not fabricate
fields it cannot load.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

# --- WP5 core_test, exactly as reported in reports/RESULTS.md ---
MODELS = ["WP2-pool", "U-Net", "WP4-morph", "WP3-UFF"]
R2 = {"WP2-pool": 0.2921, "U-Net": 0.7129, "WP4-morph": 0.8358, "WP3-UFF": 0.8461}
COLOR = {"WP2-pool": "#c0564a", "U-Net": "#5f8a63",
         "WP4-morph": "#7a5aa6", "WP3-UFF": "#4a7fb5"}

# --- WP5 OOD, delta-R2 vs core_test (negative = degradation) ---
OOD_REGIMES = ["h_rms↑", "λf↑", "γ↑", "γ↓", "λp↑", "λp↓", "skew↑", "kurt↑"]
OOD_MODELS = ["WP3-UFF", "WP4-morph", "U-Net", "WP2-pool"]
OOD = np.array([
    [-0.0124, -0.0069, -0.0208, -0.0320, -0.0351, -0.0223, -0.0073, -0.0014],  # WP3-UFF
    [-0.0140, -0.0056, -0.0324, -0.0366, -0.0314, -0.0210, -0.0088, +0.0021],  # WP4-morph
    [-0.0216, +0.0193, -0.0128, -0.0561, -0.0529, +0.0018, -0.0153, +0.0115],  # U-Net
    [-0.0289, +0.1997, -0.1869, +0.1082, +0.0900, -0.1095, -0.0389, +0.0402],  # WP2-pool
])


def bar_r2() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=170)
    xs = np.arange(len(MODELS))
    vals = [R2[m] for m in MODELS]
    ax.bar(xs, vals, color=[COLOR[m] for m in MODELS], width=0.62, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10.5, weight="bold", color="#1e2327")
    ax.set_xticks(xs)
    ax.set_xticklabels(MODELS, fontsize=10.5)
    ax.set_ylabel("R²  (core_test, fluid cells)", fontsize=10.5)
    ax.set_ylim(0, 0.95)
    ax.set_title("In-distribution accuracy on the identical core split (541 unseen layouts)",
                 fontsize=10.8, weight="bold")
    ax.grid(axis="y", color="#e3e6ea", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "core_test_r2.png", bbox_inches="tight", facecolor="white")
    print("wrote", OUT / "core_test_r2.png")


def heatmap_ood() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.7), dpi=170)
    vmax = np.abs(OOD).max()
    im = ax.imshow(OOD, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(OOD_REGIMES)))
    ax.set_xticklabels(OOD_REGIMES, fontsize=10)
    ax.set_yticks(range(len(OOD_MODELS)))
    ax.set_yticklabels(OOD_MODELS, fontsize=10.5)
    for i in range(OOD.shape[0]):
        for j in range(OOD.shape[1]):
            v = OOD[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8.2,
                    color="#1e2327" if abs(v) < 0.10 else "white")
    ax.set_title("OOD ΔR² vs core_test  (negative = degradation on the morphology tail)",
                 fontsize=10.6, weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    fig.text(0.012, 0.02,
             "WP2-pool's positive deltas are a collapse artifact, not robustness "
             "(it sits near the conditional mean). See reports/RESULTS.md.",
             fontsize=7.6, color="#5b6570")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "ood_delta_r2_heatmap.png", bbox_inches="tight", facecolor="white")
    print("wrote", OUT / "ood_delta_r2_heatmap.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bar_r2()
    heatmap_ood()


if __name__ == "__main__":
    main()
