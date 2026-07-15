"""
Render the UrbanFormer-Field (UF-F) architecture diagram.

Data-free: this figure documents the WP3/WP4 forward pass and can be regenerated
without the dataset or checkpoints. Writes docs/figures/architecture.png (and .svg).

    python scripts/make_arch_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

INK = "#1e2327"
MUTED = "#5b6570"
ENC = "#dbe7f3"
ENC_E = "#4a7fb5"
DEC = "#e7dff1"
DEC_E = "#7a5aa6"
CORE = "#f6e0dc"
CORE_E = "#c0564a"
IO = "#e4ece2"
IO_E = "#5f8a63"


def box(ax, x, y, w, h, text, fc, ec, fs=10.5, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, weight=weight, zorder=3, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, zorder=1,
        shrinkA=2, shrinkB=2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.6, 6.6), dpi=170)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    ax.text(0.15, 6.32, "UrbanFormer-Field  (WP3 / WP4)", fontsize=15,
            weight="bold", color=INK)
    ax.text(0.15, 6.02,
            r"$G_\theta(B,\, m,\, x, y)\ \rightarrow\ \bar{u}(x, y,\ z{=}h_m/2)\,/\,U_{ref}$",
            fontsize=11.5, color=MUTED)

    # ---- inputs ----
    box(ax, 0.15, 4.30, 2.05, 1.15,
        "Building set  B\n{ [x_c, y_c, w, l, h] }\nvariable length, 16–44",
        IO, IO_E, fs=9.3, weight="bold")
    box(ax, 0.15, 1.05, 2.05, 1.15,
        "Query points\n(x, y) on the\nmid-canopy plane",
        IO, IO_E, fs=9.3, weight="bold")

    # ---- encoder ----
    box(ax, 2.75, 4.15, 2.35, 1.45,
        "Token Encoder\n\nLinear embed  →\n3× Transformer\nencoder layers\n(set, permutation-\ninvariant)",
        ENC, ENC_E, fs=9.2, weight="bold")
    ax.text(3.92, 3.92, "memory", fontsize=8.6, color=ENC_E, ha="center", style="italic")

    # ---- query featurization ----
    box(ax, 2.75, 0.85, 2.35, 1.55,
        "Query features\n\nFourier(x,y)\n+ kNN scalars\n+ local height patch\n→ linear  →  q",
        IO, IO_E, fs=9.2, weight="bold")

    # ---- decoder stack ----
    dx, dy, dw, dh = 5.75, 1.15, 3.15, 4.35
    ax.add_patch(FancyBboxPatch(
        (dx - 0.12, dy - 0.12), dw + 0.24, dh + 0.24,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5, edgecolor=DEC_E, facecolor="#f4f0fa", zorder=1))
    ax.text(dx + dw / 2, dy + dh + 0.02, "Decoder  ·  4 residual blocks",
            ha="center", va="bottom", fontsize=10.2, weight="bold", color=DEC_E)

    box(ax, dx + 0.18, dy + 3.05, dw - 0.36, 0.95,
        "Relative-geometry\ncross-attention  (q → buildings)\nanisotropic streamwise/spanwise +\nupstream/downstream asymmetry",
        CORE, CORE_E, fs=8.5, weight="bold")
    box(ax, dx + 0.18, dy + 1.75, dw - 0.36, 0.95,
        "Axial self-attention\nover the (Ny, Nx) query grid\nrow then column,  O(Nx + Ny)\n(load-bearing lever)",
        CORE, CORE_E, fs=8.5, weight="bold")
    box(ax, dx + 0.18, dy + 0.70, dw - 0.36, 0.72,
        "Feed-forward  +  residual", DEC, DEC_E, fs=9.2)
    ax.text(dx + dw / 2, dy + 0.28, "×4  (coarse → fine)", ha="center",
            fontsize=8.8, color=DEC_E, style="italic")

    # internal vertical arrows in decoder
    arrow(ax, dx + dw / 2, dy + 3.05, dx + dw / 2, dy + 2.70, color=DEC_E)
    arrow(ax, dx + dw / 2, dy + 1.75, dx + dw / 2, dy + 1.42, color=DEC_E)

    # ---- head / output ----
    box(ax, 9.45, 2.55, 1.95, 1.55,
        "Head\nLayerNorm → MLP\n→ scalar per query",
        DEC, DEC_E, fs=9.2, weight="bold")
    box(ax, 9.45, 0.80, 1.95, 1.35,
        "Velocity field\n78 × 78\n" + r"$\bar{u}/U_{ref}$",
        IO, IO_E, fs=9.6, weight="bold")

    # ---- flows ----
    arrow(ax, 2.20, 4.87, 2.75, 4.87)                     # B -> encoder
    arrow(ax, 2.20, 1.62, 2.75, 1.62)                     # queries -> query feats
    arrow(ax, 5.10, 4.87, 5.63, 4.10, color=CORE_E)       # memory -> decoder (cross-attn keys)
    ax.text(5.28, 4.62, "keys/values", fontsize=8.0, color=CORE_E, rotation=-38)
    arrow(ax, 5.10, 1.62, 5.63, 1.95, color=IO_E)         # q -> decoder
    ax.text(5.30, 1.44, "q", fontsize=9.0, color=IO_E)
    arrow(ax, 8.90, 3.30, 9.45, 3.30, color=DEC_E)        # decoder -> head
    arrow(ax, 10.42, 2.55, 10.42, 2.15, color=IO_E)       # head -> field

    # ---- footnote ----
    ax.text(0.15, 0.30,
            "One model for the whole family of layouts — no per-layout retraining.  "
            "1.63 M parameters.  Corrected column-permute axial attention "
            "(arch. rev. uff-axial-fix).",
            fontsize=8.6, color=MUTED)

    fig.savefig(OUT / "architecture.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "architecture.svg", bbox_inches="tight", facecolor="white")
    print("wrote", OUT / "architecture.png")


if __name__ == "__main__":
    main()
