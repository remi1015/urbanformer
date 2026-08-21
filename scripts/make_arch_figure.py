"""
Render the UrbanFormer-Field (UF-F) architecture diagram.

Data-free: this figure documents the stage 3/4 forward pass and can be regenerated
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


def box(ax, x, y, w, h, text, fc, ec, fs=10.5, weight="normal", title=None):
    """Rounded box with centered text; optional bold `title` line on top."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.014,rounding_size=0.025",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    if title is not None:
        ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center",
                fontsize=fs + 1.0, color=ec, weight="bold", zorder=3)
        ax.text(x + w / 2, y + (h - 0.48) / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, weight=weight, zorder=3, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, weight=weight, zorder=3, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.9, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=15,
        linewidth=lw, color=color, zorder=1,
        shrinkA=3, shrinkB=3))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.6, 7.4), dpi=220)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    ax.text(0.20, 7.06, "UrbanFormer-Field  —  the flagship model", fontsize=17,
            weight="bold", color=INK)
    ax.text(0.20, 6.66,
            r"$G_\theta(B,\, m,\, x, y)\ \rightarrow\ \bar{u}(x, y,\ z{=}h_m/2)\,/\,U_{ref}$",
            fontsize=12.5, color=MUTED)

    # ---- inputs (left column) ----
    box(ax, 0.20, 4.55, 2.45, 1.35,
        "{ [x_c, y_c, w, l, h] }\nvariable-length set, 16–44",
        IO, IO_E, fs=9.6, title="Building set  B")
    box(ax, 0.20, 1.15, 2.45, 1.20,
        "(x, y) on the\nmid-canopy plane",
        IO, IO_E, fs=9.6, title="Query points")

    # ---- encoder ----
    box(ax, 3.05, 4.25, 2.60, 1.68,
        "linear embed →\n3× Transformer\nencoder layers\n(permutation-invariant set)",
        ENC, ENC_E, fs=9.2, title="Token Encoder")
    ax.text(4.35, 4.08, "memory", fontsize=9.0, color=ENC_E,
            ha="center", style="italic")

    # ---- query featurization ----
    box(ax, 3.05, 0.80, 2.60, 1.75,
        "Fourier(x, y)\n+ kNN scalars\n+ local height patch\n→ linear  →  q",
        IO, IO_E, fs=9.4, title="Query features")

    # ---- decoder container ----
    dx, dy, dw, dh = 6.30, 0.95, 3.55, 5.10
    ax.add_patch(FancyBboxPatch(
        (dx, dy), dw, dh, boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.7, edgecolor=DEC_E, facecolor="#f4f0fa", zorder=1))
    ax.text(dx + dw / 2, dy + dh + 0.18, "Decoder  ·  4 residual blocks",
            ha="center", va="bottom", fontsize=11.5, weight="bold", color=DEC_E)

    ix, iw = dx + 0.22, dw - 0.44
    box(ax, ix, 4.52, iw, 1.10,
        "Relative-geometry\ncross-attention  (query → buildings)\nanisotropic streamwise / spanwise,\nupstream/downstream asymmetry",
        CORE, CORE_E, fs=8.7)
    box(ax, ix, 2.86, iw, 1.10,
        "Axial self-attention\nover the (Ny, Nx) query grid\nrow then column,  O(Nx + Ny)\n(load-bearing lever)",
        CORE, CORE_E, fs=8.7)
    box(ax, ix, 1.72, iw, 0.74,
        "Feed-forward  +  residual", DEC, DEC_E, fs=9.4)
    ax.text(dx + dw / 2, 1.34, "×4   (coarse → fine)", ha="center",
            fontsize=9.0, color=DEC_E, style="italic")

    # internal vertical arrows in the decoder
    cx = dx + dw / 2
    arrow(ax, cx, 4.52, cx, 3.96, color=DEC_E)
    arrow(ax, cx, 2.86, cx, 2.46, color=DEC_E)

    # ---- head / output ----
    box(ax, 10.35, 3.20, 2.25, 1.45,
        "LayerNorm → MLP\n→ scalar per query",
        DEC, DEC_E, fs=9.4, title="Head")
    box(ax, 10.35, 1.00, 2.25, 1.45,
        "78 × 78\n" + r"$\bar{u}/U_{ref}$",
        IO, IO_E, fs=9.8, title="Velocity field")

    # ---- flows between blocks ----
    arrow(ax, 2.65, 5.20, 3.05, 5.15)                    # B -> encoder
    arrow(ax, 2.65, 1.72, 3.05, 1.68)                    # queries -> query feats
    arrow(ax, 5.65, 5.10, 6.42, 5.05, color=CORE_E)      # memory -> cross-attn (keys/values)
    ax.text(5.98, 5.34, "K, V", fontsize=8.8, color=CORE_E, ha="center", weight="bold")
    arrow(ax, 5.65, 1.80, 6.42, 2.45, color=IO_E)        # q -> decoder
    ax.text(5.96, 2.28, "q", fontsize=9.8, color=IO_E, weight="bold")
    arrow(ax, 9.85, 3.92, 10.35, 3.92, color=DEC_E)      # decoder -> head
    arrow(ax, 11.47, 3.20, 11.47, 2.45, color=IO_E)      # head -> field

    # ---- footnote ----
    ax.text(0.20, 0.34,
            "One model for the whole family of layouts — no per-layout "
            "retraining.    1.63 M parameters.",
            fontsize=9.2, color=MUTED)

    fig.savefig(OUT / "architecture.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    fig.savefig(OUT / "architecture.svg", bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    print("wrote", OUT / "architecture.png")


if __name__ == "__main__":
    main()
