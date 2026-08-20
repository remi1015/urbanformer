"""
Qualitative field figures: CFD ground truth vs UrbanFormer-Field prediction.

Two modes:

    python scripts/make_field_figure.py                 # REAL: needs data + a checkpoint
    python scripts/make_field_figure.py --schematic      # data-free labeled illustration

REAL mode renders, per example case, a three-panel row
``[ CFD (ground truth) | UrbanFormer-Field | signed error ]`` over the fluid
cells (buildings masked), exactly as ``notebooks/05_cross_model_ood.ipynb`` does.
It requires the processed dataset (``data/processed/``) and a core-retrain
checkpoint (``checkpoints/``); fetch both with ``scripts/fetch_data.py --all``.
Like ``scripts/make_figures.py`` it **does not fabricate fields it cannot load**:
if the assets are missing it says exactly what it needs and exits cleanly.

SCHEMATIC mode needs no data. It renders a *synthetic, illustrative* field in the
same layout so a reader can see the output format — masked buildings, low-speed
wakes downstream of buildings, faster open canyons, and an error map. It is
clearly banner-labeled as an illustration and is **not** CFD data or a real model
prediction. This is the image committed to ``docs/figures/`` for the README.

    python scripts/make_field_figure.py --schematic      # -> docs/figures/field_schematic.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "splits"
CKPT_DIR = ROOT / "checkpoints"
OUT = ROOT / "docs" / "figures"

# Checkpoint directory per UF-F tag (fetch_data.py unzips into these).
CKPT_DIRS = {"WP3-UFF": CKPT_DIR / "wp3_uff", "WP4-morph": CKPT_DIR / "wp4_morph"}
EXPECTED_MORPH = {"WP3-UFF": "none", "WP4-morph": "token"}


# ---------------------------------------------------------------------------
# shared rendering (identical colormap/masking to notebooks 03/05)
# ---------------------------------------------------------------------------
def _velocity_cmap(vmin: float, vmax: float) -> mcolors.LinearSegmentedColormap:
    colors = [(0, 0, 0.5), (0, 0.8, 0.8), (0, 0.25, 0), (0.7, 0.7, 0), (0.4, 0, 0)]
    positions = [0, 0.15, (0 - vmin) / (vmax - vmin), 0.55, 1]
    return mcolors.LinearSegmentedColormap.from_list("uvel", list(zip(positions, colors)))


def _panel(ax, u_norm, solid, title, vmin=-1.0, vmax=3.0):
    cmap = _velocity_cmap(vmin, vmax)
    data = u_norm.copy()
    data[solid] = 0.0
    cax = ax.imshow(data, cmap=cmap, interpolation="bicubic",
                    vmin=vmin, vmax=vmax, origin="upper")
    # overlay building cells as solid grey
    ax.imshow(np.where(solid, 0.0, np.nan), cmap="gray",
              interpolation="nearest", origin="upper", zorder=2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(cax, ax=ax, orientation="horizontal",
                 fraction=0.046, pad=0.08, extend="both")


def _error_panel(ax, pred, target, fluid, title="signed error  (pred − truth)"):
    err = (pred - target) * fluid
    v = float(np.abs(err[fluid > 0]).max()) if (fluid > 0).any() else 1.0
    im = ax.imshow(np.where(fluid > 0, err, np.nan), cmap="RdBu_r",
                   vmin=-v, vmax=v, origin="upper")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.08)


def _render_rows(rows, suptitle, outfile, subtitles=None):
    """rows: list of (pred, target, fluid) float arrays (Ny, Nx)."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 4.4 * n), squeeze=False)
    for r, (pred, target, fluid) in enumerate(rows):
        solid = fluid == 0
        rmse = float(np.sqrt(np.mean(((pred - target)[fluid > 0]) ** 2))) \
            if (fluid > 0).any() else float("nan")
        _panel(axes[r][0], target, solid, "CFD (ground truth)")
        _panel(axes[r][1], pred, solid, "UrbanFormer-Field")
        _error_panel(axes[r][2], pred, target, fluid)
        tag = subtitles[r] if subtitles else f"case {r}"
        axes[r][0].set_ylabel(f"{tag}\nfluid-cell RMSE = {rmse:.3f}",
                              fontsize=10, rotation=90, labelpad=12)
    fig.suptitle(suptitle, fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=150)
    print("wrote", outfile)


# ---------------------------------------------------------------------------
# REAL mode: load data + checkpoint, predict, render
# ---------------------------------------------------------------------------
def _assets_present(tag: str) -> bool:
    return (DATA_DIR.exists() and any(DATA_DIR.iterdir())
            and CKPT_DIRS[tag].exists())


def _report_missing(tag: str) -> None:
    print("Cannot render the real field figure: dataset and/or checkpoint missing.\n")
    print(f"  expected processed cases under: {DATA_DIR}")
    print(f"  expected {tag} checkpoint under: {CKPT_DIRS[tag]}")
    print("\nFetch them first:")
    print("  python scripts/fetch_data.py --all")
    print("\nOr render the data-free schematic instead:")
    print("  python scripts/make_field_figure.py --schematic")


def _load_uff(tag: str):
    import torch

    import urbanformer.models.field as field_mod
    from urbanformer.models.field import UrbanFormerField
    from urbanformer.provenance import (
        check_morph_provenance,
        extract_state_dict,
        find_checkpoint,
        strict_load,
    )

    obj = torch.load(find_checkpoint(str(CKPT_DIRS[tag])),
                     map_location="cpu", weights_only=False)
    sd, cfg = extract_state_dict(obj)
    cfg = cfg or {}
    check_morph_provenance(tag, cfg)                 # refuse a mislabeled checkpoint
    field_mod.MULTISCALE = cfg.get("MORPH_MODE", EXPECTED_MORPH[tag]) == "token"
    model = UrbanFormerField()
    strict_load(model, sd, tag)
    return model.eval()


def _resolve_case_dirs(split: str, explicit):
    if explicit:
        dirs = [DATA_DIR / c for c in explicit]
    else:
        split_file = SPLITS_DIR / f"{split}_cases.txt"
        if split_file.exists():
            names = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
            dirs = [DATA_DIR / n for n in names]
        else:
            dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    return [d for d in dirs if (d / "building_tokens.npy").exists()
            and (d / "U_mid.npy").exists()]


def _render_real(args) -> int:
    tag = args.model
    if not _assets_present(tag):
        _report_missing(tag)
        return 0

    import torch
    from torch.utils.data import DataLoader

    from urbanformer.data import TokenFieldDataset, collate_field

    case_dirs = _resolve_case_dirs(args.split, args.cases)
    if not case_dirs:
        print(f"No usable cases found for split={args.split!r} under {DATA_DIR}.")
        return 0

    # Rank a sample by fluid-cell RMSE, then show best / median / worst spread.
    sample = case_dirs[: args.sample]
    model = _load_uff(tag)
    loader = DataLoader(TokenFieldDataset(sample, train=False),
                        batch_size=8, shuffle=False, collate_fn=collate_field)
    preds, tgts, masks = [], [], []
    with torch.no_grad():
        for tokens, pad, qxy, qf, pa, U, fluid in loader:
            Ny, Nx = U.shape[1], U.shape[2]
            pred = model(tokens, pad, qxy, qf, pa, Ny, Nx).reshape(-1, Ny, Nx)
            preds.append(pred)
            tgts.append(U)
            masks.append(fluid)
    P = torch.cat(preds).numpy()
    T = torch.cat(tgts).numpy()
    M = torch.cat(masks).numpy()

    rmse = np.array([
        np.sqrt(np.mean(((P[i] - T[i])[M[i] > 0]) ** 2)) if (M[i] > 0).any() else np.nan
        for i in range(len(P))
    ])
    order = np.argsort(np.nan_to_num(rmse, nan=np.inf))
    if args.cases:
        picks = list(range(min(len(P), len(args.cases))))
        labels = [Path(sample[i].name).name for i in picks]
    else:
        n = min(args.n, len(order))
        sel = np.linspace(0, len(order) - 1, n).round().astype(int)
        picks = [int(order[s]) for s in sel]
        names = ["best", "median", "worst"] if n == 3 else [f"pct{int(100*s/(n-1))}" for s in range(n)]
        labels = [f"{names[k]} ({sample[picks[k]].name})" for k in range(n)]

    rows = [(P[i], T[i], M[i]) for i in picks]
    outfile = OUT / f"field_prediction_{tag.lower().replace('-', '_')}.png"
    _render_rows(rows, f"{tag}: CFD ground truth vs prediction "
                       f"({args.split}, fluid cells only)", outfile, subtitles=labels)
    return 0


# ---------------------------------------------------------------------------
# SCHEMATIC mode: synthetic, clearly-labeled illustration (no data)
# One building layout drives BOTH the input canopy (3-D boxes) and the output
# field, so the two docs images are a consistent input -> output pair.
# ---------------------------------------------------------------------------
def _synthetic_layout(seed: int = 3, Ny: int = 78, Nx: int = 78):
    """A synthetic building set. Returns (buildings, solid, Ny, Nx).

    Each building is a dict of grid-cell geometry ``{y0, x0, ny, nx, h}`` with a
    normalized height ``h in [0, 1]`` — i.e. the same information a 5-D input
    token carries. Buildings are scattered on a jittered grid (16-44 per case in
    the real data; a similar count here).
    """
    rng = np.random.default_rng(seed)
    solid = np.zeros((Ny, Nx), bool)
    buildings = []
    for cy in range(6, Ny - 8, 13):
        for cx in range(6, Nx - 8, 13):
            if rng.random() < 0.16:
                continue
            ny = int(rng.integers(6, 11))                   # spanwise (y) extent
            nx = int(rng.integers(6, 11))                   # streamwise (x) extent
            y0 = max(0, cy + int(rng.integers(-2, 3)))
            x0 = max(0, cx + int(rng.integers(-2, 3)))
            h = float(rng.uniform(0.12, 0.34))              # normalized roof height
            solid[y0:y0 + ny, x0:x0 + nx] = True
            buildings.append(dict(y0=y0, x0=x0, ny=ny, nx=nx, h=h))
    return buildings, solid, Ny, Nx


def _field_from_layout(buildings, solid, Ny, Nx, seed: int = 3):
    """A plausible-looking but entirely synthetic field for that layout.

    NOT physics: taller buildings shelter a stronger low-speed wake downstream
    (+x) and flow accelerates through open lanes, purely to illustrate the output
    *format*. Also returns a smoothed, slightly-biased "prediction" that mimics a
    good-but-imperfect surrogate (sharp wakes soften, small errors at edges).
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed + 100)
    u = np.full((Ny, Nx), 1.2, float)                       # free-stream base speed
    for b in buildings:
        y0, x0, ny, nx = b["y0"], b["x0"], b["ny"], b["nx"]
        strength = 1.1 + 2.0 * b["h"]                       # taller -> stronger wake
        for d in range(1, 26):                              # wake downstream (+x)
            xx = x0 + nx + d
            if xx >= Nx:
                break
            u[y0:y0 + ny, xx] -= strength * np.exp(-d / 11.0)
        for dy in (-2, -1):                                 # lateral jets (open lanes)
            if 0 <= y0 + dy:
                u[y0 + dy, x0:min(Nx, x0 + nx + 10)] += 0.55 * strength
        for dy in (0, 1):
            if y0 + ny + dy < Ny:
                u[y0 + ny + dy, x0:min(Nx, x0 + nx + 10)] += 0.55 * strength

    u += 0.55 * gaussian_filter(rng.standard_normal((Ny, Nx)), sigma=7)
    u = gaussian_filter(u, sigma=1.0)
    fluid = (~solid).astype(float)
    pred = gaussian_filter(u, sigma=1.6) + 0.10 * gaussian_filter(
        rng.standard_normal((Ny, Nx)), sigma=8)
    return pred * fluid, u * fluid, fluid


def _render_input_canopy(buildings, Ny, Nx, outfile) -> None:
    """3-D box plot of the building set — the model's geometric input."""
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")   # 3d projection auto-registers (mpl>=3.2)
    xs = [b["x0"] / Nx for b in buildings]
    ys = [b["y0"] / Ny for b in buildings]
    dxs = [b["nx"] / Nx for b in buildings]
    dys = [b["ny"] / Ny for b in buildings]
    dzs = [b["h"] for b in buildings]
    zeros = [0.0] * len(buildings)
    ax.bar3d(xs, ys, zeros, dxs, dys, dzs,
             color="#3b74b8", edgecolor="#123b66", linewidth=0.6, shade=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_zticks([0, 1 / 3, 2 / 3, 1])
    ax.set_xlabel("x  (streamwise, wind →)", labelpad=8)
    ax.set_ylabel("y  (spanwise)", labelpad=8)
    ax.set_zlabel("z  (height / h_ref)", labelpad=4)
    ax.set_title(f"Input: building set  B  ({len(buildings)} tokens "
                 "[x_c, y_c, w, l, h])\nexample synthetic layout", fontsize=11)
    ax.view_init(elev=26, azim=-58)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=150)
    print("wrote", outfile)


def _render_field_schematic(pred, target, fluid, outfile) -> None:
    solid = fluid == 0
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    _panel(axes[0], target, solid, "“CFD” ground truth")
    _panel(axes[1], pred, solid, "“UrbanFormer-Field”")
    _error_panel(axes[2], pred, target, fluid)

    banner = ("SCHEMATIC — illustrative field FORMAT only.  "
              "Synthetic data: NOT CFD, NOT a trained-model prediction.")
    fig.suptitle(banner, fontsize=12.5, weight="bold", color="#8a1c1c")
    fig.text(0.5, 0.005,
             "Wind blows left→right (+x); buildings are masked (grey); wakes are "
             "the low-speed regions just downstream of buildings.  Real figures: "
             "fetch data/checkpoints (scripts/fetch_data.py) then run "
             "scripts/make_field_figure.py.",
             ha="center", va="bottom", fontsize=8.4, color="#5b6570")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=150)
    print("wrote", outfile)


def _render_schematic() -> int:
    """Render the paired input-canopy and output-field illustrations."""
    buildings, solid, Ny, Nx = _synthetic_layout()
    _render_input_canopy(buildings, Ny, Nx, OUT / "input_canopy.png")
    pred, target, fluid = _field_from_layout(buildings, solid, Ny, Nx)
    _render_field_schematic(pred, target, fluid, OUT / "field_schematic.png")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CFD-vs-UrbanFormer field figures.")
    ap.add_argument("--schematic", action="store_true",
                    help="render the data-free labeled illustration (docs image)")
    ap.add_argument("--model", choices=sorted(CKPT_DIRS), default="WP3-UFF",
                    help="which UF-F checkpoint to visualize (real mode)")
    ap.add_argument("--split", default="core_test",
                    help="split file <split>_cases.txt under splits/ (real mode)")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="explicit processed-case names to render (real mode)")
    ap.add_argument("--n", type=int, default=3,
                    help="number of example cases (best…worst spread) in real mode")
    ap.add_argument("--sample", type=int, default=120,
                    help="cases to rank by RMSE before picking examples (real mode)")
    args = ap.parse_args(argv)

    if args.schematic:
        return _render_schematic()
    return _render_real(args)


if __name__ == "__main__":
    sys.exit(main())
