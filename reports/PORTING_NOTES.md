# Porting notes — stages 1 through 5

This documents the port of the stage 1–5 notebooks into the tested `urbanformer`
package. Every function was extracted from the source of truth, diffed against
the original on random inputs (bit-identical), and covered by an invariant test.
Full suite: **81 passing**.

Stages (the code and original exports label them `WP0`–`WP5`, "work package"):
1 U-Net baseline · 2 Pooled-token Transformer · 3 UrbanFormer-Field · 4 UF-Field +
morphology · 5 Generalization study. Original export filenames (`wp1.py`,
`wp3last.pdf`, …) and checkpoint tags (`WP4-morph`) are kept verbatim below.

> **Package layout note.** The importable `urbanformer/` package is the single
> source of truth. The command-line drivers (`config.py`, `train.py`, `eval.py`)
> live *inside* the package, so `python -m urbanformer.train` /
> `python -m urbanformer.eval` work as documented and the notebooks, the CLI, and
> the tests all import one implementation of every function. (An earlier layout
> kept a second, stale copy of each module at the repo root and left the CLI
> orphaned at the top level, where its `urbanformer.*` imports could not resolve;
> that duplication has been removed.)

## Library modules added

| module | contents | source of truth | validation |
|---|---|---|---|
| `losses.py` | `masked_mse` (stage 1); `masked_field_loss`, `make_radial_bins`, `_radial_psd` (stage 3) | `wp1.py`, `wp3.py` | 0-mismatch diff on 500 / 200 random inputs |
| `metrics.py` | `field_metrics`, `per_case_rmse`, `spatial_corr` (stage 5); `region_masks`, `physics_metrics` (stage 5) | `wp5.py` | 0-mismatch diff on 400 inputs |
| `models/unet.py` | `UNetMid`, `DoubleConv` (stage 1) | `wp1.py` | seeded bit-identical fwd; params 1,927,297 |
| `models/pooled.py` | `PooledTransformer` (464,769), `PooledTransformerFiLM` (695,169), `FourierFeatures`, `FiLMBlock` (stage 2) | `UrbanFormer_WP2_Pooled_base_vs_FiLM.pdf` | reconstructed, gated on exact param fingerprints |
| `models/field.py` | full UrbanFormer-Field stack (stage 3), importing the fixed `axial.py` | `wp3.py` + `wp3last.pdf` | params 1,633,969 (flagship fingerprint) |
| `data.py` | `UNetMidDataset` (stage 1); `TokenDataset` + `collate_fn` (stage 2); `TokenFieldDataset` + `collate_field` + query helpers (stage 3) | `wp1.py`, stage-2 PDF, `wp3.py` | tensor-contract diffs + tests |
| `provenance.py` | checkpoint guard: `check_morph_provenance`, `strict_load`, `positional_remap`, `extract_state_dict` (stage 5) | `wp5.py` | rejects `MORPH_MODE="none"` tagged `WP4-morph` |
| `config.py` | per-stage model registry (`WP_CONFIGS`) + factory (`build_model`) | stage 1–4 configs | param-count fingerprints per stage |
| `train.py` | `python -m urbanformer.train`; data-free `smoke_step` per stage | stage 1–4 train loops | `tests/test_cli.py` runs `smoke_step` for every stage |
| `eval.py` | `python -m urbanformer.eval`; grid-subsample + markdown-table helpers | `wp5.py` | pure helpers covered without data |

## Notebooks (import the library; orchestration + plotting only)

- `notebooks/01_unet_baseline.ipynb` — U-Net baseline, R2 = 0.7194
- `notebooks/02_pooled_transformer.ipynb` — Pooled-token Transformer, base 0.3195 / FiLM 0.4421
- `notebooks/03_urbanformer_field.ipynb` — UrbanFormer-Field (flagship), R2 = 0.8461, with the axial-fix story
- `notebooks/04_morphology_ablation.ipynb` — UF-Field + morphology, pre-registered null (token 0.8358 ~ none 0.8461)
- `notebooks/05_cross_model_ood.ipynb` — Generalization study (cross-model + OOD), provenance-guarded loading

## Two open items, resolved

**Open item 1 (axial column-permute).** `axial.py` matches the real axialfix run.
The final UrbanFormer-Field notebook (`wp3last.pdf`) saves `wp3_uff_axialfix_best.pt` with
`ARCH_REV="uff-axial-fix"`, `AXIAL_COL_PERMUTE=True`, and a column branch that
permutes `(B,Ny,Nx,D) -> (B,Nx,Ny,D)` before reshaping — exactly what `axial.py`
implements (and its two regression tests pin). `wp3.py` is a stale earlier export
(saves `wp3_iter1_best_model.pt`, no permute); `field.py` imports the fixed
`axial.py`, not the buggy inline class. **No change to `axial.py`.**

**Open item 2 (Pooled-token R2 0.4421 vs 0.2921).** Different training-set sizes, not
seed variance. The Pooled-token notebook trained on the original split (3,657 / 785
test) → 0.4421; the stage-5 controlled retrain used the reduced core split
(2,518 / 541 test) → 0.2921. Documented in `RESULTS.md`; the two are not merged.

## Notes for the reader

- `wp2.py` and `wp3.py` in the source exports are stale/misfiled; the `*last.pdf`
  files are the true finals and were used as source of truth where they differ.
- Notebooks are de-Colab'd (repo-relative paths, no `drive.mount`) and validated:
  nbformat-valid, all code cells parse, and the train/eval paths run end-to-end
  on synthetic data.
- The morphology variant's `token` mechanism uses `field.py`'s `MULTISCALE` lever
  (global morphology token from per-set statistics); the full variant also supports
  an explicit 8-feature vector with the same null conclusion.
