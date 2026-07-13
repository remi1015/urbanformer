# Results, work package by work package

Every metric is computed over fluid cells only (`fluid_mask_mid == 1`). `R2` is over the pooled fluid cells of the test set, not averaged per case.

---

## WP0 — Preprocessing

5,225 cases, 78 x 78, `H_REF = 0.870`, `SOLID_CODE = 8`, `DX = 1.0` (isotropic lattice units).

Buildings per case: min 16, mean 28.0, max 44.

Canonical morphology descriptors across the dataset:

| descriptor | min | mean | max | std |
|---|---:|---:|---:|---:|
| lambda_p | 0.2091 | 0.3912 | 0.5358 | 0.0459 |
| lambda_f | 0.0154 | 0.0285 | 0.0448 | 0.0045 |
| h_m | 0.4710 | 0.6329 | 0.7971 | 0.0668 |
| h_rms | 0.0480 | 0.1084 | 0.1712 | 0.0178 |
| h_skew | -2.1092 | -0.0519 | 1.5111 | 0.4613 |
| h_kurt | -1.6575 | -0.5773 | 5.3057 | 0.6037 |
| gamma_m | 0.2987 | 0.4162 | 0.5700 | 0.0488 |
| h_max | 0.6087 | 0.8290 | 0.8696 | 0.0525 |

Alignedness family (Lu et al. 2023). `gamma_m*` and `gamma_s` are C/H aspect ratios, hence the different scale:

| descriptor | min | mean | max | std |
|---|---:|---:|---:|---:|
| gamma_m | 0.2987 | 0.4162 | 0.5700 | 0.0488 |
| gamma_m_star | 28.4479 | 53.4519 | 96.3143 | 10.4813 |
| gamma_s | 14.7314 | 31.4334 | 60.4049 | 6.3063 |
| gamma_p | 0.4944 | 0.7977 | 1.0000 | 0.1067 |
| gamma_c | 0.1085 | 0.2418 | 0.3858 | 0.0378 |

Periodicity is handled explicitly in `row_canyons`: the row is rotated so a building sits at index 0, which prevents an air run from being split across the `x = 0 / x = Nx` seam. Penetrating streets (`H_ahead == 0`) leave `C/H` undefined, so `gamma_m*` credits those rows with the strongest sheltered-canyon ratio in the layout, which keeps the quantity finite and guarantees `gamma_m* >= gamma_s`.

Split: by full urban layout, 70/15/15, seed 42, giving 3,657 / 783 / 785. Overlap asserted empty.

---

## WP1 — U-Net baseline

3-level U-Net, 4 input channels (`height_map`, `footprint_mask`, `x_grid`, `y_grid`), masked MSE, Adam, lr 1e-3, batch 8, early stop patience 10. `F.interpolate` after each up-conv so the odd 78 -> 39 -> 19 -> 9 pyramid round-trips.

Early stop at epoch 18, best val 0.7031 at epoch 8.

| metric | value |
|---|---:|
| Test RMSE | 0.8217 |
| Test MAE | 0.4821 |
| Test R2 | 0.7194 |

Per-case: best RMSE 0.2773, worst 1.8783, 95th percentile 1.2773.

*(These are on the original WP0 split. The WP5 core-split retrain of the same architecture gives R2 = 0.7129.)*

---

## WP2 — Pooled building-token Transformer

Encoder: 3 layers, `d_model = 128`, 4 heads, FF 256. Mean-pool to `z_geom`. Two decoders compared:

- **base**: MLP on `[query_xy, z_geom]`, 464,769 params.
- **FiLM**: random Fourier query embedding (64 frequencies, `FOURIER_SCALE = 10.0`) and two FiLM blocks modulated by `z_geom`, 695,169 params.

Trained on the core split, 2,000 fluid query points per case per step, token order shuffled for set-invariance.

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| WP1 U-Net (raster) | 0.8217 | 0.4821 | 0.7194 |
| WP2 pooled, base | 1.2796 | 0.8635 | 0.3195 |
| WP2 pooled + Fourier/FiLM | 1.1587 | 0.7722 | 0.4421 |

**Reading.** The first WP2 iteration collapsed to a near-mean field at R2 ≈ 0.06. Fourier features plus FiLM recovered a large fraction of that (0.06 -> 0.44), which proves the decoder's spectral bias was *a* bottleneck. It did not reach the U-Net. The residual gap is mean-pooling: one latent vector cannot carry per-location geometry. That is the decisive read, and it is what makes WP3's per-query cross-attention mandatory rather than optional.

> **Open discrepancy to resolve before publishing.** WP5's `wp2-pooled-core-retrain` row scores R2 = 0.2921 on `core_test`, below the 0.4421 recorded here. The two runs should be reconciled (independent retrain, different seed and schedule) or the WP5 row relabeled. Do not ship the repo with both numbers unqualified.

---

## WP3 — UrbanFormer-Field (UF-F)

```
G_theta(B, x, y) -> u_bar(x, y, h_m/2) / U_ref
```

Decodes the whole query grid jointly rather than independently sampled points, so neighbouring queries cohere into sharp structures. Each block is `{relative-geometry cross-attention to buildings -> axial query self-attention over the grid -> FFN}`.

Levers, each behind a flag so the ablation matrix attributes each gain:

| lever | flag | attacks |
|---|---|---|
| axial query self-attention | `QUERY_SELFATTN` | no spatial coherence (load-bearing) |
| streamwise-anisotropic relative cross-attn | `REL_COORD` | conditional geometry, upstream/downstream asymmetry |
| query-local height patch + scalars | `QUERY_PATCH`, `QUERY_KNN` | information-poor query, train-R2 ceiling |
| residual refinement depth | `RESIDUAL_DEPTH` | range compression (coarse -> sharp) |
| gradient + spectral loss | `SPECTRAL_LOSS` | missing high-frequency energy |
| global morphology token | `MULTISCALE` | meso/global scale (WP4 bridge) |

1,633,969 parameters. Best val 0.3871 at epoch 33.

| metric | value |
|---|---:|
| Test RMSE | 0.6192 |
| Test MAE | 0.3722 |
| Test R2 | 0.8461 |
| rel-L2 | 0.3553 |

**Reading.** The diagnostics that motivated UF-F showed attention was already well-structured in the WP2 iterations (entropy 0.85, top-1 0.13) while R2 sat flat at 0.471. The failure was never attention placement. It was spatial coherence plus range compression: predictions collapsed into a narrow band, channels and wakes blurred, train R2 plateaued near 0.60 from coordinates alone. Axial self-attention over the query grid is what fixes coherence, and it is the load-bearing lever.

The hypothesis going in was that the object-based field would still trail the U-Net's 0.719. It did not. That hypothesis is falsified.

### Architecture revision `uff-axial-fix`

See the bug section of the top-level README. In short: the column branch of `AxialSelfAttention` never permuted `(B, Ny, Nx, D) -> (B, Nx, Ny, D)` before reshaping, so it attended over rows twice and scattered the second result transposed. `reshape` never raised because the element count factors identically. Every UF-F number prior to this run, including the original headline R2 = 0.8284, was produced with streamwise coupling only.

The result stands. The mechanism attributed to it does not. Weights do not transfer, so WP3 is a retrain. It also restores WP-isolation: WP3 and WP4 now differ in `MORPH_MODE` alone, whereas before they differed in `MORPH_MODE` *and* axial correctness, which is where WP4's apparent `+0.386 dR2` came from.

---

## WP4 — Morphology-aware UrbanFormer

```
G_theta(B, m, x, y) -> u_bar(x, y, h_m/2) / U_ref
```

Single isolated variable: `MORPH_MODE ∈ {none, token, query}`. Everything else (encoder, relative cross-attention, axial self-attention, residual depth, grad + spectral loss, augmentation) frozen at the UF-F config, so any R2 change is attributable to `m`.

| `MORPH_MODE` | how `m` enters | roadmap |
|---|---|---|
| `none` | not used, reproduces WP3 UF-F | control (A) |
| `token` | `m -> MLP -> global token`, prepended to the building set | main (B) |
| `query` | `m` concatenated to every query feature | (C) |

Recipe corrections held constant across every run, so the morphology comparison is not noise-driven:

1. Regularization rolled back to the best logged run (`WD = 5e-4`, `DROPOUT = 0.15`, val 0.3804) rather than the config that shipped in iter1 (`WD = 1e-3`, `DROPOUT = 0.2`, val 0.3867).
2. `ReduceLROnPlateau` replaced by cosine-to-floor with `T_MAX = 50`. The plateau scheduler was firing on ±0.1 val noise, which was the real cause of the "destructive second halving".
3. Per-epoch R2 readout from a precomputed val-target variance, so model selection and the morphology effect are visible live.
4. `MORPH_DROP_GROUPS` for per-feature-group attribution, zeroing a group in standardized space.

`MORPH_MODE = "token"`:

| metric | value |
|---|---:|
| Test RMSE | 0.6397 |
| Test MAE | 0.3895 |
| Test R2 | 0.8358 |
| rel-L2 | 0.3671 |

### Decision rule, stated before the run

> WP4 succeeds only if `token` (or `query`) beats `none` on R2 **and** the gain dies under shuffle (`R2(shuffle) ≈ R2(none)`). If shuffle keeps the gain, it was the morphology head's extra parameters, not the information.

`token` = 0.8358, `none` = 0.8461. The first condition failed. The shuffle control was never in play.

**Conclusion: honest null.** The building-token set already encodes everything `m` adds. `lambda_p`, `lambda_f`, `h_m`, `h_rms`, the height moments and `gamma_m` are all computable from the tokens, and the encoder evidently computes them. A global descriptor vector is not a free source of information when the model already sees the objects the descriptors summarize.

The binding constraint is therefore layout generalization (val floor ≈ 0.38), not representation. That floor is a representation lever, out of scope for WP4.

**Remaining.** `query` and `token+shuffle` are specified in the matrix but not yet logged to `wp4_results.json`. Each run is a separate Kaggle session (dual T4 runs two configs in parallel). The null rests on `token` vs `none` alone until those land.

---

## WP5 — Generalization evaluation

Evaluation only. All four models retrained from scratch on the identical core split (`core_train` = 2,518, `core_val` = 539, `core_test` = 541), so any delta is architecture, not training-set exposure. Every OOD number is read relative to `core_test`, never relative to the original WP1 to WP4 test numbers, which used the larger 3,657-case train set.

### Provenance guard

A checkpoint from an earlier run scores against a training set it never saw, so its row is not comparable. WP5 refuses it. Two kinds of evidence, in order of strength:

1. **Positive.** The config states what it was trained on (`WP`, `SPLIT`, `N_TRAIN`). Dispositive.
2. **Negative.** The config lacks keys the WP5-era training script always writes (`LR_SCHED`, `LR_MIN`, `T_MAX`, `WEIGHT_DECAY`, `M_DIM`). A backstop for pre-provenance checkpoints.

Plus one check that makes the WP4 row mean anything: a `WP4-morph` checkpoint with `MORPH_MODE = "none"` is silently a second WP3 run, and the headline WP4 finding would then be an artifact of the shipped file rather than of the model. `EXPECT_MORPH_MODE` catches it.

There is no deny-list. `MULTISCALE` was briefly treated as a stale-checkpoint fingerprint, but it is a legitimate WP3 architecture lever. It correlated with staleness only because one stale artifact happened to carry it. The missing-keys test rejects that artifact on its own.

### Core test

| model | RMSE | MAE | R2 | rel-L2 | Spearman |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.8457 | 0.4900 | 0.7129 | 0.4853 | 0.8755 |
| WP2-pool | 1.3280 | 0.8819 | 0.2921 | 0.7620 | 0.5804 |
| WP3-UFF | 0.6192 | 0.3722 | 0.8461 | 0.3553 | 0.9483 |
| WP4-morph | 0.6397 | 0.3895 | 0.8358 | 0.3671 | 0.9451 |

### OOD regimes

Eight tails at `TAIL_PCT = 95`. `high_h_max` was dropped as degenerate on the quantized `h_max` column and replaced by `high_lambda_f`.

ΔR2 versus `core_test`, negative meaning degradation:

| model | h_rms↑ | λf↑ | γ↑ | γ↓ | λp↑ | λp↓ | skew↑ | kurt↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | -0.0216 | +0.0193 | -0.0128 | -0.0561 | -0.0529 | +0.0018 | -0.0153 | +0.0115 |
| WP2-pool | -0.0289 | +0.1997 | -0.1869 | +0.1082 | +0.0900 | -0.1095 | -0.0389 | +0.0402 |
| WP3-UFF | -0.0124 | -0.0069 | -0.0208 | -0.0320 | -0.0351 | -0.0223 | -0.0073 | -0.0014 |
| WP4-morph | -0.0140 | -0.0056 | -0.0324 | -0.0366 | -0.0314 | -0.0210 | -0.0088 | +0.0021 |

Note WP2-pool's positive deltas on `λf↑`, `γ↓` and `λp↑`. A model that has collapsed toward the conditional mean improves on regimes whose target variance is easier, which is a diagnostic of collapse rather than of robustness. Its robustness gap is *negative* (-0.0092) for the same reason. Do not read that row as generalization.

### Ranking

| model | core R2 | mean OOD R2 | robustness gap | worst regime | worst R2 |
|---|---:|---:|---:|---|---:|
| WP3-UFF | 0.8461 | 0.8288 | 0.0173 | λp↑ | 0.8110 |
| WP4-morph | 0.8358 | 0.8173 | 0.0185 | γ↓ | 0.7991 |
| U-Net | 0.7129 | 0.6972 | 0.0158 | γ↓ | 0.6568 |
| WP2-pool | 0.2921 | 0.3014 | -0.0092 | γ↑ | 0.1053 |

### Automated outcome read

- UF-F ≥ U-Net on `core_test` -> the object-based representation matches and exceeds the rasterized CNN.
- Both degrade similarly on OOD (0.0185 vs 0.0158) -> the failure is data-driven, not architecture-driven.
- Hard for all models: `γ↑` -> a structural limit of the surrogate approach on this dataset, not of any one model.

### Physics-oriented metrics

Beyond aggregate R2, WP5 also reports `plane_avg_err`, `wake_rmse` (6 cells downstream), `canyon_rmse` (4 cells), `deficit_rmse`, and low/high-speed area errors at thresholds `u/U_ref` of 0.5 and 1.5. These say *where* each model fails, which is what a client's engineer actually asks about.
