# Glossary: fluid mechanics ↔ machine learning

A dictionary for readers who are comfortable with ML but new to the fluid-dynamics
vocabulary in this repo. Every entry says what the term *means* and what *role* it
plays in the model or data. Definitions are grounded in the code
(`urbanformer/morphology.py`, `urbanformer/data.py`, `urbanformer/metrics.py`),
not just in general physics.

If you only read one thing: **UrbanFormer is a supervised surrogate model. The
inputs are building geometry; the target is one scalar field that a CFD simulator
produced; the network approximates the simulator, it does not re-derive physics.**

---

## 1. The physical setup, in one paragraph

Picture wind blowing across a neighborhood of buildings. The **urban canopy** is
the layer of air from the ground up to roughly roof height, threaded between the
buildings. Down in that layer the flow is slow, swirling, and completely governed
by *where the buildings are* — which is exactly what controls whether a street is
comfortable to walk down and whether pollution flushes out. UrbanFormer takes a
**horizontal slice** through that layer at mid-building height and predicts the
**time-averaged along-wind speed** at every point of the slice, straight from the
building layout — no new simulation per layout.

---

## 2. Geometry, coordinates, and flow

| Term | Plain meaning | Role here |
|---|---|---|
| **Urban canopy** | The air layer between/around buildings, up to ~roof height. | The region whose flow we predict. |
| **Canopy / mid-canopy plane** | A horizontal (x–y) slice at height `z = h_m / 2`, i.e. half the *mean* building height. | The 2-D domain of the output field. One fixed plane. |
| **Streamwise (x)** | The direction the wind blows (here always `+x`). | Axis 1 of the grid; the model has a built-in **upstream/downstream asymmetry** along it. |
| **Spanwise (y)** | Horizontal, perpendicular to the wind. | Axis 0 of the grid; flow is statistically symmetric in `±y`, which is why training augments with a spanwise flip. |
| **Wall-normal (z)** | Vertical. | Fixed: we only model the single plane `z = h_m/2`. |
| **Upstream / downstream** | Toward where the wind comes from / goes to. | A building shelters what's downstream of it; the cross-attention bias encodes this. |
| **Footprint** | The ground area a building covers (its base). | `height_map > 0`; the plan-view silhouette. |
| **Height map** | 2-D array of building height at each ground cell (0 = open ground). | Raw geometry input; the U-Net baseline consumes it as a raster. |

---

## 3. The velocity field and its normalization

| Symbol / name | Meaning | Role here |
|---|---|---|
| **`u_bar` (mean velocity)** | Time-averaged along-wind speed at a point. "Bar" = time average; the turbulence is averaged out. | The physical quantity being predicted (as a field). |
| **`u_ref` / `u_tau` (friction velocity)** | A reference speed set by the wall shear stress, `u_tau = sqrt(τ_wall / ρ)`. Standard non-dimensionalizing scale in wall-bounded flows. | Divides `u_bar` so the target is dimensionless and comparable across cases. |
| **`U_mid`** | The training target: `u_bar / u_tau` on the mid-canopy plane (`data.case_fields`). | The regression label field, shape `(Ny, Nx)`. |
| **Reynolds number / regime** | Ratio of inertial to viscous forces; sets how turbulent the flow is. | Held **fixed** across the dataset — the model generalizes over *layout*, not over flow regime. |

Think of the normalization the way you'd think of standardizing a regression
target: it removes a global scale so the network learns the *pattern*, not the
units.

---

## 4. The simulator (where the labels come from)

| Term | Meaning | Role here |
|---|---|---|
| **CFD** | Computational Fluid Dynamics — numerically solving the equations of fluid motion. | The slow ground-truth generator we are replacing. |
| **LBM (Lattice-Boltzmann Method)** | A CFD method that evolves particle-distribution functions on a regular lattice and recovers the flow from their moments. Naturally parallel; common for complex geometries. | Produced all **5,225** labeled cases. For ML purposes it is just an (expensive, deterministic) label oracle. |
| **Surrogate / emulator** | A cheap model trained to reproduce a simulator's output. | What UrbanFormer *is*. |
| **Lattice units** | The simulator's internal, dimensionless unit system (grid spacing `DX = 1`). | Why cell size is 1 and lengths are counted in cells. |

---

## 5. Domain, grid, and masks

| Term | Meaning | Role here |
|---|---|---|
| **Doubly periodic domain** | The block tiles infinitely in both `x` and `y`: what exits the right edge re-enters on the left, same for top/bottom. | Removes boundary artifacts and justifies **periodic** morphology descriptors and periodic-translation augmentation. |
| **78 × 78 grid** | The mid-plane is discretized into `Ny × Nx = 78 × 78` cells. | `Q = Ny·Nx = 6084` query points per case for the field models. |
| **`geom` / cell-type code** | Integer label per cell; `SOLID_CODE = 8` marks a building cell at the plane. | Distinguishes fluid from solid. |
| **Fluid cell vs solid cell** | A cell that is air vs a cell blocked by a building at this height. | `fluid_mask_mid = (geom != 8)`. **The loss and every metric are computed over fluid cells only** — predictions inside buildings are ignored. |
| **Masked loss** | Loss summed over fluid cells and divided by their count. | See `losses.masked_mse`; solid cells contribute nothing to the gradient. |

---

## 6. Buildings as tokens

Each building is reduced to a **5-dim token**, all features normalized to `[0, 1]`
(`data.build_tokens`):

```
token = [ x_center, y_center, l_x, l_y, h ]
          └ centre ┘ └ width/length ┘ └height┘
```

- `x_center, y_center` — footprint centroid, normalized by `Nx, Ny`.
- `l_x` — streamwise extent (width along the wind), normalized by `Nx`.
- `l_y` — spanwise extent (length across the wind), normalized by `Ny`.
- `h`   — roof height, normalized by a global reference height `h_ref`.

The set of tokens **is** the model's input geometry. Because buildings have no
natural order, the encoder is **permutation-invariant** and training shuffles
token order. Variable set size (16–44) is handled by padding + a boolean
`padding_mask`.

---

## 7. Morphology descriptors

"Morphology" = summary statistics of the building layout. The canonical **8-vector**
(`morphology.MORPHOLOGY_KEYS`, order matters) plus an **alignedness family** used
for ablations. In ML terms these are *hand-engineered global features*; WP4 tests
whether adding them helps, and finds they are **redundant** given the token set
(the encoder can compute them itself).

### The canonical 8-vector

| Descriptor | Name | Definition (as coded) | Intuition |
|---|---|---|---|
| `lambda_p` | **plan area density** | footprint area / total area (`footprint.sum() / (Nx·Ny)`) | How much ground the buildings cover (packing). |
| `lambda_f` | **frontal area density** | Σ(height · spanwise width) / total area | How much the buildings "face into" the wind (blockage). |
| `h_m` | mean height | mean of per-building roof heights | Typical building height. |
| `h_rms` | height std (σ_H) | std of roof heights | Height variability. |
| `h_skew` | height skewness | 3rd standardized moment | Asymmetry of the height distribution. |
| `h_kurt` | height (excess) kurtosis | 4th standardized moment, Fisher | Heavy tails / a few very tall buildings. |
| `gamma_m` | **alignedness** | mean over rows of the longest open streamwise run / `Nx` | How long and open the along-wind streets are. |
| `h_max` | max height | tallest building | Extreme height. |

### The alignedness family (`morphology.ALIGNEDNESS_KEYS`, Lu et al. 2023)

Different ways to quantify how "channelized" the flow paths are along the wind.
`gamma_m*` and `gamma_s` are canyon **aspect ratios** (length/height, `C/H`),
hence a much larger numeric scale than the `[0,1]` ratios.

| Descriptor | What it measures |
|---|---|
| `gamma_m` | Mean longest open streamwise run per row (includes fully-open streets). |
| `gamma_m_star` | Same, but penetrating (never-blocked) streets are credited the layout's strongest sheltered-canyon ratio, keeping it finite. |
| `gamma_s` | Sheltering aspect ratio `C/H` over building-bounded canyons only. |
| `gamma_p` | "Principal" alignedness: mean of the top `lambda_p` fraction of the per-row runs. |
| `gamma_c` | Mean longest run counting only non-penetrating (building-bounded) canyons. |

**OOD tail notation** in the results (e.g. `λp↑`, `γ↓`) means "the 95th-percentile
tail of that descriptor" — held-out layouts with unusually high/low packing,
alignedness, etc. Used to probe generalization.

---

## 8. Flow structures (used by the physics-oriented metrics)

Beyond aggregate error, `metrics.physics_metrics` scores where each model fails,
in regions a fluids engineer cares about:

| Structure | Meaning | How it's found (`metrics.region_masks`) |
|---|---|---|
| **Wake** | The sheltered, low-speed region *downstream* of a building. | Fluid cells within `WAKE_D = 6` cells downstream (`+x`) of a solid cell. |
| **(Street) canyon** | A fluid corridor flanked by buildings on both spanwise sides. | Fluid cells with solid cells within `CAN_D = 4` on both `±y` sides. |
| **Velocity deficit** | Regions where the flow is much slower than reference. | Fluid cells with target `u/u_ref < 0.5`. |
| **Low- / high-speed area** | Fraction of the plane below/above a speed threshold. | Thresholds `LOW_THR = 0.5`, `HI_THR = 1.5`. |

These are the "physical closure quantities" the surrogate is judged on; they are
defined by the physics, and reported separately from the ML fit metrics.

---

## 9. ML building blocks used here

Fluids readers can skim this; ML readers can skim §1–§8 and read this.

| Block | Where | One-line description |
|---|---|---|
| **Set encoder / permutation invariance** | `models/field.py: TokenEncoder` | A Transformer encoder over building tokens with no positional order → invariant to token permutation (à la Deep Sets / Set Transformer). |
| **Query→building cross-attention** | `models/field.py: RelCrossAttention` | Each output coordinate attends to the buildings, with a **relative-geometry bias** (anisotropic streamwise/spanwise + an upstream/downstream linear term). Conditioning like a Perceiver IO decoder. |
| **Coordinate decoding / neural field** | `models/field.py: UFFieldDecoder` | The decoder is a function of the query `(x, y)`, so the field can be sampled anywhere at any resolution (NeRF/SIREN-style). |
| **Fourier features** | `FourierFeatures` (field & pooled) | Lift `(x, y)` through random sinusoids so an MLP can fit high-frequency structure, fighting spectral bias (Tancik et al. 2020). |
| **FiLM conditioning** | `models/pooled.py: FiLMBlock` | Feature-wise affine modulation of the decoder by a global geometry vector (Perez et al. 2018). WP2 only. |
| **Axial self-attention** | `models/axial.py: AxialSelfAttention` | Attention along rows then columns of the query grid — `O(Nx + Ny)` instead of `O(Nx·Ny)` — to make neighbouring predictions spatially coherent. The **load-bearing** lever of UF-F (and the subject of the [bug story](../README.md#a-bug-that-changed-the-story)). |
| **Masked / structural loss** | `losses.masked_field_loss` | Masked MSE (tail-weighted) + a finite-difference **gradient** term + a radial **spectral (PSD)** term, to punish blur and missing high-frequency energy. |

---

## 10. Metrics

| Metric | Meaning |
|---|---|
| **RMSE / MAE** | Standard error magnitudes over fluid cells. |
| **R²** | Fraction of target variance explained, pooled over all fluid cells of the test set (not averaged per case). `1.0` perfect, `0.0` = predicting the mean. |
| **rel-L2** | `‖pred − target‖₂ / ‖target‖₂` over fluid cells — a scale-free relative error. |
| **Spearman** | Rank correlation between predicted and true speeds; robust to monotone distortions. |
| **Robustness gap** | `core R² − mean OOD R²`. Small = degrades little out of distribution. (A *negative* gap can be a collapse artifact — see WP2 in `reports/RESULTS.md`.) |

---

## 11. One-screen translation table

| You see (fluids) | Read as (ML) |
|---|---|
| "predict the mid-canopy mean velocity field" | regress a 2-D scalar field on a fixed plane |
| "conditioned on building geometry" | input = a variable-length set of 5-D tokens (+ optional global vector) |
| "for any layout, no re-simulation" | one trained model; generalize across the input set distribution |
| "masked to fluid cells" | loss/metrics computed on a boolean mask; ignore building cells |
| "LBM ground truth" | expensive deterministic label oracle |
| "morphology descriptors" | hand-engineered global features (found redundant in WP4) |
| "wake / canyon / deficit RMSE" | error on physically-defined sub-regions of the output |
| "OOD morphology tails (λp↑, γ↓, …)" | held-out slices at the 95th percentile of a geometry statistic |

---

*Sources for the physical definitions: the descriptor and region code in
`urbanformer/`, and Lu et al. (2023) for the alignedness family. Where a quantity
is a modeling **convention** (e.g. crediting penetrating streets in `gamma_m*`,
the mid-plane at `z = h_m/2`, thresholds `0.5 / 1.5`), it is flagged as such above
and in `reports/RESULTS.md`.*
