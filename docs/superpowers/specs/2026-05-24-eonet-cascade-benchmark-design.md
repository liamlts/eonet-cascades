# EONET Natural-Hazard Cascade Benchmark — Design

**Date:** 2026-05-24
**Status:** Approved for implementation planning
**Author:** Liam Schmidt

## 1. Project Goal

Build a portfolio-grade, methodologically rigorous benchmark suite for spatio-temporal point process models on multi-catalog natural-hazard cascade data. Primary outcomes:

- A polished public repository and writeup demonstrating modern ML on a real-world dataset.
- A defensible classical baseline (parametric multivariate Hawkes) alongside two neural variants (continuous-time LSTM and transformer).
- Extracted interpretable cross-mark triggering structure — the "cascade graph" of natural hazards over CONUS + Mexico, 2000–present.
- An evolvable foundation that can later incorporate ERA5 climate reanalysis as exogenous covariates and become a useful predictive tool.

Goal priority: portfolio > methodological rigor > learning vehicle. Real-world utility (tool for journalists / NGOs / insurers) is a future extension, not a v1 target.

## 2. Scope

**Geographic.** CONUS + Mexico as the development region. Framework is region-agnostic; global is a configuration change, not a rewrite.

**Temporal.** 2000-01-01 to present. Native event timestamps preserved (no daily binning). Catalogs with day-precision events are kept as-is; the likelihood absorbs the uncertainty.

**Spatial granularity.** Continuous lat/lon in WGS84 for modeling. ~0.25° grid for visualization and future ERA5 alignment.

**Mark space.** Unified 12-category mark space, harmonized across catalogs:

`wildfire`, `severe_storm`, `tropical_cyclone`, `tornado`, `flood`, `earthquake`, `volcanic_eruption`, `landslide`, `drought`, `dust_haze`, `temperature_extreme`, `sea_lake_ice`.

The set is fixed for v1. Additions (e.g., `snow`, `manmade`) require an explicit schema-migration step and are out of scope here.

## 3. Data Layer

### 3.1 Source catalogs (v1)

Four high-quality sources to start. Pluggable for additions (HURDAT2, GDACS, Smithsonian GVP, etc.).

| Catalog | What it provides | API |
|---|---|---|
| **EONET (NASA)** | Meta-catalog, wide event-type coverage | REST, no key |
| **USGS ComCat** | Global earthquake catalog | FDSN / REST, no key |
| **NOAA Storm Events DB** | US severe storms, tornadoes, hail, wind | CSV bulk, no key |
| **NASA FIRMS (MODIS + VIIRS)** | Active fire detections | REST, free key |

### 3.2 Unified event schema

```
event_id            text         -- "{catalog}:{native_id}"
source_catalog      text
time_start          timestamp    -- UTC
time_end            timestamp    -- UTC, nullable
longitude           float64
latitude            float64
mark                text         -- harmonized category
magnitude           float64      -- nullable; catalog-native (M, FRP, EF, ...)
metadata            json         -- full original payload, never lossy
ingested_at         timestamp    -- UTC
dedup_group_id      text         -- nullable
```

### 3.3 Storage

**DuckDB + Parquet**, persisted on the external Seagate drive:

```
/Volumes/Seagate_Ext/eonet-cascades-data/
├── raw/{catalog}/year=YYYY/*.parquet  # preserved original ingestions
├── events.duckdb                       # harmonized table + indexes
└── manifests/{catalog}_state.json      # last successful fetch per catalog
```

Data path is configurable in YAML. If the drive is not mounted, ingestion and training commands fail loudly with a clear error; read-only commands (e.g. `--help`) still work.

### 3.4 Ingestion pipeline

Each catalog implements a common interface:

```python
class CatalogFetcher(Protocol):
    name: str
    def fetch(self, since: datetime, until: datetime) -> Iterable[RawEvent]: ...
    def harmonize(self, raw: RawEvent) -> Event: ...
```

The driver `scripts/ingest.py` is idempotent: reads `manifests/`, fetches only new windows, writes raw Parquet, harmonizes into the DuckDB table. Rate limits are respected per source (≤1 req/sec by default, configurable).

### 3.5 Cross-catalog deduplication

Same physical event can appear in multiple catalogs (a hurricane in EONET and in NOAA Storm Events). Post-ingest spatio-temporal clustering: same-mark events within a spatial threshold and temporal threshold are assigned a shared `dedup_group_id`. Defaults:

| Mark | Spatial threshold | Temporal threshold |
|---|---|---|
| `earthquake` | 5 km | 1 h |
| `volcanic_eruption` | 5 km | 24 h |
| `wildfire` | 25 km | 24 h |
| `tropical_cyclone` | 200 km | 12 h |
| `tornado`, `severe_storm` | 25 km | 6 h |
| `flood`, `landslide` | 50 km | 24 h |
| `drought`, `temperature_extreme`, `dust_haze`, `sea_lake_ice` | 100 km | 7 days |

This is a separate post-processing step so thresholds can be tuned without re-fetching.

## 4. Model Layer

**Framework.** PyTorch 2.x.

### 4.1 Common interface

All tiers conform to:

```python
class PointProcessModel(Protocol):
    def log_intensity(self, t, x, k, history) -> Tensor: ...
    def integrated_intensity(self, history, window) -> Tensor: ...
    def sample(self, history, window) -> list[Event]: ...
```

Every tier optimizes the same likelihood:

$$\log L = \sum_i \log \lambda_{k_i}(t_i, x_i \mid H_{t_i}) - \int_{T \times X} \sum_k \lambda_k(t, x \mid H_t)\, dt\, dx$$

The integral term is approximated via Ogata thinning. Identical math across tiers; only the parameterization of $\lambda$ differs.

### 4.2 Tier 0 — Parametric multivariate Hawkes

$$\lambda_k(t, x \mid H_t) = \mu_k(x) + \sum_{(t_j, x_j, k_j) \in H_t} \alpha_{k_j \to k}\, g_t(t - t_j;\, \beta_{k_j \to k})\, g_x(x - x_j;\, \sigma_{k_j \to k})$$

- $\mu_k(x)$: per-mark baseline on ~1° spatial grid.
- $\alpha$: K×K excitation matrix — **the cascade graph**, the central interpretable object.
- $g_t$: exponential decay (default) or power-law, learnable rate per pair.
- $g_x$: isotropic Gaussian (default) or exponential, learnable bandwidth per pair.

Fitting: EM (Veen & Schoenberg 2008) or direct MLE with L-BFGS. Implemented from scratch (NumPy + SciPy) — small enough to debug fully, valuable as a learning exercise. K=12 → ~150 parameters. Fast on CPU.

### 4.3 Tier 1 — Neural Hawkes (continuous-time LSTM)

Mei & Eisner 2017 architecture. Continuous-time LSTM maintains a hidden state $h(t)$ that decays between events and is updated at each event. Intensity decomposes:

$$\lambda(t, x, k \mid h(t)) = \lambda_t(t \mid h) \cdot p(k \mid h, t) \cdot p(x \mid h, t, k)$$

- $\lambda_t$: scalar temporal intensity, softplus of linear projection of $h(t)$.
- $p(k \mid h, t)$: softmax over marks.
- $p(x \mid h, t, k)$: **Mixture Density Network** over 2D space — 5–10 bivariate Gaussians with learned location, scale, rotation per component.

Reference implementation: adapt the Mei & Eisner torch code; swap in the spatial MDN head.

### 4.4 Tier 2 — Transformer Hawkes + spatial MDN

Zuo et al 2020 (THP) architecture. Self-attention over event history with trigonometric temporal positional encoding. Same intensity decomposition as Tier 1; history representation comes from transformer blocks rather than an LSTM. Attention weights serve as a per-event "who triggered me" view, complementary to Tier 0's global $\alpha$ matrix.

Reference implementation: adapt the THP repository; swap in the spatial MDN head.

### 4.5 Tier 3 (stretch) — Neural Spatio-Temporal Point Process

Chen 2021 NSTPP. Continuous-space intensity parameterized by neural ODE / normalizing flow. Cleanest treatment of spatial cascades but most complex. Likely needs cloud GPU. Deferred until tiers 0–2 are working.

### 4.6 Training discipline

- Adam (neural tiers); L-BFGS (Tier 0).
- Gradient clipping (max norm 1.0), dropout 0.1, weight decay 1e-4 on neural tiers.
- Sequence batching: ~7-day windows with hidden-state carryover across consecutive windows (truncated BPTT for Tier 1). Events are not double-counted; the likelihood integral is taken over each window once and history conditioning crosses boundaries via the carried hidden state.
- Time-based split: train 2000–2020, val 2021–2023, test 2024–present.
- Early stopping on validation NLL.

## 5. Evaluation

### 5.1 Splits

Time-based only. Train 2000–2020, val 2021–2023, test 2024–present (touched once, at the end). No spatial holdout in v1 — the model is supposed to learn spatial structure; regional cross-validation is a future ablation.

### 5.2 Primary metric

**Held-out negative log-likelihood per event.** Apples-to-apples across tiers. If the classical baseline beats neural variants, the underlying process is Hawkes-like and that itself is a defensible result. If neural variants win, they captured structure the parametric kernel could not. Either outcome is publishable.

### 5.3 Secondary metrics

| Metric | What it tests | How |
|---|---|---|
| Time-to-next-event RMSE | Temporal calibration | Predict next event time per held-out sequence |
| Mark top-1 / top-3 accuracy | Type cascade structure | Predict next event type from history |
| Spatial LL at true location | Spatial calibration | Evaluate $\log p(x \mid h, t, k)$ at actual coords |
| Energy distance to held-out empirical | Spatial distribution quality | N forward simulations vs. actual held-out events |

### 5.4 Cascade reconstruction task

A model-comparison task tailored to the cascade question. Hold out a documented cascade (e.g., Hurricane Ida 2021 → Northeast flooding → landslides). Feed initial event(s) as conditioning history. Sample N=1000 forward trajectories. Score whether actual downstream events fall in the high-density regions of the predicted space-time intensity. This is the explicit "did you learn cascades?" test, distinct from global likelihood.

### 5.5 Synthetic sanity check (critical gate)

Before training on real data, generate synthetic cascades from a hand-designed multivariate Hawkes process with known $\alpha$, $\beta$, $\sigma$. Each tier must demonstrate recovery:

- **Tier 0:** mean relative error on $\alpha$ entries < 5%, on $\beta$ entries < 10%, on $\sigma$ entries < 10%. Sparsity pattern of $\alpha$ recovered exactly (true zeros stay near zero under thresholding).
- **Tiers 1–2:** aggregated attribution recovers the qualitative cascade graph (sparsity pattern + relative ordering of pair strengths). Quantitative parameter recovery is not expected.

A tier that cannot pass synthetic recovery is broken and is not advanced to real data.

### 5.6 Interpretability outputs

- **Tier 0:** the learned K×K $\alpha$ matrix rendered as a heatmap; per-pair time and space kernel parameters as a structured table.
- **Tiers 1 / 2:** attention rollouts (Tier 2) and gradient × hidden state attribution (both tiers) to produce a per-event "who likely triggered me" view. Aggregated across many events, this recovers an *empirical* cascade graph comparable to Tier 0's parametric one. Agreement = robustness signal; disagreement triggers investigation.

### 5.7 Uncertainty quantification

- Block bootstrap on the test set (year-long blocks) for confidence intervals on all metrics.
- 3–5 seeds for neural-tier training. Report mean ± std.

### 5.8 Visualization deliverables (headline outputs)

1. CONUS map of intensity for a chosen mark on a chosen day (one panel per model).
2. Excitation-matrix heatmap (Tier 0) + neural-tier empirical equivalents.
3. Per-pair temporal kernel plots (decay shapes).
4. Per-pair spatial kernel plots (decay over distance).
5. Hurricane Ida 2021 case study: actual event chain overlaid with model's predicted intensity and most-likely "parents" per event.

## 6. Architecture

**Repo location:** `~/Projects/eonet-cascades/`.

### 6.1 Layout

```
eonet-cascades/
├── pyproject.toml              # uv-locked deps; ruff + pytest + mypy config
├── README.md                   # what it is, how to run, headline result
├── configs/                    # YAML configs (no hardcoded params)
│   ├── data/{conus,global}.yaml
│   └── model/{hawkes,neural_hawkes,thp}.yaml
├── src/eonet_cascades/
│   ├── data/
│   │   ├── base.py             # CatalogFetcher protocol + Event schema
│   │   ├── eonet.py
│   │   ├── usgs.py
│   │   ├── noaa_storms.py
│   │   ├── firms.py
│   │   ├── harmonize.py        # mark mapping, dedup clustering
│   │   └── store.py            # DuckDB read/write layer
│   ├── models/
│   │   ├── base.py             # PointProcessModel ABC
│   │   ├── hawkes.py           # Tier 0
│   │   ├── neural_hawkes.py    # Tier 1
│   │   ├── thp.py              # Tier 2
│   │   └── components/         # shared spatial-MDN head, time encoders
│   ├── training/
│   │   ├── loop.py             # model-agnostic train loop
│   │   ├── sampler.py          # sequence batching with time-window splits
│   │   └── thinning.py         # Ogata thinning
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── cascade_recon.py    # case-study harness
│   │   ├── synthetic.py        # ground-truth generator + recovery
│   │   └── bootstrap.py
│   ├── interpret/
│   │   ├── excitation.py       # Tier 0 α extraction + plotting
│   │   ├── attention.py        # Tier 2 attention rollouts
│   │   └── attribution.py      # per-event attribution for neural tiers
│   ├── viz/
│   │   ├── intensity_map.py    # cartopy + matplotlib
│   │   ├── kernels.py
│   │   └── cascade_graph.py
│   └── cli.py                  # Typer entry point
├── scripts/
│   ├── ingest.py
│   ├── train.py
│   ├── eval.py
│   └── case_study.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_hawkes_baseline.ipynb
│   ├── 03_neural_models.ipynb
│   └── 99_paper_figures.ipynb
├── tests/                      # pytest + hypothesis
└── data/                       # gitignored; symlink to Seagate_Ext path
```

### 6.2 Module dependency direction

Strict layering, no cycles:

```
cli → scripts → {training, eval, interpret, viz} → {models, data} → schema
```

`models/` depends only on the event schema. `training/` and `eval/` depend on `models/` exclusively through the `PointProcessModel` ABC — never on concrete classes. This is what makes the benchmark a suite: swapping a model is a config change.

### 6.3 Configuration

All hyperparameters in YAML files under `configs/`. Resolved via **pydantic-settings** for type safety. A run is uniquely identified by `(data_config_path, model_config_path, seed)`. Configs are versioned in git; trained checkpoints reference their config hash.

### 6.4 Surface

**CLI (Typer):**

```bash
eonet ingest --catalogs eonet,usgs,noaa,firms --since 2000-01-01
eonet train --config configs/model/hawkes.yaml --seed 0
eonet eval --run-id <hash> --split test
eonet case-study --event "Hurricane Ida 2021" --run-id <hash>
```

**Notebooks** are for exploration, sanity checking, and paper figures. They import from `src/`; they do not define the science.

### 6.5 Dependencies

| Layer | Library |
|---|---|
| Package mgmt | uv (fast, locked) |
| ML | PyTorch 2.x |
| Data | DuckDB, polars, pyarrow |
| Geo | shapely, pyproj, cartopy (viz only) |
| Config | pydantic, pydantic-settings, PyYAML |
| CLI | Typer + rich |
| Plotting | matplotlib + scienceplots |
| Test | pytest, hypothesis |
| Lint | ruff |
| Notebooks | jupyterlab |

No PyMC, no Stan, no PyTorch Lightning in v1. Bayesian inference is a future direction.

### 6.6 Infrastructure

- **Local dev:** Mac. Tier 0 runs CPU-only in minutes. Tiers 1–2 use PyTorch MPS (Metal backend) — hours not days at CONUS scale.
- **Cloud GPU:** only if Tier 3 stretch is attempted (Lambda Labs / RunPod / Modal). Avoided in v1.
- **Storage:** all event data on `/Volumes/Seagate_Ext/eonet-cascades-data/`. Internal SSD holds only code, configs, and small artifacts. The repo `data/` directory is a symlink to the Seagate path; `.gitignore` excludes it. Configurable so a different drive or laptop-only mode works without code changes.

### 6.7 Reproducibility

- All randomness seeded (Python, NumPy, PyTorch CPU/MPS/CUDA).
- `uv.lock` committed.
- Each training run writes `runs/{model}_{config_hash}_{seed}/` containing: resolved config snapshot, code git hash, environment freeze, training/val curves, final checkpoint, evaluation metrics.
- `make headline` regenerates the README's headline figure from a clean checkout. Litmus test of portfolio-grade reproducibility.

### 6.8 Testing

- **Unit:** harmonizer (catalog → unified schema), thinning algorithm (regenerates known Poisson moments), MDN log-density (matches scipy on simple cases), Hawkes EM (recovers synthetic ground truth within tolerance).
- **Property (hypothesis):** schema invariants (timestamps UTC, coords in valid ranges, mark in vocab).
- **Integration (slow tier, off by default):** end-to-end pipeline on a 1-year mini-CONUS slice. NLL must beat a uniform-Poisson baseline.

## 7. Milestones

Phase estimates in **focused 4-hour work sessions**; calendar pace varies and may be substantially compressed.

| Phase | Goal | Sessions | Gate |
|---|---|---|---|
| 0. Bootstrap | Skeletal repo runs | 1 | `eonet --help` returns; CI green |
| 1. Data layer | All 4 catalogs ingested + harmonized + deduped | 3–4 | **Dataset smell test** — counts / maps / timeseries pass manual review |
| 2. Tier 0 + synthetic | Parametric Hawkes works | 3 | **Synthetic recovery** within 5% on hand-designed ground truth |
| 3. Tier 1 | Neural Hawkes + spatial MDN | 3–4 | Val NLL no worse than Tier 0 by more than 10% (relative); ideally better |
| 4. Tier 2 | Transformer Hawkes + spatial MDN | 3–4 | Val NLL no worse than Tier 1 |
| 5. Eval + interpret + case study | Cross-tier comparison + Hurricane Ida case study | 3 | **Cascade reconstruction** — Ida downstream events in high-intensity regions |
| 6. Writeup | README, paper figures notebook, blog draft | 2 | Clean clone → `make headline` → headline figure |

**Total v1:** ~18–22 sessions.

**Stretch goals (not part of v1 scope):**

- Tier 3 NSTPP — ~5 sessions, requires cloud GPU.
- ERA5 exogenous covariates — ~5 sessions, real data engineering. Bumps the project from "events trigger events" to "events trigger events given climate state."

**Gates summary.** Four hard gates: dataset smell test (post-Phase 1), synthetic recovery (mid-Phase 2), per-tier NLL sanity (Phases 3–4), cascade reconstruction (Phase 5). Each is a real "fix or proceed" checkpoint, not a checkbox.

## 8. Out of Scope (v1)

Explicitly deferred to keep v1 focused:

- Global geographic coverage (CONUS + Mexico only).
- ERA5 / climate-reanalysis exogenous covariates.
- Tier 3 NSTPP.
- Bayesian inference (uncertainty is via bootstrap + seeds, not posterior sampling).
- A public web dashboard or hosted service.
- Real-time / streaming ingestion (batch only).
- Spatial cross-validation across regions.
