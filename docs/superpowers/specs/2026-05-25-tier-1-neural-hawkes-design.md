# Tier 1 — Neural Hawkes (CTLSTM + MDN) Design

**Date:** 2026-05-25
**Status:** Approved for implementation planning
**Author:** Liam Schmidt
**Depends on:** Plan 3 (vectorized Tier 0 + L1 + NOAA registry fix) landing first

## 1. Goal

Implement the Tier 1 spatio-temporal Neural Hawkes (Mei & Eisner 2017 continuous-time LSTM + spatial Mixture Density Network head) from scratch, validate against synthetic data and a maintained reference library, train on real data via cloud GPU, and produce a head-to-head cascade-graph comparison against Tier 0.

This is the second model tier in the benchmark suite defined in [`2026-05-24-eonet-cascade-benchmark-design.md`](2026-05-24-eonet-cascade-benchmark-design.md) §4.3. Where Tier 0 gives a fully interpretable parametric cascade structure, Tier 1 trades parametric simplicity for the expressivity of a deep continuous-time model, and recovers an interpretable cascade graph via post-hoc gradient attribution and forward-simulation.

**Goal priority** (from spec §1): portfolio-grade > methodological rigor > learning vehicle. Real-world utility is still a future extension.

## 2. Implementation strategy

**Write the model from scratch in modern PyTorch**, using Mei & Eisner's published code at [`github.com/HMEIatJHU/neurawkes`](https://github.com/HMEIatJHU/neurawkes) as an architectural and mathematical sanity check (not a port — the original code is Python-2 era and not idiomatic), and benchmark our trained model's NLL against a maintained TPP library (e.g. `easy-temporal-point-process` or `tpps`) on the same data to confirm correctness.

This combination — *write from scratch + cross-reference the canonical implementation + benchmark against a third party* — directly serves the portfolio goal: shows deep understanding (from-scratch implementation), methodological rigor (validation against published code), and external verification (library benchmark).

## 3. Architecture

### 3.1 Continuous-time LSTM (CTLSTM) cell

Mei & Eisner 2017 §3. Hidden state $h(t) \in \mathbb{R}^{64}$ evolves continuously between events via per-dimension exponential decay, and is updated at each event via a standard LSTM gate cycle on the event embedding.

**Hidden dim:** 64.
**Event embedding:** `concat(mark_embedding[k_i], spatial_embedding(x_i))` → 32 dims.
- Mark embedding: learned, 16 dims, K=12 marks.
- Spatial embedding: 2-layer MLP $\mathbb{R}^2 \to \mathbb{R}^{16}$ with ReLU.

### 3.2 Intensity decomposition

$$\lambda(t, x, k | h(t)) = \lambda_t(t | h) \cdot p(k | h, t) \cdot p(x | h, t, k)$$

**Temporal intensity** $\lambda_t(t | h)$: scalar, $\text{softplus}(W_t \cdot h(t))$.

**Mark distribution** $p(k | h, t)$: softmax over K=12 marks from a linear head on $h(t)$.

**Spatial density** $p(x | h, t, k)$: **Mixture Density Network** producing K_mix = 8 **full-covariance bivariate Gaussian** components. Each component parameterized via mean (2) + Cholesky lower-triangular (3) + mixture weight = 6 params; total $8 \cdot 6 = 48$ output dims per call. **Shared head**: the MDN takes `concat(h(t), mark_embedding[k])` as input, conditioning on the mark via embedding rather than per-mark heads. Mixture weights from softmax over K_mix logits.

### 3.3 Total parameter count

~30,000 at K=12, hidden=64, K_mix=8. Comfortably small; trains in hours on a single GPU.

## 4. Data and training mechanics

### 4.1 Data scope (scale B)

- **Time window:** 2022-01-01 → 2024-12-31 (3 years, FIRMS era only — earlier years dominated by sparser catalogs).
- **Marks:** all 12 unified marks.
- **Event count:** ~5M events in scope.
- **Source:** existing DuckDB store at `/Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb`, no re-ingestion needed (Plan 3 Task 8 refreshes NOAA registry; Tier 1 training happens after that lands).

Scale-C (full historical 2000–present, A100 GPU, ~$50+) is **out of scope** for this design — explicitly deferred to a future plan if Tier 1 proves competitive at scale B.

### 4.2 Sequence batching

- **Chunking:** the event stream is split into **7-day windows**, each ~5,000–15,000 events at this scale.
- **Truncated BPTT:** gradient flows only through events in a single chunk. Hidden state at chunk end is detached and used to initialize the next chunk's hidden state. Standard pattern.
- **Batch size:** 1 chunk per step in v1 (variable-length sequences make wider batching messy and unnecessary at this data scale).
- **Shuffling:** chunks are processed in chronological order within an epoch (truncated BPTT requires it); epochs are re-shuffled within each epoch's training window only across non-overlapping chunks.

### 4.3 Train / val / test split (time-based)

| Split | Window |
|---|---|
| Train | 2022-01-01 → 2024-06-30 |
| Val | 2024-07-01 → 2024-12-31 |
| Test | 2025-01-01 → present (untouched until final eval) |

### 4.4 Likelihood integral

NLL formula reuses the spec §4.1 form. Two integral approximations:

- **Temporal integral** $\int_{t_i}^{t_{i+1}} \lambda_t(t | h(t)) dt$ between consecutive events: **Monte Carlo**, 20 uniform samples per inter-arrival interval. The CTLSTM hidden state is evaluated at each sample via its continuous-state update. Average × interval length.
- **Spatial integral** of the spatial Gaussian mass over the bbox: **Assumption A1** reused from Tier 0 — approximated as 1. Refined in Plan 5+ (Tier 3 NSTPP).

### 4.5 Optimizer and training loop

- **Optimizer:** AdamW, lr = 1e-3, weight_decay = 1e-4.
- **LR schedule:** cosine decay with 5% linear warmup.
- **Gradient clipping:** L2 norm 1.0.
- **Dropout:** 0.1 on the LSTM hidden state outputs only (not on the cell state).
- **Epochs:** target 10-20, early-stopping on val NLL with patience 3.
- **Reproducibility:** seeded Python / NumPy / PyTorch (CPU + CUDA) RNGs.

### 4.6 Checkpointing

Each training run writes `runs/tier1/{model}_{config_hash}_{seed}/`:
- `config.yaml` — resolved config snapshot
- `git_sha.txt` — code commit at training time
- `env.txt` — `uv pip freeze` output
- `train_curves.csv` — per-epoch train + val NLL, mark accuracy, time-to-event RMSE
- `checkpoint_best.pt` — model state at best val NLL
- `checkpoint_final.pt` — model state after the last epoch
- `attribution_matrix.csv`, `attribution_matrix.png` (post-training, see §5.A)
- `forward_sim_matrix.csv`, `forward_sim_matrix.png` (post-training, see §5.B)

## 5. Interpretability outputs

### 5.A Per-event attribution (primary cascade graph)

For each test-set event $i$ with $(t_i, x_i, k_i)$, compute the gradient of $\log \lambda_{k_i}(t_i, x_i | h(t_i))$ with respect to each prior event's contribution to the hidden state, weighted by a temporal-decay term:

$$A_{k_j \to k_i}\ \mathrel{+}=\ \| \nabla_{h_j} \log \lambda_{k_i}(t_i, x_i | h(t_i)) \|_1 \cdot e^{-(t_i - t_j)/\tau}$$

with a fixed decay $\tau = 7\ \text{days}$ (matches the typical 7-day chunk window so cascades are weighted by their freshness within the chunk).

Aggregate over all test-set events to produce a K × K **neural excitation matrix** comparable to Tier 0's α. Same axes convention (rows = parents, columns = children), same color scale and rendering — overlay-able with Tier 0's `alpha.png`.

### 5.B Forward-simulation transition counts (sanity check)

For each mark pair (parent, child), simulate 1,000 short forward trajectories seeded with a single event of mark *parent* via Ogata thinning, and count how often each child mark appears within a fixed temporal window. Produces a K × K transition-frequency matrix that should qualitatively agree with both Tier 0's α and the attribution matrix above.

**Three matrices in agreement → robust cascade signal. Disagreement is the interesting finding** and triggers investigation.

## 6. Evaluation

### 6.1 Primary metric

**Held-out NLL per event.** Identical metric to Tier 0, directly comparable. Headline question: does Tier 1's per-event NLL beat Tier 0's by a meaningful margin (>5% relative)?

### 6.2 Secondary metrics (same as Tier 0, spec §5.3)

- Time-to-next-event RMSE
- Mark top-1 / top-3 accuracy
- Spatial LL at true location
- Energy distance to held-out empirical spatial distribution

### 6.3 Synthetic recovery gate

Reuse Plan 2's synthetic Hawkes generator (`eonet_cascades.eval.synthetic.simulate_hawkes`). Train Tier 1 on the same synthetic data Tier 0's recovery test uses, then verify:

- Training NLL strictly decreases across epochs.
- The aggregated attribution matrix (§5.A) **qualitatively** matches the true α sparsity pattern (the non-zero entries are recovered, the zeros stay below threshold). Tier 1 has no α parameter, so this is the analog of Tier 0's quantitative recovery test.

**Gate criterion:** ≥ 70% of true non-zero α entries appear in the top quartile of the attribution matrix; ≤ 20% of true zero entries appear there.

### 6.4 Reference library cross-check

Fit the same training data with the **`easy-temporal-point-process` (EasyTPP)** library — actively maintained as of 2024, ships a CTLSTM implementation that matches Mei & Eisner 2017. If install or compatibility breaks, fall back to `tpps`. Our Tier 1's held-out NLL should be within **2% relative** of the reference library on the same data and same hyperparameters. Mismatch indicates an implementation bug.

### 6.5 Hurricane Ida 2021 case study

Same cascade-reconstruction test as Tier 0 (spec §5.4). Both Tier 0 and Tier 1 forward-simulate from Ida's initial track point in late August 2021; compare the spatial intensity maps over the following days against actual downstream flooding + landslide events.

### 6.6 Cross-tier comparison artifacts (headline deliverables)

1. **Side-by-side cascade-graph heatmaps:** Tier 0 α vs Tier 1 attribution vs Tier 1 forward-sim. Three panels.
2. **Tier 0 vs Tier 1 NLL** per held-out 7-day chunk (line plot over time).
3. **Tier 0 vs Tier 1 mark top-1 accuracy** (bar plot).
4. **Agreement / disagreement table:** for each (parent, child) pair, do Tier 0 and Tier 1 agree on the cascade signal? Which model is correct when they diverge?

## 7. Compute and deployment

### 7.1 Local dev (Intel Mac, CPU-only)

All scaffolding, synthetic recovery, unit tests, ablations on ≤10,000 events. Never run scale-B training locally — CPU training of the CTLSTM at 5M events for 10 epochs is multi-day.

### 7.2 Cloud GPU (Lambda Labs A10, budget ≤ $20)

Training workflow:

1. `git push` project to a private GitHub repo (set up at start of Plan 4 if not already).
2. Provision a Lambda Labs A10 instance (~$0.75/hr).
3. On the instance: `git clone`, `uv sync --extra ml`, then either `scp` the DuckDB snapshot (~200 MB) from the user's Mac, or run `eonet ingest` on the cloud fresh (slower, uses FIRMS API budget).
4. Launch training: `eonet model train-neural-hawkes --since 2022-01-01 --until 2024-06-30 ...`. Logs to a tail-able file. Checkpoints to `runs/tier1/{ts}/`.
5. When training finishes: `scp` the run dir back to the Mac.
6. Terminate the instance.

Estimated cost for scale B: ~15-20 wall-clock hours × $0.75/hr = **~$11-15**, under the $20 cap.

### 7.3 Reproducibility on the cloud

The instance is ephemeral. Everything that matters lives in the repo + the `runs/` directory `scp`'d back. The `config.yaml` + `git_sha.txt` + `env.txt` triple in each run dir is sufficient to reproduce the training.

## 8. Architecture summary (file layout)

```
src/eonet_cascades/
├── models/
│   ├── neural_hawkes.py     # CTLSTM cell, intensity heads, NeuralHawkes class
│   └── components/          # NEW package
│       ├── ctlstm.py        # CTLSTM cell — the math piece
│       ├── mdn_head.py      # MDN spatial head (shared, mark-conditioned)
│       └── embeddings.py    # mark + spatial embeddings
├── training/
│   ├── neural_loop.py       # NEW — training driver (NLL + MC integral + BPTT)
│   └── monte_carlo.py       # NEW — Monte Carlo integral helper
├── interpret/
│   ├── attribution.py       # NEW — per-event gradient attribution → K×K matrix
│   └── forward_sim_matrix.py # NEW — simulation-based transition matrix
└── cli.py                   # MODIFY — add `eonet model train-neural-hawkes`
```

## 9. Milestones

Same per-phase structure as Plan 1 + Plan 2, scaled to Tier 1's complexity.

| Phase | Goal | Sessions | Gate |
|---|---|---|---|
| 1 | Bootstrap module + CTLSTM cell + MDN head + intensity decomposition + forward pass | 2 | Forward pass returns finite tensors on synthetic input |
| 2 | NLL with Monte Carlo temporal integral; training loop on small synthetic data | 2 | Synthetic NLL strictly decreases over epochs |
| 3 | **GATE: synthetic cascade recovery** — train on Plan 2 synthetic data, verify attribution matrix qualitatively matches true α | 1-2 | Sparsity pattern recovered per §6.3 thresholds |
| 4 | Library cross-check (`easy-temporal-point-process` or `tpps`) | 1 | Our NLL within 2% of reference impl |
| 5 | CLI `eonet model train-neural-hawkes` + checkpoint loading + interpretability outputs | 1 | Local end-to-end run on synthetic data produces all artifacts |
| 6 | **Operational: cloud-GPU scale-B training run** | 1-2 days wall time | Trained checkpoint + Tier 0 vs Tier 1 comparison plots |
| 7 | Hurricane Ida case study + writeup notebook | 1-2 | Side-by-side cascade graphs + NLL table |

**Total v1:** ~8-12 focused sessions plus 1-2 days of cloud-GPU wall time.

## 10. Out of scope (deferred to Plan 5+)

- **Scale C** (full 2000-present + A100 GPU) — only if Tier 1 proves competitive on scale B and the user has appetite for ~$50+ in cloud spend.
- **Tier 2** (Transformer Hawkes, Zuo 2020 THP) — its own brainstorming after Tier 1 lands.
- **Tier 3** (NSTPP, Chen 2021) — stretch from spec §4.5.
- **Exact spatial integral over bbox** (Assumption A1 still in force).
- **Attention-rollout visualization** (Transformer-specific; not relevant to CTLSTM).
- **Wider batching across chunks** (variable-length-sequence batching machinery).
- **Hyperparameter sweep** beyond a single ablation pass.
