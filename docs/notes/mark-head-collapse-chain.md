# The mark-head collapse chain — four runs, one diagnosis

**Date:** 2026-05-28 (closing note for the H1–H4 experiment series)
**Total cost:** ~$19 across four Lambda Labs A10 cloud runs
**Total wall time:** ~5 sessions over ~3 days

This document consolidates the four-run experiment chain into a single
source of truth. Each run has its own detail note (linked below);
this note is what you read first.

## TL;DR

We set out to fit a Neural Hawkes model on the NASA EONET hazard
catalog (2.4M events, 8 mark types, 2000–present) and recover a K×K
cross-mark cascade matrix. The model fits. The cascade matrix has a
specific extraction pathology: forward-simulated transitions from a
trained Neural Hawkes are **row-degenerate** — the per-mark
conditional distribution given history is the same across all parent
marks, regardless of which parent fired.

Across four targeted interventions (class rebalancing, MLP head,
auxiliary cross-entropy loss, plus the original baseline), the
pathology proved robust. Each intervention changed *something* about
the mark head's output (the marginal it produces, its parameter
count, its training objective) but none made the output
*parent-conditional*.

The diagnosis converged on an **encoder bottleneck**: the LSTM hidden
state `h(t)` does not encode per-parent variation at the granularity
the mark head's linear projections can extract. The cascade
information *is* present in `h(t)` — gradient attribution recovers
it cleanly — but at a representational rank that the mark head's
linear functions cannot separate into per-parent categorical outputs.

This is a clean methodological finding rather than a model failure
in the practical sense. **Tier 1-MLP is a working model** for the
core prediction task (val NLL/event 3.38, 20% improvement over the
parametric baseline). The cascade graph it implicitly encodes is
recoverable through gradient attribution rather than forward
simulation. The cross-view interpretability triangulation that
identified the gap — comparing the two derived K×K matrices to each
other and to the parametric baseline — generalizes beyond Hawkes
models.

## The four runs

| run | spec | mark head | training | val NLL | probe row-dev | wildfire marginal | result doc |
|-----|------|-----------|----------|---------|---------------|-------------------|-----------|
| **Tier 1** (baseline) | [original](../superpowers/specs/2026-05-25-tier-1-neural-hawkes-design.md) | Linear | joint Hawkes NLL | **4.20** | 0.0012 | 0.90 | (in `runs/tier1/20260525_162056/`) |
| **Tier 1.5** | [spec](../superpowers/specs/2026-05-26-tier1-5-class-rebalance-design.md)¹ | Linear | + rebalance + stratified subsample | 6.80 (+62%) | 0.0000 | 0.19 | [tier1_5-result.md](tier1_5-result.md) |
| **Tier 1-MLP** | [spec](../superpowers/specs/2026-05-26-tier1-mlp-mark-head-design.md) | **MLP** (64→32→K ReLU) | joint Hawkes NLL | **3.38** (−20%, best) | 0.0000 | 0.94 | [tier1_mlp-result.md](tier1_mlp-result.md) |
| **Tier 1-aux** | [spec](../superpowers/specs/2026-05-26-tier1-aux-mark-loss-design.md) | MLP | + aux CE loss λ=1.0 | 4.94 (+18%) | 0.0002 | 0.27 | [tier1_aux-result.md](tier1_aux-result.md) |

¹ Tier 1.5 didn't have its own brainstorm-spec because it was scoped
during the same session as the original collapse-diagnosis probe;
its design is captured in `tier1_5-result.md` and the runbook.

All four runs used the same data splits (2.385M train events
2022-01-01 to 2024-06-30, 356k val 2024-07-01 to 2024-12-31), same
hyperparameters except for the per-run intervention, same
acceptance criteria. The acceptance bar in each case was:

| criterion | threshold |
|-----------|-----------|
| (P) Forward-sim probe max row-deviation across parents | > 0.1 |
| (S) Val NLL/event | ≤ 4.41 (within 5% of Tier 1's 4.20) |

The cleanest result of any single run is **probe row-deviation
< 0.001** — three orders of magnitude below the > 0.1 threshold.

## What the probe shows (across runs)

The forward-simulation probe runs `scripts/probe_forward_sim.py`
against each checkpoint. For each parent mark, it computes the
row-normalized intensity `λ_k(h_post_parent) / Σ_l λ_l(h_post_parent)`
— the categorical distribution the forward simulator's multinomial
samples from at the first step after the parent. The K×K table of
rows-by-parent should differ across parents if the model conditions
on parent mark; it should be constant if it does not.

The probe also computes the same matrix on a warm start (50 real
historical events from the val slice + the parent event) to test
whether parent-conditioning emerges with richer history.

In all four runs, the matrix is row-degenerate in both seed
configurations. The aux loss (Tier 1-aux) is the most informative:
it dramatically reshaped the output marginal from ~94% wildfire to
nearly-uniform across the top four marks — proving the mark head
has the representational capacity to produce non-trivial output
distributions — but it kept those reshaped outputs identical across
parents.

## What each intervention ruled out

The four runs are a clean ablation that successively eliminates
hypotheses about the failure mechanism:

| run | intervention | hypothesis tested | hypothesis status |
|-----|-------------|-------------------|-------------------|
| Tier 1 | none (baseline) | — | establishes the pathology |
| Tier 1.5 | inverse-sqrt class weights + stratified subsample to force rare marks into training | H_imbalance: class imbalance creates a marginal-prediction local minimum | **REFUTED** — the rebalance flattens the marginal but does not break row-degeneracy; in fact tightens it |
| Tier 1-MLP | replace linear mark head with `Linear → ReLU → Linear` MLP (4.6× the parameter count) | H_capacity: linear head's K independent functions of `h` are rank-collapsed into a single direction | **REFUTED** — the MLP collapses identically and achieves better likelihood, demonstrating the head's representational capacity is not the bottleneck |
| Tier 1-aux | add explicit cross-entropy auxiliary loss on `softmax(z)` against observed marks | H_gradient: joint Hawkes NLL provides insufficient gradient signal on the relative magnitudes of `z`; the head needs an explicit composition signal | **REFUTED** — the aux loss successfully reshapes the marginal (dramatically) but does NOT make outputs parent-conditional; the rank-1 row-degeneracy survives |

After Tier 1-aux, the remaining hypothesis with strongest evidence
is the **encoder bottleneck**:

> **H6:** The LSTM hidden state `h(t)` does not encode per-parent
> distinctions at the representational rank the mark head's linear
> projections can extract. The information IS in `h` at some level
> (gradient attribution through `h` recovers cascade structure) but
> not at the granularity that linear functions of `h` can separate
> into per-mark categorical distributions.

This hypothesis is supported by *exclusion* (all mark-head-targeted
interventions fail) AND by *direct evidence*: gradient attribution
on the same Tier 1 / Tier 1-MLP checkpoints does recover the expected
off-diagonal cascade structure (severe_storm self-excitation,
earthquake→wildfire, etc.) — proving the information is in `h`, just
not at the rank the mark head can extract.

## The diagnostic methodology

The pattern that produced this diagnosis is not Hawkes-specific. It
generalizes to any model where two derived interpretability views
should agree at infinite data:

1. **Two K×K views** of the same model that should produce the same
   K×K cascade graph (here: gradient attribution through `h`, and
   forward simulation through the full intensity head).
2. **The views disagree** at finite training data. This is the
   diagnostic signal.
3. **Successive targeted interventions** at the obvious failing
   component (the mark head) narrow the diagnosis through which
   interventions DO change behavior versus which do not.
4. **The intervention that changes the most without solving the
   problem** is the most informative — Tier 1-aux here, because it
   shows the failing component has the capacity to behave
   differently but cannot be steered to a parent-conditional
   solution.

A practitioner who looked at only one view (forward-sim alone) would
have concluded "the model has no cascade structure" — wrong; it does,
recoverable through the other view. A practitioner who looked at only
the other view (gradient attribution alone) would have concluded "the
model works as expected" — also incomplete, because the model's
*generative* behavior (sampling from the mark head's output) is
broken in a non-obvious way.

The triangulation requires having BOTH views and noticing they
disagree, then doing the targeted intervention dance. To the best of
our reading of the Neural Hawkes literature (Mei & Eisner 2017,
EasyTPP benchmark, the RMTPP family), this kind of
multi-view-disagreement-as-diagnostic is not standard practice in
the field.

## What we actually have

**Tier 1-MLP at `runs/tier1_mlp/20260526_141553/checkpoint_best.pt`
is a working Neural Hawkes model** for the original prediction task:

- **Val NLL/event 3.38** — the lowest of any tier; 20% improvement
  over the parametric Tier 1 baseline.
- **Correctly tracks total event rate** as a function of history. The
  wildfire-channel intensity goes 40 → 76 between cold-start and
  50-event warm-start probes, demonstrating the LSTM is reading
  history for rate prediction.
- **Correctly predicts per-mark spatial distributions** via the MDN.
- **Cascade structure is recoverable** from this same checkpoint via
  the gradient-attribution path (`scripts/run_task13_v2.py` and the
  cross-tier notebook), showing the expected off-diagonal structure
  including wildfire self-excitation, earthquake→wildfire, and
  severe_storm→severe_storm.

What no run delivers is a **forward-simulation derived K×K cascade
graph that conditions on parent mark**. This is the gap that
characterizes the project's contribution: the gap is not in the
model's predictive power, not in the mark head's representational
capacity, not in the training signal — it is in the LSTM's encoding
of per-parent variation at the granularity the mark head reads.

## Open follow-ups

| ID | item | scope | priority |
|----|------|-------|----------|
| H2 | Test mark-agnostic spatial head — drop `mark_emb` from the MDN input. Hypothesis: the MDN's `(h, mark_emb)` conditioning absorbs the per-mark variation that would otherwise force the LSTM to encode it. Tests whether the encoder bottleneck is an artifact of the MDN providing a back-channel for mark info. | ~2 hr code + ~$5 cloud | LOW — would only succeed if a specific implementation detail is the cause; H6's encoder-bottleneck story stands even if H2 succeeds (it just means the bottleneck has an architecturally-fixable shape) |
| WRITE | Manuscript outline + venue decision | ~1-2 hr | MEDIUM — gates everything downstream |
| FIG | Re-render cross-tier notebook `03_tier0_vs_tier1.ipynb` to incorporate Tier 1-MLP + Tier 1-aux checkpoints alongside the original Tier 1 | ~1-2 hr | MEDIUM — needed for any submission |
| TASK#34 | Persist `mark_rebalance` / `stratify_train` flags in checkpoint config dicts (pre-existing gap surfaced by H4 review) | ~10 min | LOW |

## Where the runs live

All four checkpoints are on the local Mac under `runs/` (gitignored
as binary artifacts). They are:

- `runs/tier1/20260525_162056/` — Tier 1 baseline (cloud-trained
  2026-05-25 on Lambda Labs A10)
- `runs/tier1_5/20260526_043203/` — Tier 1.5 rebalance
- `runs/tier1_mlp/20260526_141553/` — Tier 1-MLP **(best model
  for downstream use)**
- `runs/tier1_aux/20260527_224337/` — Tier 1-aux

Each directory contains `checkpoint_best.pt`, `checkpoint_final.pt`,
`train_curves.csv`, and (for the H4 run) `train_tier1_aux.log`.

The forward-sim probe at `scripts/probe_forward_sim.py` reads
`mark_head` and `aux_lambda` from the checkpoint's config dict and
works across all four. Point `RUN_DIR` at any of them to reproduce
the row-degeneracy verdict.

## Pivot

This is the closing note for the H1–H4 experiment series. The next
work item is the manuscript / writeup, not another cloud run.
