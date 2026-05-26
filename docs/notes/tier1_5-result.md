# Tier 1.5 retrain result — class rebalance does NOT fix the mark-head collapse

**Date:** 2026-05-26
**Checkpoint:** `runs/tier1_5/20260526_043203/checkpoint_best.pt` (gitignored)
**Training log:** `runs/tier1_5/20260526_043203/train_tier1_5.log` (gitignored)
**Cost:** ~$5 on Lambda Labs A10 (5h 20m wall, $0.75/hr)

## TL;DR

Trained Tier 1 with `--mark-rebalance --stratify-train` on Lambda Labs A10, 15
epochs, otherwise identical hyperparameters to the original Tier 1 run. Result:
the forward-sim mark-head collapse documented in commit `420d5a3` survived
both interventions cleanly. Class imbalance is **not** the sole cause of the
collapse; the diagnosis from earlier this session is refuted.

## Acceptance criteria (from runbook commit `f541a68`)

| criterion | target | result | verdict |
|-----------|--------|--------|---------|
| (1) Forward-sim probe max row-deviation | > 0.1 | **0.0000** | **FAIL (regression)** |
| (2) Val NLL/event | within ~5% of 4.20 | 6.80 (+62%) | **FAIL** |

Both criteria failed.

## What the probe shows

Running `scripts/probe_forward_sim.py` against the Tier 1.5 checkpoint:

### Seed A (cold start, single bbox-center parent event)

Row-normalized λ_k / Σ_k λ_k at t = 0.5 days after the parent event, for all 8
parent marks:

```
                 dust_haze   earthquake   flood   landslide   severe_storm   tornado   volcanic_er   wildfire
dust_haze           0.0173       0.2724  0.2025      0.0592         0.1282    0.1272        0.0046     0.1885
earthquake          0.0173       0.2724  0.2025      0.0592         0.1282    0.1272        0.0046     0.1885
flood               0.0173       0.2724  0.2025      0.0592         0.1282    0.1272        0.0046     0.1885
landslide           0.0173       0.2724  0.2025      0.0592         0.1282    0.1272        0.0046     0.1885
severe_storm        0.0173       0.2724  0.2025      0.0592         0.1282    0.1272        0.0046     0.1885
tornado             0.0173       0.2724  0.2025      0.0592         0.1282    0.1272        0.0046     0.1885
volcanic_eruption   0.0173       0.2724  0.2025      0.0592         0.1282    0.1272        0.0046     0.1885
wildfire            0.0173       0.2724  0.2025      0.0592         0.1282    0.1272        0.0046     0.1885
```

All 8 rows are **numerically identical** — max deviation across parents = 0.0000.

### Seed B (warm, 50 real historical events + parent)

Same result. All 8 rows numerically identical. Max deviation across parents
= 0.0000.

### Time sweep on Seed A

| t (days) | total ǀrow − row_meanǀ | max ǀrow − row_meanǀ |
|---------:|----------------------:|--------------------:|
| 0.0001   | 0.000108              | 0.000011 |
| 0.0010   | 0.001829              | 0.000207 |
| 0.0100   | 0.000136              | 0.000014 |
| 0.1000   | 0.000000              | 0.000000 |
| 0.5000   | 0.000000              | 0.000000 |
| 1.0000   | 0.000000              | 0.000000 |
| 5.0000   | 0.000000              | 0.000000 |

For comparison, the original Tier 1 probe (commit `420d5a3`) showed:
- Seed A total dev = 0.0012
- Seed B total dev = 0.0001
- Max dev at t=10⁻⁴ = 0.0023

**Tier 1.5 has a tighter collapse, not a looser one.**

## What the model DID learn

The absolute intensity λ_k changes with history just like in Tier 1: the warm
seed gives different absolute λ values (wildfire channel ~40 cold → ~52 warm,
similar ratio to Tier 1). The temporal/intensity head reads h(t) and predicts
overall event rate.

The *marginal* the mark head locked onto is different from Tier 1:

| mark              | Tier 1 marginal | Tier 1.5 marginal |
|-------------------|-----------------|-------------------|
| wildfire          | 0.90            | 0.19              |
| earthquake        | 0.02            | 0.32 ← largest    |
| flood             | 0.02            | 0.21              |
| severe_storm      | 0.05            | 0.12              |
| tornado           | 0.01            | 0.13              |
| landslide         | <0.01           | 0.06              |
| dust_haze         | <0.01           | 0.02              |
| volcanic_eruption | (absent)        | <0.01             |

The rebalance did its job in one sense — it flattened the marginal away from
the wildfire-dominant baseline. But the marginal itself is still **constant
across parents**. The mark head still ignores h(t).

## What this means for the diagnosis

The earlier hypothesis (commit `420d5a3`): *"the softmax mark head has
collapsed to outputting the empirical marginal P(k) regardless of input —
a known failure mode of softmax classifiers on heavily imbalanced data."*

That hypothesis is now refuted. The pathology survives:
- Stratified subsampling that forces all rare marks into the training set
- Inverse-sqrt mark weights spanning 400× dynamic range
- 15 epochs of training under that rebalanced objective

The collapse must therefore be driven by something other than (or in addition
to) the class-imbalance gradient signal. Remaining hypotheses, in order of
ascending implementation cost:

1. **Under-training.** 15 epochs may not be enough to let the mark head
   learn off-diagonal structure. The training log shows the val NLL was still
   improving by ~0.01/epoch at epoch 14; a 50- or 100-epoch run would test
   this.
2. **MDN spatial head absorbs the mark conditioning.** When `p(x | h, k)`
   can express mark-specific spatial distributions, the joint NLL is
   minimized without the mark head having to do work. Test: train with a
   mark-agnostic spatial head (single MDN shared across all k) and see if
   the mark head wakes up.
3. **Linear mark head insufficient capacity.** Current `W_lambda_k h(t)` is
   a single linear layer (hidden_dim=64 → K=8). Replace with a small MLP
   (e.g., 64 → 32 → 8 with ReLU); see if non-linear features make the head
   condition on h.
4. **Auxiliary mark-classification loss** with a stronger gradient signal
   than the implicit one from the joint Hawkes NLL.

## Methodological observation (publishable framing)

The cross-view interpretability triangulation (gradient attribution vs.
forward-sim) identified a mark-head pathology in the Tier 1 Neural Hawkes
that is **robust to the standard class-imbalance remedy**. Inverse-sqrt
rebalancing combined with stratified subsampling changes *which* marginal
the head collapses to (from 90% wildfire to a flatter 8-class distribution)
but not the underlying mode-collapse to a constant function of the LSTM
hidden state.

This is arguably a stronger result than "we fixed it." It documents a
failure mode that:

- Is reachable from a vanilla Neural Hawkes implementation on imbalanced
  marked point process data
- Is *not* visible in val NLL alone — Tier 1 had val NLL 4.20 and still
  had this pathology
- Is *not* visible in forward-sim alone if you only look at one parent at
  a time — the row degeneracy is only obvious in the K×K view
- Is *not* visible in attribution alone — attribution operates on h directly,
  bypasses the broken mark head, so it shows healthy off-diagonal structure
- Survives the textbook fix

The diagnosis required *two* derived interpretability views, designed to be
equivalent at infinite data, and their disagreement at finite data.

## Next steps decision

Either:
- (a) Test hypothesis 1 (longer training, ~$15 cloud) — cheapest experiment,
  cleanest answer.
- (b) Test hypothesis 3 (MLP mark head, code + ~$5 cloud) — most likely fix.
- (c) Stop iterating on the model; write up the negative result as the
  primary contribution. The cross-view-triangulation-as-diagnostic story is
  complete and arguably more interesting with the failed rebalance attached.
