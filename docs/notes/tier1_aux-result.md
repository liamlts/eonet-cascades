# Tier 1 with auxiliary mark-classification loss — H4 refuted; collapse is in the LSTM, not the head

**Date:** 2026-05-28
**Checkpoint:** `runs/tier1_aux/20260527_224337/checkpoint_best.pt` (gitignored)
**Training log:** `runs/tier1_aux/20260527_224337/train_tier1_aux.log` (gitignored)
**Spec:** `docs/superpowers/specs/2026-05-26-tier1-aux-mark-loss-design.md`
**Implementation plan:** `docs/superpowers/plans/2026-05-26-tier1-aux-mark-loss.md`
**Cost:** ~$4 on Lambda Labs A10 (5h 20m wall, $0.75/hr)

## TL;DR

Added an explicit cross-entropy auxiliary loss on the mark head via the
`--aux-lambda 1.0` flag. Same architecture as Tier 1-MLP (MLP head),
same hyperparameters, same data splits. Both acceptance criteria
failed: forward-sim probe row-deviation is 0.0002 (Seed A) / 0.0000
(Seed B) against a > 0.1 threshold, and val NLL/event is 4.94 against
a ≤ 4.41 threshold.

But the run produced an unexpectedly informative result: **the aux
loss dramatically reshaped what marginal the mark head outputs (from
~94% wildfire-dominant in Tier 1-MLP to a nearly-uniform mix across
the top four marks in Tier 1-aux) without breaking the rank-1
collapse**. The mark head still outputs the SAME distribution across
all parent marks — just a different distribution than before.

This sharpens the diagnosis substantially. The pathology is not "the
mark head can't express enough output distributions" — it can, and
under aux loss it does. The pathology is that **the LSTM hidden
state `h(t)` does not distinguish histories that should produce
different marks** in a way the mark head's linear extractions can
separate.

## Acceptance criteria

| criterion | threshold | result | verdict |
|-----------|-----------|--------|---------|
| (P) Forward-sim probe max row-deviation | > 0.1 | **0.0002** (Seed A) / **0.0000** (Seed B) | **FAIL** |
| (S) Val NLL/event | ≤ 4.41 (within 5% of 4.20) | **4.940** (+18%) | **FAIL** |

Both criteria fail. Per the spec's decision table, this rules out H4
and the writeup pivots to the methodological negative-chain framing.

## What the probe shows

### Seed A (cold-start, single bbox-center parent event)

Row-normalized `λ_k / Σ_k λ_k` at t = 0.5 days, for all 7 parents:

```
                 dust_haze   earthquake     flood   landslide   severe_storm   tornado   wildfire
dust_haze           0.0003       0.2288    0.2323      0.0024         0.2604    0.0110     0.2648
earthquake          0.0003       0.2288    0.2323      0.0024         0.2604    0.0110     0.2648
flood               0.0003       0.2288    0.2323      0.0024         0.2604    0.0110     0.2648
landslide           0.0003       0.2288    0.2323      0.0024         0.2604    0.0110     0.2648
severe_storm        0.0003       0.2288    0.2323      0.0024         0.2604    0.0110     0.2648
tornado             0.0003       0.2288    0.2323      0.0024         0.2604    0.0110     0.2648
wildfire            0.0003       0.2288    0.2323      0.0024         0.2604    0.0110     0.2648
```

All 7 rows identical to 4 decimal places. Max deviation across
parents = 0.000026 at t = 10⁻⁴ days, 0.000030 at t ≥ 10⁻² days.

### Seed B (warm 50-event history)

Same result. All 7 rows identical, max deviation = 0.0000.

### What the marginal looks like across runs

| mark | Tier 1 | Tier 1.5 | Tier 1-MLP | **Tier 1-aux** |
|------|--------|----------|------------|----------------|
| wildfire | 0.904 | 0.194 | 0.939 | **0.265** |
| severe_storm | 0.048 | 0.116 | 0.034 | **0.260** |
| flood | 0.022 | 0.207 | 0.012 | **0.232** |
| earthquake | 0.019 | 0.316 | 0.013 | **0.229** |
| tornado | 0.007 | 0.012 | 0.003 | 0.011 |
| landslide | <0.001 | 0.043 | 0.000 | 0.002 |
| dust_haze | <0.001 | 0.017 | 0.000 | 0.000 |

The aux loss flattened the marginal dramatically (wildfire 94% → 27%,
severe_storm 3% → 26%, flood 1% → 23%, earthquake 1% → 23%).
Tornado, landslide, and dust_haze stay near zero — these marks are
rarely event-targets in the training data, so the cross-entropy
gradient barely touches them.

**The reshape DID happen.** What didn't happen is making this reshape
*parent-conditional*.

## Training trajectory

Convergence was clean and monotonic across all 15 epochs:

```
epoch  val_nll   Δ                vs Tier 1-MLP
  0    38.24    —                 +1%
  1    12.03   −26.21             +12%
  2     7.75   −4.29              +28%
  3     6.47   −1.28              +36%
  4     5.97   −0.49              +41%
  5     5.59   −0.39              +42%
  6     5.35   −0.23              +40%
  7     5.19   −0.16              +41%
  8     5.06   −0.13              +42%
  9     5.06   −0.00              +45%
 10     4.99   −0.07              +45%
 11     4.96   −0.02              +46%
 12     4.95   −0.01              +46%
 13     4.94   −0.01              +46%
 14     4.94   −0.00              +46%
```

Asymptote at val_nll ≈ 4.94 — paralleled Tier 1-MLP's curve about
40-46% higher. The aux loss paid measurable likelihood cost (~17%
over Tier 1's 4.20 baseline) without producing any composition
benefit.

## The four-run picture

| run | mark head | training objective | val NLL | probe row-dev | wildfire marginal |
|-----|-----------|--------------------|---------|---------------|-------------------|
| Tier 1 | Linear | joint Hawkes | 4.20 | 0.0012 | 0.90 |
| Tier 1.5 | Linear | + rebalance + stratified | 6.80 (+62%) | 0.0000 | 0.19 |
| Tier 1-MLP | **MLP** | joint Hawkes | **3.38** (−20%) | 0.0000 | 0.94 |
| **Tier 1-aux** | MLP | + aux CE λ=1.0 | 4.94 (+18%) | 0.0002 | 0.27 |

Four interventions targeting the mark head and its training signal.
Four failures of the primary acceptance criterion. The pathology is
robust across:

1. The training distribution (Tier 1.5 forced rare marks in)
2. The loss weighting (Tier 1.5 inverse-sqrt weights)
3. The head architecture (Tier 1-MLP)
4. The composition gradient signal (Tier 1-aux explicit CE)

## Sharpened diagnosis: it's not the mark head

The naive reading of the rank-1 collapse — that the mark head is
mathematically unable to produce per-parent outputs — is **directly
refuted by Tier 1-aux**. The aux loss steered the mark head's outputs
to a substantially flatter marginal, demonstrating that:

- The mark head has the representational capacity to produce
  non-wildfire-dominant outputs
- The training signal can push it in non-trivial directions
- The composition matrix `W_λ` has rank > 1 in the sense that it can
  produce many different output distributions in the K-simplex

What it cannot do — under any of the four interventions tried — is
make the SPECIFIC output a function of which parent fired most
recently.

The cleanest remaining hypothesis is **H6: the LSTM hidden state
`h(t)` representations do not distinguish histories that should
produce different marks**. The hidden state for a sequence ending in
a wildfire vs ending in an earthquake may be near-identical along the
directions the mark head reads from `h`. The information is
*present* in `h` at some level (gradient attribution recovers
cascade structure, as shown in the cross-tier notebook), but it is
not present in a form the mark head's linear functions can extract
as a per-mark categorical distribution.

This is a much more interesting finding than "the mark head is
broken." It suggests the bottleneck is the **encoder**, not the
**decoder** — and that the cascade information lives in the LSTM
hidden state at a representational granularity that is recoverable by
some interpretability methods (gradient attribution) but not by
others (forward-sim through the mark head).

## Hypothesis ledger after four runs

| H | hypothesis | status | evidence |
|---|-----------|--------|----------|
| H1 | Under-training | **REFUTED** | All runs asymptote in ≤ 15 epochs |
| H2 | MDN spatial head absorbs mark conditioning | not tested | next experiment if iterating |
| H3 | Linear mark head insufficient capacity | **REFUTED** | Tier 1-MLP shows MLP head behaves identically |
| H4 | Joint Hawkes NLL has weak composition gradient | **REFUTED** | Tier 1-aux's aux loss DID reshape the marginal but not the conditioning |
| **H6 (new)** | **LSTM hidden state representations don't encode per-parent distinctions at the granularity the mark head's linear projections can extract** | **SUPPORTED by exclusion** | All mark-head-targeted interventions fail; gradient attribution recovers cascade structure → info IS in `h` somewhere, just not at the rank the head sees |

## Methodological framing (writeup angle)

The cross-view interpretability triangulation has now produced an
unusually clean methodological result. The diagnostic pattern was:

1. **Observation:** forward-sim K×K matrix is row-degenerate;
   gradient-attribution K×K matrix shows cascade structure
2. **Hypothesis 1:** mark head's output is collapsed →
   try reshape it (Tier 1.5, 1-MLP, 1-aux)
3. **Result:** all three reshape attempts succeed at changing the
   output, but none make it parent-conditional
4. **Refined hypothesis:** the mark head's OUTPUTS are flexible; the
   problem is that h(t) doesn't contain the per-parent variation the
   head needs to express different outputs for different parents

This pattern — "two interpretability views designed to be equivalent
at infinite data, disagreeing at finite data, and the disagreement
narrows the diagnosis through successive interventions" — is a
methodological contribution distinct from any specific Hawkes
finding. The pattern works regardless of whether the model is a
Hawkes process; it relies only on having two derived K×K matrices
from a trained model that should agree, that don't.

## What we actually have

**Tier 1-MLP is a working Neural Hawkes model for the intended
prediction task.** Val NLL 3.38 is the lowest of any tier and is a
20% improvement over the original Tier 1 baseline. It predicts
overall event rates, spatial densities, and (via gradient
attribution) per-mark cascade structure correctly.

What none of the runs deliver is a **forward-sim derived K×K cascade
graph that conditions on parent mark**. This is a real gap in the
project's original scope — but it's a gap that's now characterized
precisely:

- The gap is not in the model's predictive power
- The gap is not in the mark head's representational capacity
- The gap is in the LSTM's encoding of per-parent variation at the
  granularity the mark head reads
- The gap is recoverable through gradient attribution (which doesn't
  go through the mark head's linear projection)

## Decision

Per the spec's decision table, P fails (P + S both fail) → H4 ruled
out; writeup pivots to the negative-chain framing.

Two reasonable paths from here, both viable:

- **(A) Stop iterating; write up the four-run chain as the
  methodological contribution.** Cross-view triangulation diagnosing
  a robust encoder-bottleneck pathology. Tier 1-MLP is the
  best-performing model for downstream use (gradient-attribution
  cascade structure works). ~Clean closing point.
- **(B) One more cheap experiment: H2 (mark-agnostic spatial head).**
  Tests whether the MDN is secretly absorbing parent-mark information
  via its `(h, mark_emb)` input. ~$5. If it succeeds, we get a
  working model. If it fails, H6 is even more strongly supported.

Both paths are defensible. (A) closes a story; (B) buys one more
chance at a working model at modest cost.
