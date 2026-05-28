# Future work — what stems from the encoder-bottleneck finding

**Date:** 2026-05-28
**Context:** Closing memo for the eonet-cascades project. Reading order: start with [mark-head-collapse-chain.md](mark-head-collapse-chain.md) for the finding itself, then this doc for what's worth doing next.

The four-run negative chain ends with a precise diagnosis (LSTM hidden state h(t) does not encode per-parent variation at the rank the mark head reads) and a working model (Tier 1-MLP, val NLL 3.38). That opens several reasonable research directions. This memo lays them out, ranked by yield-per-effort and grouped by what they'd test or build.

## Tier 1 — direct continuation of the encoder-bottleneck investigation

These are experiments that would either (a) confirm the H6 diagnosis from a new angle, or (b) actually fix the bottleneck. All are scoped to fit on a single Lambda Labs A10 run (~$5, ~5 h).

### 1.1. Effective-rank diagnostic on `W_λ_k @ Cov(h)`

**Question:** can we quantify the encoder bottleneck directly, without running another model?

**Method:** Load each of the four checkpoints. Run a large batch of historical event sequences through `forward()` to get a sample of post-event hidden states `h ∈ ℝ^{N × hidden_dim}`. Compute the empirical covariance `Cov(h)` and the effective rank of `W_λ_k @ Cov(h)` (the matrix whose rank determines how many distinct output distributions the model can produce across input variations). If the effective rank is ≪ K = 7, this is a quantitative confirmation that the bottleneck is exactly where we've localized it.

**Cost:** 1-2 hours of code; no cloud. The probe script is already structured for this.

**Yield:** High. Confirms the diagnosis numerically and gives a publishable diagnostic that practitioners can apply to any marked-TPP model in ~minutes.

### 1.2. Wider LSTM hidden state (`hidden_dim ∈ {128, 256}`)

**Question:** if the bottleneck is representational rank in `h(t)`, does just enlarging `h` fix it?

**Method:** Re-run the Tier 1-MLP recipe with `--hidden-dim 128` and `--hidden-dim 256`. Otherwise identical. Run the probe.

**Cost:** ~$10 total cloud, ~3 hr code (no code change required beyond the existing CLI; just two more cloud runs).

**Yield:** Medium-high. If the row-degeneracy persists at hidden_dim=256, H6 is strongly confirmed and the bottleneck is structural (not just a too-small-`h` issue). If it breaks, H6 was about a specific capacity threshold and we can pin it.

### 1.3. Transformer Hawkes encoder (the original Tier 2 scope)

**Question:** is the encoder bottleneck specifically an LSTM problem or a deeper marked-TPP issue?

**Method:** Replace the CTLSTM with a Transformer encoder (self-attention over event embeddings + time-positional encodings). Keep the rest of the architecture identical: same MDN spatial head, same MLP mark head, same training data. Run the probe.

**Cost:** ~5-8 hr code (non-trivial — needs a continuous-time-aware attention mechanism or one of the off-the-shelf neural TPP transformer flavors), ~$10 cloud.

**Yield:** High. If the Transformer encoder also exhibits the row-degeneracy, the finding generalizes beyond LSTMs and is a much stronger paper. If the Transformer breaks the collapse, we have a working model AND localize the LSTM-specificity of the failure.

This is the most natural single-experiment follow-up if the H6 diagnosis is correct and the paper wants a stronger generality claim.

### 1.4. Per-mark-specific LSTM encoders

**Question:** does the bottleneck come from a single shared `h(t)` having to encode multiple per-mark features, leading to interference?

**Method:** Train K parallel LSTMs (one per mark) with shared event embeddings, then combine their outputs through the mark head. The hidden state for mark `i` is shaped by events of all marks but its `W_λ_k_i` reads only from its own LSTM. This breaks the encoder bottleneck by giving each mark its own representational rank.

**Cost:** ~1 day code, ~$10 cloud. Architectural change requires care with backprop through K parallel paths.

**Yield:** Medium. If it works, we have a working model with a non-trivial architectural insight. If it fails, the bottleneck is in the shared input or training objective rather than in the encoder itself.

### 1.5. H2 — mark-agnostic spatial head (queued)

**Question:** is the MDN's `(h, mark_emb)` input providing a back-channel for per-mark variation that lets the LSTM avoid encoding it?

**Method:** Drop `mark_emb` from the MDN input. Otherwise identical to Tier 1-MLP. Cloud run + probe.

**Cost:** ~2 hr code, ~$5 cloud.

**Yield:** Low-medium. H6 stands regardless of H2's outcome — even if H2 succeeds, it just means the bottleneck has a fixable architectural shape via removing the spatial back-channel; the underlying observation that the LSTM doesn't encode parent-variation in `h` is unchanged. Lower priority than 1.1-1.3.

## Tier 2 — methodology generalization

These take the cross-view triangulation finding and ask whether it generalizes. The diagnostic itself is the publishable contribution; these would strengthen its claim.

### 2.1. Apply cross-view triangulation to other Neural TPP models

**Question:** does the rank-1 forward-sim collapse appear in other marked TPP architectures fit on heavily imbalanced data?

**Method:** Take the EONET corpus. Fit RMTPP (Du et al. 2016), Transformer Hawkes (Zuo et al. 2020), Attentive Neural Hawkes (Mei & Eisner extended). Run the same forward-sim vs. gradient-attribution probe pair against each fit. Check for the same row-degeneracy pattern.

**Cost:** ~1-2 weeks. Each model needs a faithful reimplementation or careful use of an existing library (EasyTPP exposes most of these).

**Yield:** Very high. If the row-degeneracy is a general property of marked TPP architectures fit on heavily-imbalanced data, the finding is a robust methodological one and the paper writes itself. If it's CTLSTM-specific, the paper is narrower but still publishable.

### 2.2. Formalize the triangulation methodology

**Question:** what's the general statement of "two interpretability views that should agree at infinite data, diagnose disagreement at finite data"? Does it generalize beyond TPPs?

**Method:** A theoretical / methods note. Identify the conditions: when can two derived quantities be expected to agree at infinite data? What inferences are licensed by disagreement? Possible analogues: saliency maps vs. occlusion in image classification, attention weights vs. probing classifiers in NLP, mechanistic interpretability vs. behavioral interpretability in language models.

**Cost:** Pure writing. ~1-2 weeks of focused work plus literature review.

**Yield:** High if you target a methods venue (NeurIPS, ICLR, AI workshops); medium if hazard-domain or portfolio piece.

### 2.3. Synthetic ground-truth dataset (Hawkes recovery test)

**Question:** how do the four interventions perform when we KNOW the true cascade matrix?

**Method:** Generate a Hawkes process with a known α matrix (mix of diagonal self-excitation + 2-3 off-diagonal couplings, plus heavily-imbalanced baseline rates that mimic the EONET wildfire dominance). Run the four interventions (Tier 1, 1.5, 1-MLP, 1-aux) on the synthetic data. Measure α-recovery error directly (no proxy).

**Cost:** ~3-5 days code (the synthetic generator is the main work), ~$20 cloud (four runs).

**Yield:** Very high. If the synthetic experiment also exhibits the row-degeneracy, the finding is about the model class, not the EONET data. If it doesn't, the EONET data has properties (e.g., spatial clustering, mark-correlated bursting) that aren't captured by vanilla Hawkes generators and are part of why the model collapses on real data. Either result is publishable.

This is probably the SECOND-most-valuable single experiment after Tier 2.1 (Transformer encoder).

## Tier 3 — data and domain work

These are valuable for the EONET catalog specifically, more than for the methodological story.

### 3.1. Region-stratified fits

**Question:** is the cascade structure spatially non-stationary in a way the global fit misses?

**Method:** Stratify the data into ~5-10 regions (CONUS subregions, Mexico, central America, etc.). Fit Tier 1-MLP on each region's subset. Compare cascade matrices.

**Cost:** ~1 week (mostly data subsetting + parallel cloud runs), ~$40-50 cloud.

**Yield:** Domain-specific. The headline question becomes "do floods follow storms differently in the Gulf coast vs. the Pacific Northwest?" — interesting for hazard-risk modelers.

### 3.2. Multi-resolution time

**Question:** is the cascade structure time-scale-dependent?

**Method:** Current Tier 1 model is in days. Re-fit with time in hours (for storm cascades) and weeks (for fire-then-mudslide cascades). Compare which couplings emerge at which time scale.

**Cost:** ~3-5 days code, ~$20 cloud.

**Yield:** Domain-specific. Interesting for risk-modeling stakeholders; less so for ML methodology.

### 3.3. Cross-catalog validation

**Question:** do the EONET findings replicate on other event catalogs?

**Method:** Run the same pipeline against IRIS earthquake catalog (~3M events), NOAA storm reports, USGS landslide catalog, FIRMS active fires. Even within the EONET set, our work hasn't actually used everything available.

**Cost:** ~1-2 weeks per additional catalog (data ingest is the long pole).

**Yield:** Validates the findings across data sources but doesn't open methodological ground.

## Tier 4 — operational / downstream work

These productionize what already works, without further research.

### 4.1. Hazard-rate forecasting service

**What:** Tier 1-MLP's overall event-rate predictions are accurate (val NLL 3.38, beating Tier 1 by 20%). Wrap as a service that takes a region + time window + recent history and returns predicted rates per mark.

**Yield:** Operational. Useful for stakeholders, not a research output.

### 4.2. Gradient-attribution-based cascade dashboard

**What:** Since gradient attribution recovers cascade structure on Tier 1-MLP (where forward-sim fails), build that as the user-facing cascade visualization. The vectorized attribution kernel makes this fast enough for an interactive dashboard.

**Yield:** Operational. Demonstrates the methodology in a deployable form.

## Recommended sequence if you continue this line of work

If you keep doing research on this topic, my ranked recommendation:

1. **Tier 1.1 — effective-rank diagnostic.** Cheap (1-2 hr code, no cloud), confirms H6 numerically, gives a transferable diagnostic that other researchers can use on their own marked-TPP models.
2. **Tier 2.3 — synthetic ground-truth experiment.** Separates "EONET-data property" from "model-class property" in the row-degeneracy finding. Probably the single most informative follow-up.
3. **Tier 1.3 — Transformer encoder.** Tests whether the encoder-rank bottleneck is LSTM-specific. Strengthens the paper's generality claim if Transformer also collapses; gives a working model if it doesn't.
4. **Tier 2.1 — apply to other TPP models.** Most labor-intensive but most generalizable. Pursue only if 1.1 + 2.3 + 1.3 all converge on a clear methodological framing worth a full-scale generality study.

Steps 1-3 together are ~$25 in cloud spend and ~2 weeks of focused work — a reasonable "Tier 5" project scope.

## What to skip

- **Tier 1.5 follow-ups (H2, longer training, more aggressive rebalancing):** the four-run chain has decisively eliminated mark-head-targeted interventions. More variations of those don't add information.
- **Larger architectural overhauls before the diagnostic experiments (Tier 1.1, 2.3) are done.** Don't burn cloud on, e.g., per-mark encoders (Tier 1.4) before you know whether the bottleneck is encoder-class or model-class.
- **Hyperparameter sweeps.** All four runs use the same `lr 1e-3`, `hidden_dim 64`, etc. Trying small variations is unlikely to move the row-degeneracy needle when the architecture-level changes didn't.

## The honest summary

The natural next experiment is the Transformer encoder (Tier 1.3). The natural next analysis is the synthetic recovery test (Tier 2.3). The natural next paper is the cross-view triangulation methodology (Tier 2.2), with the EONET four-run chain as the headline case study and either Transformer (if also collapses) or RMTPP (if you do Tier 2.1) as the generality evidence.

If none of this happens — if the project ends with the writeup of what we have — that's also a defensible stopping point. Four cloud runs, $19 spent, one working model (Tier 1-MLP), one characterized failure mode, one methodological contribution. That's a complete piece of work as-is.
