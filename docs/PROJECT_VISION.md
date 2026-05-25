# eonet-cascades — project vision

**Last updated:** 2026-05-25 (during Task 12 cloud training wait window)
**Owner:** Liam Schmidt
**Status:** mid-Phase 3 (Plan 4 / Tier 1 in active training)

This is a living orientation doc. Falls back here when context drifts.
Sections marked **[NEEDS YOUR INPUT]** are hedges where I'm guessing —
edit in place and remove the tag.

---

## 1. The question

What is the **cascade structure of natural hazards** over CONUS + Mexico,
2000–present? Specifically: which event types trigger which others, with
what time lag and what spatial signature, and how stationary is that
structure across years and regions?

Operationally: infer the K×K cross-mark excitation matrix from a
spatio-temporal marked point process fit to NASA EONET catalog data,
and ask whether it agrees with priors (earthquake → landslide,
severe_storm → flood) AND surfaces non-obvious structure that domain
priors would miss.

---

## 2. The data

- **Source:** NASA EONET (Earth Observatory Natural Event Tracker) +
  supporting catalogs harmonized to a common schema.
- **Storage:** DuckDB at `/Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb`
  (1.1 GB; ~2.4M events).
- **Marks (K=7):** dust_haze, earthquake, flood, landslide, severe_storm,
  tornado, wildfire.
- **Per event:** `(time, longitude, latitude, mark)`.
- **Splits used in Tier 1 scale-B run:**
  - Train: 2022-01-01 → 2024-06-30 (200k subsample of 2.385M)
  - Val: 2024-07-01 → 2024-12-31 (356k events)

---

## 3. The method — three model tiers

Same likelihood interface. Same val/test split. Same eval harness.
Apples-to-apples comparison across functional forms of conditional
intensity `λ_k(t, x | history)`.

| Tier | Form | Tradeoff |
|---|---|---|
| **0** | Parametric multivariate Hawkes: constant `α_{ij}` matrix × exponential time kernel × isotropic Gaussian space kernel | Max interpretability; rigid functional form; closed-form NLL |
| **1** | Neural Hawkes: CTLSTM (Mei & Eisner 2017) hidden state + softplus temporal intensity + softmax mark head + full-covariance bivariate MDN spatial head | History-dependent, non-stationary; attribution matrix recovers α-equivalent K×K view; ~30k params |
| **2** | Transformer Hawkes (not yet built) | Long-range attention; scales differently |

**Interpretability bridge (Tier 1 → cascade graph):** Tier 1 has no
explicit α. Two derived K×K views, both implemented and tested:
1. **Gradient attribution** — per-event Jacobian of
   `log λ_{k_child}` w.r.t. earlier hidden states, exponentially decayed
   in time and aggregated parent→child.
2. **Forward-simulation transitions** — Monte Carlo simulate forward 14 days
   from each starting mark, count child marks.

Side-by-side these three K×K matrices (Tier 0 α, Tier 1 attribution,
Tier 1 forward-sim) are the **headline figure** (`make headline`,
currently stub).

---

## 4. What "done" looks like for the project

- [ ] Tier 0 fit on full corpus, NLL/event reported (Plan 3)
- [ ] Tier 1 trained on cloud GPU (Plan 4 / **this session, in progress**)
- [ ] Tier 1 attribution + forward-sim matrices computed (Plan 4 Task 13)
- [ ] Tier 0 vs Tier 1 cross-comparison notebook (Plan 4 Task 14)
- [ ] Tier 2 Transformer Hawkes (future plan, not scoped)
- [ ] Synthetic cascade-recovery gate passes on a controlled-α dataset
      for both tiers (partial — Tier 0 done, Tier 1 evidence in
      `docs/notes/tier1-recovery-gate-evidence.md`)
- [ ] Headline figure: K×K cascade graph(s), publication-quality
- [ ] Writeup — venue + length **[NEEDS YOUR INPUT]**

---

## 5. Audience & output

**[NEEDS YOUR INPUT]** — these are my guesses; correct them.

Candidate audiences:
- **Hazard / risk / disaster research community** — direct domain
  contribution. Likely venue: ESS Open Archive preprint → an
  *Environmental Modelling & Software* / *Natural Hazards* / *Risk
  Analysis* journal.
- **ML methods community** — benchmark contribution.
  Likely venue: workshop at NeurIPS / ICML on spatiotemporal data or AI
  for science. Less obvious headline science.
- **Portfolio / interview deliverable** — for industry ML roles
  (per memory: career direction is industry-leaning, NY/NE constrained,
  2–3 yr PhD horizon). End-to-end ML + infra + interpretability story.

Best guess: **all three** can be served by the same artifact set if the
writeup is structured to lead with the methodological benchmark and
the cascade-graph result as the "real-world demonstration." The
portfolio angle is essentially free once the scientific writeup exists.

Decide before drafting the writeup which audience the *headline framing*
serves; the others get a paragraph each.

---

## 6. How this connects to the broader picture

### 6a. To Liam's physics work

**[NEEDS YOUR INPUT — this is the most speculative section.]** My read,
to be confirmed or rejected:

**Method-level transfer (firm-ish):**
- Marked point processes show up directly in spectroscopy: photon
  counting in RIXS, neutron event timing, single-photon EXAFS at
  modern sources. The likelihood structure (intensity over time +
  mark + observable) is mechanically similar. Skills here probably
  transfer to a custom RIXS detector-data pipeline if you wanted one.
- Probabilistic deep learning workflow (TDD-style component testing,
  synthetic-recovery gates, cloud GPU training, interpretability via
  attribution) is portable to any physics ML project — magnon
  spectroscopy fitting, altermagnetic band-structure inference, etc.

**Conceptual analogy (speculative):**
- Self-excitation / cascade structure in Hawkes is mathematically
  similar to correlated-electron cascade phenomena (e.g. magnon-magnon
  decay, multi-quasiparticle generation after a core hole). Not a
  direct mapping but a useful frame.

**Pragmatic (firm):**
- This project is your demonstrated end-to-end ML fluency on a
  non-trivial probabilistic model, with infrastructure and
  interpretability. That demo is valuable independent of any physics
  content if/when the industry-pivot happens.

### 6b. To the career direction

Per `~/.claude/projects/.../memory/career-direction.md`: industry-leaning,
NY/NE required, 2-3 yr left of PhD, money-primary over 10-30 yr.
This project lines up with that path:
- ML/probabilistic modeling skills hire well in industry (climate-risk
  startups, reinsurance, hazard-modeling at govs, finance quant
  desks — all use point-process methods).
- The "scientific demonstration on real data" framing is more
  defensible in interviews than a pure-Kaggle portfolio piece.
- Reproducible pipeline + cloud GPU + Anthropic-API tooling (per the
  rest of the workspace) demonstrates ops fluency that academic
  physicists often lack.

### 6c. To the broader ML / climate research landscape

- **Climate-risk modeling** is increasingly cross-mark: reinsurance,
  catastrophe (cat) modeling, and federal hazard planning all want
  models that capture how a tornado outbreak primes flood risk,
  how a wildfire scar primes mudslides. Current operational models
  use either parametric Hawkes (decades-old) or hand-coded rule
  systems. A defensible neural alternative with interpretable
  attribution is a real gap.
- **Spatiotemporal ML benchmarks** are sparse for hazard data
  specifically. EasyTPP and related libraries exist but use mostly
  social-network / NYC-taxi data. A real-Earth hazard benchmark is
  itself a contribution.

---

## 7. Open questions / deferred decisions

- **Venue:** see §5 above.
- **Whether to add a region-stratified Tier 0/1 fit** (CONUS vs Mexico,
  or west-coast vs east-coast) to test for non-stationarity. Currently
  one global fit per tier.
- **Whether to use the val-NLL/event as the headline metric or a
  cascade-recovery metric** (e.g. agreement with priors over the
  obvious pairs). Both are useful; they answer different questions.
- **Tier 2 (Transformer Hawkes)** — scoping. Add only if Tier 1 leaves
  a clear gap; otherwise close the project at Tier 1 + cross-comparison.
- **Public data + open-source release timing.** Currently private repo
  `liamlts/eonet-cascades`. Decision needed before any preprint.

---

## 8. Change log

| Date | Change | Trigger |
|---|---|---|
| 2026-05-25 | Initial draft | During Task 12 cloud training, user asked for a fallback orientation doc |

---

*This doc is short on purpose. Keep it under ~300 lines as it grows.
Move expanded sections (e.g. detailed venue analysis, the full
methodological lit review) into `docs/notes/` and link out.*
