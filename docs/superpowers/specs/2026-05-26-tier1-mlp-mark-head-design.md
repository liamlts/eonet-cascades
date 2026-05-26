# Tier 1 with MLP mark head — design spec

**Date:** 2026-05-26
**Author:** Liam Schmidt (with Claude)
**Status:** approved (this file is the brainstorming output; implementation plan to follow via writing-plans skill)
**Related:** [PROJECT_VISION.md](../../PROJECT_VISION.md), [docs/notes/tier1_5-result.md](../../notes/tier1_5-result.md), [docs/superpowers/specs/2026-05-25-tier-1-neural-hawkes-design.md](2026-05-25-tier-1-neural-hawkes-design.md)

## 1. Motivation

The Tier 1 Neural Hawkes (commit `420d5a3`) and the class-rebalanced Tier 1.5
variant (commit `d97ae60`) both exhibit a mark-head rank collapse: the row-
normalized per-mark intensities `λ_k / Σ_l λ_l` are numerically identical
across all 8 parent marks (max deviation < 0.002 for Tier 1, exactly 0.0000
for Tier 1.5). The hidden state `h(t)` drives the *total* intensity correctly
(wildfire channel goes 40 → 56 cold→warm seed) but the *composition* across
marks is locked.

The Tier 1.5 retrain (15 epochs, stratified subsample, inverse-sqrt weights
with 400× dynamic range, Lambda Labs A10, ~$5) failed both acceptance
criteria: probe row-deviation went 0.0023 → 0.0000 (regression) and val
NLL/event went 4.20 → 6.80 (+62%). The class-imbalance diagnosis is refuted.

Three remaining hypotheses (full discussion in `tier1_5-result.md`):

1. **Under-training** — refuted by the convergence curve. Last three Δ-val-NLL
   values were −0.020, −0.008, −0.001. Linear extrapolation to 100 more
   epochs buys ~0.1 nat/event. Not breaking any collapse.
2. **MDN spatial head absorbs mark conditioning** — plausible. The MDN gets
   `(h, mark_emb)` as input and can express per-mark spatial distributions,
   which may let the joint NLL be minimized without the mark head doing work.
3. **Linear mark head insufficient capacity** — **the prime suspect.** The
   current head is `softplus(W·h)` where `W ∈ ℝ^{K×H}` is a single linear
   layer. Row degeneracy across parents corresponds to the K linear functions
   of `h` being approximately rank-1 aligned in lambda-space. A non-linear
   head with at least one hidden layer cannot exhibit this exact pathology
   because the per-mark logits are no longer linear functions of `h`.

This spec covers the test of hypothesis 3. Hypothesis 2 is the pre-committed
next experiment if this one fails.

## 2. Architecture change

Single constructor argument added to `NeuralHawkes.__init__`:

```python
def __init__(
    self,
    n_marks: int,
    hidden_dim: int = 64,
    mark_emb_dim: int = 16,
    spatial_emb_dim: int = 16,
    n_mix: int = 8,
    mark_head: str = "linear",  # NEW: "linear" | "mlp"
) -> None:
    super().__init__()
    ...
    if mark_head == "linear":
        self.W_lambda_k = nn.Linear(hidden_dim, n_marks)
    elif mark_head == "mlp":
        self.W_lambda_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 64 → 32
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_marks),     # 32 → 8
        )
    else:
        raise ValueError(f"unknown mark_head: {mark_head!r}")
    self.mark_head = mark_head
```

Nothing else in the model changes. `_lambda_k(h)`, `forward`, `log_likelihood`,
and `_lambda_total_at` all call `self.W_lambda_k(h)` polymorphically.

**MLP geometry rationale:**
- One hidden layer (smallest non-trivial topology that adds non-linearity)
- Width = `hidden_dim // 2 = 32` (smallest non-trivial width with a clear
  bottleneck, so any rank-collapse pathology would have to survive the
  bottleneck to manifest)
- ReLU activation (safe default; GELU/SiLU equivalent for 8-output heads)
- Bias on both layers (PyTorch default)

**Parameter count:** mark head goes 520 → 2,376 params; full model goes from
~30,000 → ~32,000.

**Why a flag and not a wholesale replacement:** existing checkpoints under
`runs/tier1/...` and `runs/tier1_5/...` have keys `W_lambda_k.weight` and
`W_lambda_k.bias`. The MLP arch has keys `W_lambda_k.0.weight`,
`W_lambda_k.0.bias`, `W_lambda_k.2.weight`, `W_lambda_k.2.bias`. Defaulting
`mark_head="linear"` preserves backward-compatible checkpoint loading; the
new experiment uses `mark_head="mlp"`.

The string is persisted in the checkpoint's `config` dict on save, and read
back by checkpoint-loading code (CLI eval path, probe script, attribution
runners) to construct the matching architecture.

## 3. CLI surface

`eonet model train-neural-hawkes` gains one flag:

```
--mark-head    linear | mlp   (default: linear)
```

Backwards-compatible: existing invocations without the flag get the linear
head as before. The Tier 1 MLP experiment runs with `--mark-head mlp`.

No other CLI changes. `--mark-rebalance` and `--stratify-train` remain
opt-in flags; the experiment runs with them OFF.

## 4. Training configuration

Identical to original Tier 1 except for `--mark-head mlp` and the output
directory:

```bash
nohup uv run eonet model train-neural-hawkes \
  --since 2022-01-01 --until 2024-06-30 \
  --val-until 2024-12-31 \
  --sample 200000 \
  --n-epochs 15 \
  --hidden-dim 64 \
  --lr 1e-3 \
  --device cuda \
  --mark-head mlp \
  --out-dir runs/tier1_mlp/$(date -u +%Y%m%d_%H%M%S) \
  > train_tier1_mlp.log 2>&1 &
```

Cold-start, random init for both LSTM and MLP head. No warm-starting from
Tier 1 checkpoint (would lock in the rank-1 lambda alignment we're trying
to break).

**Data note:** the cloud DuckDB contains the same 2.385M train / 356k val
events used by Tier 1 and Tier 1.5. The K=8 vocabulary (with the singleton
`volcanic_eruption`) is identical to Tier 1.5. Val NLL is therefore
directly comparable to Tier 1's 4.20.

## 5. Acceptance criteria

| criterion | threshold | rationale |
|-----------|-----------|-----------|
| (P) Forward-sim probe max row-deviation | > 0.1 | 100× the cleanest prior result; clearly breaks the collapse |
| (S) Val NLL/event | ≤ 4.41 (within 5% of Tier 1's 4.20) | guards against Pyrrhic fix |
| (T) Cross-tier notebook (03) re-execution | clean pass with no code changes | sanity check that downstream code is checkpoint-agnostic |

| outcome | decision |
|---------|----------|
| P + S both pass | H3 confirmed; MLP is the fix; advance to writeup |
| P passes, S fails | Fix-with-cost; weaker but still publishable; investigate optimizer/lr |
| P fails (whether or not S passes) | H3 ruled out; advance to H2 (mark-agnostic spatial head, spec to follow) |
| Only T fails | Local notebook patch; not a scientific finding |

## 6. Implementation order

1. Model change: add `mark_head` arg to `NeuralHawkes.__init__` + branch
2. CLI change: add `--mark-head` flag to `train-neural-hawkes`; persist in
   checkpoint's `config` dict
3. Probe script: read `mark_head` from checkpoint config before model
   construction (so probe works with both linear and mlp checkpoints)
4. Tests added to `tests/test_mark_head_mlp.py`:
   - `mark_head="mlp"` constructor builds without error
   - `log_likelihood` returns finite scalar on a synthetic K=3, N=20 sequence
   - checkpoint round-trip: save state_dict + config, reload, parameters
     numerically identical
   - backward compat: `mark_head="linear"` is bit-identical to current default
     on the same synthetic forward pass
5. Run all existing tests (`pytest tests/`) — must all pass
6. Local smoke test: 1 epoch on Jan 2024 slice, `--mark-head mlp --device cpu`,
   confirms the full training loop executes end-to-end and saves a checkpoint
7. Commit changes + push to `origin/main`
8. Update `docs/notes/tier1-cloud-runbook.md` with a Step-5″ section for the
   MLP-head invocation; commit + push
9. **Hand off to user** for the cloud run. Local-side work is complete.
10. (Post-cloud) User SCPs `runs/tier1_mlp/<ts>/` back to local machine
11. (Post-cloud) Update `scripts/probe_forward_sim.py` `RUN_DIR` to the new
    checkpoint, run probe, capture output
12. (Post-cloud) Decision per acceptance table; if pass, re-render notebook 03
    with the new checkpoint and commit

## 7. Risks and mitigations

| risk | mitigation |
|------|-----------|
| Checkpoint load break on Tier 1/1.5 paths | Default `mark_head="linear"` + test (4d) explicitly checks bit-identity to current behavior |
| MLP overfits on rare marks | Unlikely with K=8 and width-32, but check per-mark val NLL in the post-run analysis |
| Cloud cost overrun | Time-box this experiment at $10 (~13 hr at $0.75/hr). If GPU sits idle or training stalls, kill within 30 min |
| Local smoke test passes but cloud fails on K=8 vocabulary | K=8 quirk is now expected; pre-flight on a 1-month slice before the full run, as we did for Tier 1.5 |
| Optimizer fails to find a good init for the MLP | Standard PyTorch init has been fine for 100M-parameter models; should be a non-issue here. Fallback: try Kaiming-He init on the linear layers if loss diverges in epoch 0. |

## 8. Out of scope

- Larger MLP geometries (width 64 single hidden, two hidden layers, etc.) —
  deferred until the minimal MLP is shown to be insufficient
- Different activations (GELU, SiLU) — deferred; ReLU is the safe default
- Other hypotheses (H2 mark-agnostic spatial head, H4 auxiliary loss) —
  pre-committed as the next experiment if H3 fails, but not in this spec
- Re-fitting Tier 0 on a fair val slice — separate work item per
  `PROJECT_VISION.md §4`
- Manuscript drafting — gated on a clean H3 result

## 9. Definition of done

- [ ] Code changes committed and pushed to `origin/main`
- [ ] All existing tests still pass; new mark-head tests pass
- [ ] Local smoke test runs end-to-end on the Jan 2024 slice without error
- [ ] Cloud runbook updated with the MLP-head invocation
- [ ] Cloud experiment run; checkpoint pulled back to local
- [ ] Forward-sim probe run; result recorded against acceptance table
- [ ] Outcome documented in `docs/notes/tier1_mlp-result.md` (new file)
- [ ] If pass, notebook 03 re-rendered; commit captures the new figure
- [ ] If fail, H2 spec drafted (next-experiment design) before any further compute spend
