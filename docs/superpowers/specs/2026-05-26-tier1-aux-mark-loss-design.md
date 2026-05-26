# Tier 1 with auxiliary mark-classification loss — H4 design spec

**Date:** 2026-05-26
**Author:** Liam Schmidt (with Claude)
**Status:** approved (brainstorming output; implementation plan to follow via writing-plans skill)
**Related:**
- [PROJECT_VISION.md](../../PROJECT_VISION.md)
- [docs/notes/tier1_5-result.md](../../notes/tier1_5-result.md) — Tier 1.5 negative result + H4 hypothesis stub
- [docs/notes/tier1_mlp-result.md](../../notes/tier1_mlp-result.md) — Tier 1-MLP negative result + sharpened H5 framing
- [docs/superpowers/specs/2026-05-26-tier1-mlp-mark-head-design.md](2026-05-26-tier1-mlp-mark-head-design.md) — prior H3 spec

## 1. Motivation

Three runs to date all exhibit the same rank-1 mark-head collapse:

| run | mark head | training | val NLL | probe row-dev |
|-----|-----------|----------|---------|---------------|
| Tier 1   | Linear | default | 4.20 | 0.0012 |
| Tier 1.5 | Linear | rebalance + stratified | 6.80 | 0.0000 |
| Tier 1-MLP | MLP | default | **3.38** | 0.0000 |

Two interventions targeted at the most obvious causes — class imbalance
(Tier 1.5) and linear-head capacity (Tier 1-MLP) — both failed the
primary acceptance criterion (probe row-deviation > 0.1). The MLP run
achieved the best val NLL of any tier yet still collapsed completely.

**Working hypothesis H5** (from `tier1_mlp-result.md`): the joint Hawkes
log-likelihood has insufficient gradient signal on mark *composition*
under heavy class imbalance. The mark head's outputs `z = W_λ(h)`
control both per-mark intensity (via softplus(z), feeding the rate
log-likelihood) AND per-mark categorical probability (via the implicit
softmax that forward-sim's multinomial samples from). The joint
Hawkes loss dominates on the rate component; the composition gradient
is too weak to drive the head away from a marginal-only optimum.

This experiment tests H4: **add an explicit cross-entropy auxiliary
loss on the mark prediction** to give the mark head a stronger,
composition-only gradient signal. If H4 works, we ship a working
interpretable model. If it fails, H5 is essentially proved by
exclusion — four interventions, none fix the collapse, the diagnosis
points squarely at the joint Hawkes loss objective itself.

## 2. Architecture change

No model architecture changes from Tier 1-MLP. Same `mark_head="mlp"`
configuration. The change is only to the training objective:

### Auxiliary loss formulation

Let `z = W_λ(h(t_i)) ∈ ℝ^K` be the raw mark-head logits at event i (the
output of `nn.Sequential(Linear(H, H//2), ReLU(), Linear(H//2, K))`).
The auxiliary loss is:

```
L_aux = sum_{i=1..N}  -log [ softmax(z_i) ]_{k_i_observed}
```

i.e., standard cross-entropy of the categorical distribution
`softmax(z)` against the observed mark `k_i_observed`. The total
training loss becomes:

```
total_loss = -log_likelihood + aux_lambda * L_aux
```

where `log_likelihood` is the existing joint Hawkes log-likelihood
(rate + spatial terms) and `aux_lambda` is the coefficient (default 1.0).

### Why softmax(z), not softplus(z)/Σsoftplus(z)?

- Softmax is shift-invariant in z. Adding a constant to every z_k
  changes the per-mark rates (softplus output) but NOT the categorical
  distribution. This cleanly separates the rate gradient (driven by
  joint NLL) from the composition gradient (driven by aux loss).
- Softplus(z)/Σ would couple the two gradients; an attempt to fix
  composition would perturb rates and trigger compensating rate-loss
  gradients. Signal mixing.
- The mark head sees `∂L_aux/∂z = softmax(z) - one_hot(k_observed)`,
  which is the standard classification gradient — well-understood
  optimization properties.

### Why not weight individual classes within the aux loss?

The original spec for Tier 1.5 added inverse-sqrt class weights. That
intervention was meant to compensate for the asymmetric gradient signal
in the joint Hawkes loss. With an explicit cross-entropy auxiliary
loss applied uniformly, every event contributes equally to the
composition gradient regardless of its mark. The aux loss IS the
explicit signal we were trying to manufacture in Tier 1.5; we don't
need to re-weight it.

## 3. CLI surface

`eonet model train-neural-hawkes` gains one flag:

```
--aux-lambda    FLOAT  (default: 0.0)
```

When `> 0.0`, applies the auxiliary cross-entropy loss with that
coefficient during training. Eval (`_tier1_eval_loop`) is unchanged and
reports pure Hawkes NLL — so val NLL stays directly comparable to the
4.20 baseline.

Default 0.0 preserves the existing Tier 1 and Tier 1-MLP behavior; this
flag is opt-in.

## 4. Training configuration

Identical to Tier 1-MLP except for `--aux-lambda 1.0` and output dir:

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
  --aux-lambda 1.0 \
  --out-dir runs/tier1_aux/$(date -u +%Y%m%d_%H%M%S) \
  > train_tier1_aux.log 2>&1 &
```

**No `--mark-rebalance`, no `--stratify-train`.** Single-variable test:
the only thing different from Tier 1-MLP is the addition of the aux
loss with coefficient 1.0.

Cold-start, random init.

Same 2.385M train / 356k val data splits as all prior runs.

## 5. Acceptance criteria

| criterion | threshold | rationale |
|-----------|-----------|-----------|
| (P) Forward-sim probe max row-deviation | > 0.1 | 100× the cleanest prior result; clearly breaks the collapse |
| (S) Val NLL/event | ≤ 4.41 (within 5% of Tier 1's 4.20) | guards against the aux loss destroying likelihood |
| (T) Cross-tier notebook re-execution | clean pass | sanity check |

| outcome | decision |
|---------|----------|
| P + S both pass | **H4 confirmed; aux loss is the fix.** Ship the model; pivot to writeup with a working interpretable cascade predictor. |
| P passes, S fails | Aux loss fixes composition but at unacceptable likelihood cost. Re-run with `--aux-lambda 0.1` or `--aux-lambda 10.0` to tune the trade-off. |
| P fails (whether S passes or not) | **H4 ruled out.** H5 essentially proved: the loss-function-itself diagnosis stands. Stop iterating and pivot to writeup of the negative chain as a methodological finding. |
| Only T fails | Local notebook patch; not a scientific finding. |

## 6. Implementation order

1. Modify `NeuralHawkes.forward()` to expose raw mark-head logits `z`
   in the returned dict as `z_at_events` (shape (N, n_marks)). Currently
   the forward pass discards z after softplus.
2. Modify `NeuralHawkes.log_likelihood()` to accept optional
   `aux_lambda: float = 0.0`. When > 0, add the cross-entropy term
   computed from `z_at_events` and the observed marks.
3. Modify `training.neural_loop.train_one_epoch` to accept and pass
   through `aux_lambda`.
4. Add `--aux-lambda` Typer flag to the CLI train command; thread it
   through to `train_one_epoch`. Persist the value in the saved
   checkpoint's `config` dict.
5. Add tests in `tests/test_aux_mark_loss.py`:
   - **default `aux_lambda=0.0`** gives bit-identical log_likelihood to
     pre-change behavior (backwards-compat guard)
   - **non-zero `aux_lambda`** changes log_likelihood value (sanity)
   - **finite output** for `aux_lambda=1.0` on tiny synthetic
   - **z_at_events shape** is (N, K) in forward() output
   - **aux gradient flows to W_lambda_k** — verify via .grad after backward
6. Run all existing tests; nothing regresses.
7. Local smoke test: 1 epoch on Jan 2024 slice with `--aux-lambda 1.0`,
   `--device cpu`. Confirm CLI + training loop work end-to-end.
8. Commit + push to `origin/main`.
9. Update `docs/notes/tier1-cloud-runbook.md` with a Step 5‴ section
   for the aux-loss invocation; commit + push.
10. Hand off to user for cloud run.
11. (Post-cloud) Probe the new checkpoint.
12. (Post-cloud) Decision per acceptance table; write up result.

## 7. Risks and mitigations

| risk | mitigation |
|------|-----------|
| `forward()` change breaks existing callers | New key `z_at_events` is additive; old keys preserved. The probe script, attribution code, and existing tests pull keys by name. New code is opt-in. |
| `aux_lambda=0.0` not bit-identical to pre-change | TDD test 5a explicitly checks this with parameter-by-parameter equality. |
| Aux loss explodes early in training (logits very large/small) | Standard cross-entropy is numerically stable via `F.cross_entropy` or `F.log_softmax`. Use PyTorch's built-in. |
| Val NLL degrades to Tier 1.5 levels (~6.8) | If P passes but S fails, we already have the H4-confirms-composition story; tune lambda down to 0.1 in a follow-up run. ~$5. |
| Aux loss provides composition signal but model "cheats" by routing through MDN | The MDN gets (h, mark_emb) as input — it conditions on mark, but the mark embedding for the *current* event is the GROUND TRUTH mark, not the mark head's prediction. So the MDN cannot be a back-door for the mark head. The aux loss must drive W_λ to learn composition or there's nothing left to compensate. |
| Cloud cost overrun | Time-box at $10. If training stalls or diverges, kill within 30 min. |

## 8. Out of scope

- Sweeping `aux_lambda` (0.1, 1.0, 10.0) in a single experiment — staged
  rollout: try 1.0 first, then tune based on outcome
- Replacing the joint Hawkes objective wholesale with a multi-task
  objective with learnable task weights — much bigger change, only
  consider if aux=1.0 partially works
- Architectural changes (the MLP head from Tier 1-MLP carries over
  unchanged)
- `--mark-rebalance` / `--stratify-train` combined with aux loss —
  single-variable test for now

## 9. Definition of done

- [ ] Code changes committed and pushed to `origin/main`
- [ ] All existing tests still pass; new aux-loss tests pass
- [ ] Local smoke test runs end-to-end on Jan 2024 slice without error
- [ ] Cloud runbook updated with the aux-loss invocation (Step 5‴)
- [ ] Cloud experiment run; checkpoint pulled back to local
- [ ] Forward-sim probe run; result recorded against acceptance table
- [ ] Outcome documented in `docs/notes/tier1_aux-result.md`
- [ ] If P passes (H4 confirmed): notebook 03 re-rendered with new
      checkpoint; PPT/Word status docs updated; manuscript outline
      drafted
- [ ] If P fails (H4 refuted): a final summary notes doc
      `docs/notes/mark-head-collapse-chain.md` consolidating all four
      runs (Tier 1, 1.5, MLP, aux) as the methodological finding;
      manuscript outline drafted around the negative chain.
