# Tier 1 with Auxiliary Mark-Classification Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--aux-lambda FLOAT` flag that augments the joint Hawkes training NLL with a cross-entropy auxiliary loss `Σ_i -log softmax(z_i)_{k_i_observed}`, where `z` are the raw mark-head logits. Eval is unchanged so val NLL stays comparable to Tier 1's 4.20. Then run on Lambda Labs A10 to test H4 against the mark-head rank-1 collapse documented across Tier 1, Tier 1.5, and Tier 1-MLP.

**Architecture:** No model architecture change from Tier 1-MLP. The mark head stays `nn.Sequential(Linear(H, H//2), ReLU(), Linear(H//2, K))`. `NeuralHawkes.forward()` gains a new `z_at_events` key in its return dict (raw pre-softplus logits). `NeuralHawkes.log_likelihood()` gains an optional `aux_lambda: float = 0.0` parameter that, when > 0, adds `aux_lambda * cross_entropy(z_at_events, marks)` to the loss. Training loop and CLI thread the value through; eval defaults to 0.0.

**Tech Stack:** Python 3.11, PyTorch 2.2 (`F.cross_entropy`, `F.log_softmax` for numerical stability), Typer CLI, pytest, uv.

**Spec:** [docs/superpowers/specs/2026-05-26-tier1-aux-mark-loss-design.md](../specs/2026-05-26-tier1-aux-mark-loss-design.md)

---

## Task 1: Write failing aux-loss tests

**Files:**
- Create: `tests/test_aux_mark_loss.py`

- [ ] **Step 1.1: Branch-assert**

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
git branch --show-current  # must be: main
git log -n 1 --oneline     # expect: 88c7e33 spec(tier1-aux): ... or a later HEAD on main
```

If branch isn't `main`, STOP and report BLOCKED.

- [ ] **Step 1.2: Create the test file with imports and shared fixture**

Write to `tests/test_aux_mark_loss.py`:

```python
"""Tests for the auxiliary mark-classification loss (H4 experiment).

Surfaces tested:
  * NeuralHawkes.forward() returns a 'z_at_events' key with raw mark-head
    logits (shape (N, n_marks)), pre-softplus.
  * NeuralHawkes.log_likelihood(..., aux_lambda=0.0) is bit-identical to
    the current default (backwards-compat guard).
  * Non-zero aux_lambda changes the log_likelihood value (sanity).
  * aux_lambda=1.0 produces a finite log_likelihood on a tiny synthetic.
  * The aux gradient flows to W_lambda_k parameters after backward().
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from eonet_cascades.models.neural_hawkes import NeuralHawkes


def _small_inputs(n_marks: int = 3, n_events: int = 20, seed: int = 0):
    """Reusable synthetic event sequence."""
    rng = np.random.default_rng(seed)
    times = torch.tensor(np.sort(rng.uniform(0.0, 20.0, size=n_events)), dtype=torch.float32)
    lons = torch.tensor(rng.uniform(-10.0, 10.0, size=n_events), dtype=torch.float32)
    lats = torch.tensor(rng.uniform(-5.0, 5.0, size=n_events), dtype=torch.float32)
    marks = torch.tensor(rng.integers(0, n_marks, size=n_events), dtype=torch.long)
    return times, lons, lats, marks
```

- [ ] **Step 1.3: Add forward()-exposes-z test**

Append:

```python
def test_forward_returns_z_at_events_with_correct_shape():
    """forward() output dict must include 'z_at_events' with shape (N, K)."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=5, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    model.eval()

    times, lons, lats, marks = _small_inputs(n_marks=5, n_events=12, seed=0)
    out = model(times, lons, lats, marks)
    assert "z_at_events" in out, f"missing 'z_at_events' key; got {sorted(out.keys())}"
    z = out["z_at_events"]
    assert z.shape == (12, 5), f"expected (12, 5), got {tuple(z.shape)}"
    # z is pre-softplus logits and can be negative; sanity-check it's finite.
    assert torch.isfinite(z).all()
```

- [ ] **Step 1.4: Add aux_lambda=0.0 backward-compat test**

Append:

```python
def test_aux_lambda_zero_is_bit_identical_to_no_aux():
    """log_likelihood(..., aux_lambda=0.0) and log_likelihood(...) with no
    aux_lambda kwarg must return the EXACT same tensor. Backwards-compat."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=4, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    model.eval()

    times, lons, lats, marks = _small_inputs(n_marks=4, n_events=15, seed=1)
    ll_default = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0))
    ll_zero = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0), aux_lambda=0.0)
    assert torch.equal(ll_default, ll_zero), "aux_lambda=0.0 should be no-op"
```

- [ ] **Step 1.5: Add non-zero aux_lambda changes value test**

Append:

```python
def test_nonzero_aux_lambda_changes_log_likelihood():
    """Sanity: aux_lambda > 0 actually changes the loss."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=4, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    model.eval()

    times, lons, lats, marks = _small_inputs(n_marks=4, n_events=15, seed=1)
    ll_zero = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0), aux_lambda=0.0)
    ll_one = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0), aux_lambda=1.0)
    assert not torch.equal(ll_zero, ll_one), "aux_lambda=1.0 should change the loss"
    # The aux loss is subtracted from log_likelihood (it's a NEGATIVE log prob
    # added as a penalty to the maximum-likelihood objective). So
    # log_likelihood with aux_lambda > 0 should be LESS than without.
    assert ll_one.item() < ll_zero.item(), (
        f"aux loss should reduce log_likelihood; got ll_one={ll_one.item():.4f}, "
        f"ll_zero={ll_zero.item():.4f}"
    )
```

- [ ] **Step 1.6: Add finite-output test**

Append:

```python
def test_aux_lambda_one_log_likelihood_finite():
    """log_likelihood(..., aux_lambda=1.0) returns a finite scalar."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=4, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    model.eval()

    times, lons, lats, marks = _small_inputs(n_marks=4, n_events=15, seed=2)
    ll = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0), aux_lambda=1.0)
    assert ll.dim() == 0, f"expected scalar, got shape {tuple(ll.shape)}"
    assert torch.isfinite(ll), f"non-finite log_likelihood: {ll.item()}"
```

- [ ] **Step 1.7: Add aux-gradient-flows test**

Append:

```python
def test_aux_loss_gradient_flows_to_mark_head():
    """When aux_lambda > 0, backward() on -log_likelihood populates .grad on
    the W_lambda_k mark-head parameters. Guards against the aux loss being
    accidentally detached from the mark head."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=4, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    # Zero out all .grad before computing.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    times, lons, lats, marks = _small_inputs(n_marks=4, n_events=15, seed=3)
    ll = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0), aux_lambda=1.0)
    loss = -ll
    loss.backward()

    # Verify the mark head's linear layers got gradient.
    head = model.W_lambda_k
    assert isinstance(head, nn.Sequential), "expected MLP head for this test"
    for i, sub in enumerate(head):
        if isinstance(sub, nn.Linear):
            g = sub.weight.grad
            assert g is not None, f"head[{i}].weight has no .grad"
            assert torch.isfinite(g).all(), f"head[{i}].weight.grad has non-finite values"
            assert g.abs().sum().item() > 0.0, f"head[{i}].weight.grad is all zeros"
```

- [ ] **Step 1.8: Run tests and confirm they FAIL with expected errors**

```bash
export PATH="$HOME/.local/bin:$PATH"
unset DYLD_LIBRARY_PATH
uv run pytest tests/test_aux_mark_loss.py -v 2>&1 | tail -25
```

Expected failures:
- `test_forward_returns_z_at_events_with_correct_shape` — KeyError or AssertionError on missing 'z_at_events' key
- `test_aux_lambda_zero_is_bit_identical_to_no_aux` — TypeError: unexpected keyword argument 'aux_lambda'
- `test_nonzero_aux_lambda_changes_log_likelihood` — same TypeError
- `test_aux_lambda_one_log_likelihood_finite` — same TypeError
- `test_aux_loss_gradient_flows_to_mark_head` — same TypeError

CRITICAL: confirm failures are about `aux_lambda` kwarg / `z_at_events` key — those prove the tests target the right surface.

- [ ] **Step 1.9: Commit failing tests**

```bash
git add tests/test_aux_mark_loss.py
git commit -m "test(neural_hawkes): failing tests for aux_lambda mark-classification loss

Will pass once aux_lambda kwarg is added to NeuralHawkes.log_likelihood
and z_at_events is added to NeuralHawkes.forward() output. Covers
backward-compat, value change under non-zero lambda, finite output,
and gradient flow to mark-head parameters.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Expose `z_at_events` in NeuralHawkes.forward()

**Files:**
- Modify: `src/eonet_cascades/models/neural_hawkes.py`

- [ ] **Step 2.1: Find the forward() method**

```bash
grep -n "def forward\|def _lambda_k\|lam_k = self._lambda_k\|return {" /Users/liamschmidt/Projects/eonet-cascades/src/eonet_cascades/models/neural_hawkes.py | head -15
```

You should see `def forward(` and inside it `lam_k = self._lambda_k(h_at_t)` (around line 99), plus a final `return {` block (around line 115).

- [ ] **Step 2.2: Refactor to compute raw `z` separately, expose it**

Find this block inside `forward()`:

```python
            lam_k = self._lambda_k(h_at_t)  # (1, n_marks)
            log_lam_at_obs = torch.log(lam_k[0, marks[i]])  # scalar
```

Replace with:

```python
            z_at_t = self.W_lambda_k(h_at_t)  # (1, n_marks) raw logits
            lam_k = torch.nn.functional.softplus(z_at_t).clamp_min(1e-12)
            log_lam_at_obs = torch.log(lam_k[0, marks[i]])  # scalar
```

Then find the lists initialization near the top of forward():

```python
        log_lambda_k_list: list[torch.Tensor] = []
        log_p_x_list: list[torch.Tensor] = []
        h_event_list: list[torch.Tensor] = []
```

Add a fourth list AFTER `h_event_list`:

```python
        log_lambda_k_list: list[torch.Tensor] = []
        log_p_x_list: list[torch.Tensor] = []
        h_event_list: list[torch.Tensor] = []
        z_event_list: list[torch.Tensor] = []
```

Inside the loop, find where the other lists are appended (right after the `log_p_x_list.append(...)` line, around line 108):

```python
            log_lambda_k_list.append(log_lam_at_obs)
            log_p_x_list.append(log_p_x_i.squeeze())
            h_event_list.append(h_at_t.squeeze(0))
```

Add a fourth append:

```python
            log_lambda_k_list.append(log_lam_at_obs)
            log_p_x_list.append(log_p_x_i.squeeze())
            h_event_list.append(h_at_t.squeeze(0))
            z_event_list.append(z_at_t.squeeze(0))
```

Finally, modify the return dict to include `z_at_events`:

```python
        return {
            "log_lambda_k_at_event": torch.stack(log_lambda_k_list),
            "log_p_x": torch.stack(log_p_x_list),
            "h_at_events": torch.stack(h_event_list),
            "z_at_events": torch.stack(z_event_list),
        }
```

- [ ] **Step 2.3: Run Task 1 test to confirm z_at_events surface is right**

```bash
uv run pytest tests/test_aux_mark_loss.py::test_forward_returns_z_at_events_with_correct_shape -v 2>&1 | tail -8
```

Expected: PASS.

- [ ] **Step 2.4: Run existing test suite; confirm no regressions**

```bash
uv run pytest tests/test_neural_hawkes.py tests/test_attribution.py tests/test_mark_rebalance.py tests/test_mark_head_mlp.py -v 2>&1 | tail -25
```

Expected: all existing tests pass. The Tier 1.5/MLP tests use the existing forward() output keys; the new `z_at_events` key is additive.

- [ ] **Step 2.5: Lint check**

```bash
uv run ruff check src/eonet_cascades/models/neural_hawkes.py 2>&1 | tail -3
```

Expected: `All checks passed!`.

- [ ] **Step 2.6: Commit**

```bash
git add src/eonet_cascades/models/neural_hawkes.py
git commit -m "feat(neural_hawkes): expose raw mark-head logits z_at_events in forward()

Refactors forward() to compute z = W_lambda_k(h) before softplus, then
softplus(z) for rates as before. The raw logits z are now exposed in
the forward() output dict under the new 'z_at_events' key, shape
(n_events, n_marks). All existing keys preserved.

Enables the auxiliary mark-classification loss for H4: softmax(z)
gives the categorical distribution over marks, and shift-invariance of
softmax keeps the composition gradient cleanly separate from the rate
gradient (which flows through softplus(z)).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `aux_lambda` to `NeuralHawkes.log_likelihood()`

**Files:**
- Modify: `src/eonet_cascades/models/neural_hawkes.py`

- [ ] **Step 3.1: Add `aux_lambda` parameter to signature**

Find:

```python
    def log_likelihood(
        self,
        times: torch.Tensor,
        lons: torch.Tensor,
        lats: torch.Tensor,
        marks: torch.Tensor,
        window: tuple[float, float],
        n_mc_samples: int = 20,
        mark_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
```

Replace with:

```python
    def log_likelihood(
        self,
        times: torch.Tensor,
        lons: torch.Tensor,
        lats: torch.Tensor,
        marks: torch.Tensor,
        window: tuple[float, float],
        n_mc_samples: int = 20,
        mark_weights: torch.Tensor | None = None,
        aux_lambda: float = 0.0,
    ) -> torch.Tensor:
```

- [ ] **Step 3.2: Add aux-loss computation in the body**

Find the existing computation in `log_likelihood`:

```python
        out = self.forward(times, lons, lats, marks)
        per_event = out["log_lambda_k_at_event"] + out["log_p_x"]
        if mark_weights is not None:
            w = mark_weights.to(per_event.device).index_select(0, marks)
            per_event = per_event * w
        sum_per_event = per_event.sum()
```

Just AFTER `sum_per_event = per_event.sum()`, INSERT:

```python
        # H4 auxiliary mark-classification loss (cross-entropy on softmax(z)).
        # Provides explicit gradient on RELATIVE z magnitudes (mark composition).
        # Softmax is shift-invariant in z, so this does not affect the rate
        # gradient which flows through softplus(z). Eval should always pass
        # aux_lambda=0.0 to keep val NLL comparable to the original Tier 1.
        if aux_lambda != 0.0:
            z = out["z_at_events"]  # (N, K) raw logits
            log_p_mark = torch.nn.functional.log_softmax(z, dim=-1)
            log_p_obs = log_p_mark.gather(1, marks.unsqueeze(1)).squeeze(1)  # (N,)
            aux_term = log_p_obs.sum()  # SUM of log P(observed mark | h)
            # Subtracted from per-event sum: we want to MAXIMIZE this log-prob,
            # so it adds (positively) to log_likelihood. The training loop
            # negates log_likelihood to get a minimizable loss, where this
            # term becomes -aux_lambda * Σ log P(k_obs|h) -- the cross-entropy.
            sum_per_event = sum_per_event + aux_lambda * aux_term
```

Verify the rest of the function is unchanged: the integral computation and
final return.

- [ ] **Step 3.3: Run the failing Task 1 tests; all should now pass**

```bash
uv run pytest tests/test_aux_mark_loss.py -v 2>&1 | tail -15
```

Expected: **5 passed**.

- [ ] **Step 3.4: Run full existing test suite; confirm no regressions**

```bash
uv run pytest tests/ --ignore=tests/test_neural_training.py 2>&1 | tail -10
```

Expected: all tests pass except the known pre-existing `test_neural_recovery` failure.

- [ ] **Step 3.5: Lint check**

```bash
uv run ruff check src/eonet_cascades/models/neural_hawkes.py 2>&1 | tail -3
```

Expected: `All checks passed!`.

- [ ] **Step 3.6: Commit**

```bash
git add src/eonet_cascades/models/neural_hawkes.py
git commit -m "feat(neural_hawkes): aux_lambda mark-classification loss in log_likelihood

Adds optional aux_lambda: float = 0.0 parameter to log_likelihood. When
> 0, augments the joint Hawkes log-likelihood with
  aux_lambda * Σ_i log softmax(z_i)_{k_i_observed}
which adds (positively) to log_likelihood -- training negates and
gets -aux_lambda * cross_entropy(z, marks).

H4 hypothesis: explicit gradient signal on softmax(z) breaks the rank-1
mark-head collapse documented across Tier 1, 1.5, and MLP. See
docs/superpowers/specs/2026-05-26-tier1-aux-mark-loss-design.md.

aux_lambda=0.0 default is backwards-compatible -- existing Tier 1 /
1.5 / MLP eval paths and checkpoints unaffected. All 5 tests in
test_aux_mark_loss.py now pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Wire `aux_lambda` through `train_one_epoch`

**Files:**
- Modify: `src/eonet_cascades/training/neural_loop.py`

- [ ] **Step 4.1: Find `train_one_epoch` signature**

```bash
grep -n "def train_one_epoch\|model.log_likelihood\|mark_weights" /Users/liamschmidt/Projects/eonet-cascades/src/eonet_cascades/training/neural_loop.py
```

- [ ] **Step 4.2: Add `aux_lambda` parameter**

Find the current `train_one_epoch` signature:

```python
def train_one_epoch(
    model: NeuralHawkes,
    chunks: Iterable[TrainChunk],
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | None = None,
    grad_clip: float = 1.0,
    device: str = "cpu",
    mark_weights: torch.Tensor | None = None,
) -> dict[str, float]:
```

Replace with:

```python
def train_one_epoch(
    model: NeuralHawkes,
    chunks: Iterable[TrainChunk],
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | None = None,
    grad_clip: float = 1.0,
    device: str = "cpu",
    mark_weights: torch.Tensor | None = None,
    aux_lambda: float = 0.0,
) -> dict[str, float]:
```

- [ ] **Step 4.3: Pass `aux_lambda` to `log_likelihood`**

Find the call inside the loop:

```python
        ll = model.log_likelihood(
            times, lons, lats, marks, chunk.window, mark_weights=mark_weights
        )
```

Replace with:

```python
        ll = model.log_likelihood(
            times, lons, lats, marks, chunk.window,
            mark_weights=mark_weights,
            aux_lambda=aux_lambda,
        )
```

- [ ] **Step 4.4: Verify all training tests still pass**

```bash
uv run pytest tests/test_neural_training.py tests/test_mark_rebalance.py tests/test_aux_mark_loss.py -v 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 4.5: Lint check**

```bash
uv run ruff check src/eonet_cascades/training/neural_loop.py 2>&1 | tail -3
```

- [ ] **Step 4.6: Commit**

```bash
git add src/eonet_cascades/training/neural_loop.py
git commit -m "feat(neural_loop): thread aux_lambda through train_one_epoch

train_one_epoch gains aux_lambda: float = 0.0 parameter, passed through
to NeuralHawkes.log_likelihood. Eval loop (_tier1_eval_loop in cli.py)
intentionally does NOT use it -- val NLL stays comparable to the
original Tier 1 baseline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add `--aux-lambda` CLI flag

**Files:**
- Modify: `src/eonet_cascades/cli.py`

- [ ] **Step 5.1: Add the Typer option**

Find the last existing flag on `train-neural-hawkes` (it's `mark_head`):

```python
    mark_head: Annotated[
        str,
        typer.Option(
            help=(
                "Mark-intensity head architecture. 'linear' is the original Tier 1 "
                "single nn.Linear head. 'mlp' is a 2-layer ReLU MLP "
                "(H -> H//2 -> n_marks) added 2026-05-26 to test whether non-linear "
                "capacity breaks the rank-1 collapse documented in tier1_5-result.md."
            )
        ),
    ] = "linear",
) -> None:
```

ADD a new option BEFORE the closing `) -> None:`:

```python
    mark_head: Annotated[
        ...
    ] = "linear",
    aux_lambda: Annotated[
        float,
        typer.Option(
            "--aux-lambda",
            help=(
                "Coefficient for the auxiliary mark-classification loss "
                "(softmax cross-entropy on raw mark-head logits z). Default 0.0 "
                "preserves existing Tier 1 / 1.5 / MLP behavior. Set to 1.0 to "
                "enable the H4 experiment (docs/superpowers/specs/"
                "2026-05-26-tier1-aux-mark-loss-design.md). Eval reports pure "
                "Hawkes NLL regardless of this value."
            ),
        ),
    ] = 0.0,
) -> None:
```

- [ ] **Step 5.2: Thread `aux_lambda` to `train_one_epoch`**

Find the existing call to `train_one_epoch` in the CLI:

```python
        train_info = train_one_epoch(
            model,
            train_chunks,
            optimizer,
            scheduler,
            device=device,
            mark_weights=mark_weights,
        )
```

Replace with:

```python
        train_info = train_one_epoch(
            model,
            train_chunks,
            optimizer,
            scheduler,
            device=device,
            mark_weights=mark_weights,
            aux_lambda=aux_lambda,
        )
```

- [ ] **Step 5.3: Persist `aux_lambda` in saved checkpoint config**

Find the `"config"` dict in the checkpoint save (there may be two saves — `checkpoint_best.pt` and `checkpoint_final.pt`):

```bash
grep -n '"config"' /Users/liamschmidt/Projects/eonet-cascades/src/eonet_cascades/cli.py
```

For EACH occurrence, find the `"mark_head": mark_head,` line and add `"aux_lambda": aux_lambda,` immediately after it. Final block looks like:

```python
            "config": {
                "since": since,
                "until": until,
                "val_until": val_until,
                "hidden_dim": hidden_dim,
                "mark_head": mark_head,
                "aux_lambda": aux_lambda,
                "n_marks": n_marks,
```

- [ ] **Step 5.4: Console-print aux_lambda when non-zero**

Find the existing console.print for `mark_rebalance`:

```python
        console.print(
            f"Mark rebalance ({rebalance_mode}): weights = "
            + ", ".join(f"{m}={mark_weights[i].item():.3f}" for i, m in enumerate(mark_names))
        )
```

Right AFTER the surrounding `else:` block that handles `mark_weights = None`, add:

```python
    if aux_lambda != 0.0:
        console.print(
            f"Auxiliary mark-classification loss enabled with aux_lambda={aux_lambda:.3f}. "
            "Eval reports pure Hawkes NLL (aux_lambda=0.0)."
        )
```

This makes the H4 experiment self-documenting in the training log.

- [ ] **Step 5.5: Verify the flag appears in --help**

```bash
uv run eonet model train-neural-hawkes --help 2>&1 | grep -A 6 "aux-lambda"
```

Expected: a paragraph mentioning `auxiliary mark-classification loss`. If absent, the Typer arg wasn't picked up — re-check Step 5.1.

- [ ] **Step 5.6: Lint check**

```bash
uv run ruff check src/eonet_cascades/cli.py 2>&1 | tail -3
```

- [ ] **Step 5.7: Commit**

```bash
git add src/eonet_cascades/cli.py
git commit -m "feat(cli): --aux-lambda flag on train-neural-hawkes

Adds --aux-lambda FLOAT (default 0.0) option that enables the H4
auxiliary mark-classification loss when > 0. Threads through to
train_one_epoch, persists in saved checkpoint config dict on both
best and final checkpoints. Console prints when non-zero so the H4
experiment is self-documenting in training logs.

Eval (_tier1_eval_loop) does not use this flag -- val NLL reported in
train_curves.csv stays pure Hawkes NLL and comparable to the 4.20
baseline. The H4 contribution is implicit in the train_nll column
when aux_lambda > 0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Local smoke test + runbook + push

**Files:**
- Modify: `docs/notes/tier1-cloud-runbook.md`

- [ ] **Step 6.1: Confirm DuckDB is reachable**

```bash
ls -lh /Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb
```

Expected: a file in the ~1.1 GB range. If missing, mount the drive before continuing.

- [ ] **Step 6.2: Run the CLI smoke test with `--aux-lambda 1.0 --mark-head mlp`**

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
uv run eonet model train-neural-hawkes \
  --since 2024-01-01 --until 2024-02-01 \
  --val-until 2024-02-15 \
  --sample 5000 \
  --n-epochs 1 \
  --hidden-dim 32 \
  --device cpu \
  --mark-head mlp \
  --aux-lambda 1.0 \
  --out-dir runs/tier1_aux/smoketest 2>&1 | tail -25
```

Expected (1-2 min on CPU):
- "K = N marks" line
- "Auxiliary mark-classification loss enabled with aux_lambda=1.000" line
- One epoch row `{'epoch': 0, ...}`
- "Saved checkpoints + curves to runs/tier1_aux/smoketest"

If a traceback appears, STOP and diagnose.

- [ ] **Step 6.3: Verify checkpoint persists aux_lambda**

```bash
uv run python -c "
import torch
ckpt = torch.load('runs/tier1_aux/smoketest/checkpoint_best.pt', weights_only=False)
print('aux_lambda:', ckpt['config'].get('aux_lambda'))
print('mark_head:', ckpt['config'].get('mark_head'))
"
```

Expected:
```
aux_lambda: 1.0
mark_head: mlp
```

- [ ] **Step 6.4: Clean up smoke test artifacts**

```bash
rm -rf runs/tier1_aux/smoketest
```

- [ ] **Step 6.5: Update runbook with Step 5‴ section**

Find the end of `docs/notes/tier1-cloud-runbook.md` (it ends with the H2 fallback line from the H3 section). Append:

```markdown

---

## Tier 1 with auxiliary mark loss — H4 experiment (added 2026-05-26)

H3's MLP head (`docs/notes/tier1_mlp-result.md`) achieved the best val
NLL of any tier (3.38) but failed the primary acceptance criterion
(forward-sim probe row-deviation 0.0000). The collapse is robust to
both class rebalancing (Tier 1.5) and mark-head architecture (MLP).

H4 from the analysis: the joint Hawkes NLL has insufficient gradient
signal on mark composition. This experiment adds an explicit
cross-entropy auxiliary loss via the `--aux-lambda` flag.

Spec: `docs/superpowers/specs/2026-05-26-tier1-aux-mark-loss-design.md`.

**Workflow:** repeat Steps 1–3 (provision + bootstrap + DuckDB
transfer). Skip Step 5, Step 5′, and Step 5″. Use Step 5‴ below.

### Step 5‴ — Launch the H4 training run

```bash
# On the cloud instance:
cd ~/eonet-cascades
git pull   # ensure the --aux-lambda flag is present
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

`--mark-rebalance` and `--stratify-train` are intentionally OMITTED.
Single-variable test: the only change from Tier 1-MLP is the addition
of `--aux-lambda 1.0`.

Expected: same ~5h 20m wall time as Tier 1-MLP. Eval NLL reported in
`train_curves.csv` is pure Hawkes NLL, directly comparable to Tier 1's
4.20.

### Acceptance — what to check after pulling results back

```bash
# On the Mac, after Step 7 pulls runs/tier1_aux/<ts>/ down:
LATEST_AUX=$(ls -t runs/tier1_aux/ | head -1)

# 1) val NLL within ~5% of Tier 1's 4.20:
tail -n 1 runs/tier1_aux/$LATEST_AUX/train_curves.csv
# Pass: val_nll <= 4.41

# 2) forward-sim probe row-deviation > 0.1:
# Edit scripts/probe_forward_sim.py RUN_DIR to the new checkpoint, then:
uv run python scripts/probe_forward_sim.py
# Pass: total |row - row_mean| > 0.1 in either Seed A or Seed B
# (was 0.0023 Tier 1, 0.0000 Tier 1.5, 0.0000 Tier 1-MLP)

# 3) Re-render the cross-tier notebook:
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/03_tier0_vs_tier1.ipynb
```

Decision table:

| outcome | next step |
|---------|----------|
| (1) + (2) both pass | H4 confirmed; the model works; pivot to writeup with a clean positive story. |
| (1) fails, (2) passes | Aux loss is too strong; re-run with `--aux-lambda 0.1`. ~$5 follow-up. |
| (2) fails | H4 ruled out; H5 (joint-Hawkes loss is the wrong objective on this data) essentially proved by exclusion. Pivot to writeup of the negative chain as a methodological finding. Draft `docs/notes/mark-head-collapse-chain.md` consolidating all four runs. |
```

- [ ] **Step 6.6: Commit runbook**

```bash
git add docs/notes/tier1-cloud-runbook.md
git commit -m "docs(runbook): Step 5-triple-prime - H4 aux-loss launch + acceptance

Adds Step 5-triple-prime section for the H4 auxiliary mark-classification
loss experiment. Uses --aux-lambda 1.0 on top of --mark-head mlp; no
--mark-rebalance, no --stratify-train. Acceptance block lists primary
(probe row-dev > 0.1), secondary (val NLL within 5% of 4.20), and the
decision tree for the three plausible outcomes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6.7: Push everything to origin/main**

```bash
git push origin main 2>&1 | tail -5
```

Expected: 6 commits pushed (failing tests, forward refactor, log_likelihood aux, neural_loop, CLI flag, runbook). If push is rejected, diagnose before retrying — don't `--force`.

---

## Task 7: Hand off to user for cloud run

- [ ] **Step 7.1: Confirm local-side state**

```bash
git log --oneline -8
git status
```

Expected: 6 new commits on top of `88c7e33`. `git status` shows nothing modified except possibly the long-running `notebooks/01_data_exploration.ipynb` file.

- [ ] **Step 7.2: Notify the user**

Tell the user:

> H4 implementation complete. Six new commits on `origin/main`. Same Lambda Labs A10 setup as Tier 1-MLP, use Step 5‴ launch command from the runbook. Cost: ~$5. Wall time: ~5 h.
>
> When the checkpoint lands, I'll run the probe and decide the H4 verdict.

Plan ends. Cloud run is user-driven via the runbook.

---

## Self-Review

(Run by the plan author after writing the plan; fix any issues inline.)

**1. Spec coverage:**
- §1 Motivation → covered as plan header context
- §2 Architecture change → Tasks 2 (forward exposes z) and 3 (log_likelihood aux term)
- §3 CLI surface → Task 5
- §4 Training configuration → Task 6 runbook (Step 5‴ command)
- §5 Acceptance criteria → Task 6 runbook acceptance block + Task 7 handoff notes
- §6 Implementation order → Tasks 1-7 (12-step list from spec all covered)
- §7 Risks → mitigations baked into Task 1 tests (backwards-compat 1.4, gradient flow 1.7) and Task 6 smoke test (5.2-5.3)
- §8 Out of scope → no tasks for those items (correct)
- §9 Definition of done → Tasks 1-7 hit every checkbox in the spec's DoD list except the post-cloud items (which are intentionally post-handoff)

**2. Placeholder scan:** No "TBD" / "TODO" / "fill in details" patterns. Every code block is complete. Every command has expected output.

**3. Type consistency:** `aux_lambda: float` everywhere. `z_at_events` shape `(N, K)` everywhere. State-dict keys (`W_lambda_k.0.weight`) consistent with the prior Tier 1-MLP plan.

**4. Step granularity:** All code-changing steps include the full text. Test code is complete and runnable. Bash commands are exact with expected outputs.
