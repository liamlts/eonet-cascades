# Tier 1 with MLP Mark Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `nn.Linear` mark head in NeuralHawkes with an MLP (`64 → 32 → 8 ReLU`) behind a backward-compatible `mark_head` flag, then run a single-variable Tier 1 retrain on Lambda Labs A10 to test whether non-linear capacity breaks the rank-1 mark-head collapse documented in `docs/notes/tier1_5-result.md`.

**Architecture:** Single constructor argument `mark_head: str = "linear"` on `NeuralHawkes`. The "linear" branch is bit-identical to current behavior (preserves existing checkpoints). The "mlp" branch constructs `nn.Sequential(Linear(H, H//2), ReLU(), Linear(H//2, K))`. The string is persisted in the checkpoint config so the probe script and downstream loaders can reconstruct the matching architecture.

**Tech Stack:** Python 3.11, PyTorch 2.2, Typer CLI, pytest, uv for env management.

**Spec:** [docs/superpowers/specs/2026-05-26-tier1-mlp-mark-head-design.md](../specs/2026-05-26-tier1-mlp-mark-head-design.md)

---

## Task 1: Write all mark-head tests (failing)

**Files:**
- Create: `tests/test_mark_head_mlp.py`

- [ ] **Step 1.1: Create the test file with imports and shared fixtures**

```python
"""Tests for the mark_head constructor argument added in commit 40cf6d3 spec.

Surfaces tested:
  * NeuralHawkes(mark_head="linear") is bit-identical to the current default.
  * NeuralHawkes(mark_head="mlp") builds an MLP head with expected shape.
  * log_likelihood returns finite scalars under both modes.
  * Unknown mark_head values raise ValueError at construction time.
  * mark_head round-trips through state_dict + config dict save/load.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from eonet_cascades.models.neural_hawkes import NeuralHawkes


def _small_inputs(n_marks: int = 3, n_events: int = 20, seed: int = 0):
    """Reusable synthetic event sequence for forward-pass tests."""
    rng = np.random.default_rng(seed)
    times = torch.tensor(np.sort(rng.uniform(0.0, 20.0, size=n_events)), dtype=torch.float32)
    lons = torch.tensor(rng.uniform(-10.0, 10.0, size=n_events), dtype=torch.float32)
    lats = torch.tensor(rng.uniform(-5.0, 5.0, size=n_events), dtype=torch.float32)
    marks = torch.tensor(rng.integers(0, n_marks, size=n_events), dtype=torch.long)
    return times, lons, lats, marks
```

Write to file `tests/test_mark_head_mlp.py`.

- [ ] **Step 1.2: Add test for backward-compat (linear default is bit-identical)**

Append to `tests/test_mark_head_mlp.py`:

```python
def test_linear_mark_head_default_is_bit_identical():
    """NeuralHawkes() and NeuralHawkes(mark_head='linear') must be bit-equal.

    Same seed before construction → same param initialization → same forward
    output on the same input. Guards against accidentally shifting RNG state.
    """
    torch.manual_seed(0)
    m_default = NeuralHawkes(
        n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2
    )

    torch.manual_seed(0)
    m_explicit = NeuralHawkes(
        n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="linear",
    )

    # All parameters must match exactly.
    for (n1, p1), (n2, p2) in zip(
        m_default.named_parameters(), m_explicit.named_parameters(), strict=True
    ):
        assert n1 == n2, f"parameter name mismatch: {n1} vs {n2}"
        assert torch.equal(p1, p2), f"parameter {n1} differs between default and explicit"

    times, lons, lats, marks = _small_inputs(n_marks=3, n_events=20, seed=0)
    out_default = m_default(times, lons, lats, marks)
    out_explicit = m_explicit(times, lons, lats, marks)
    assert torch.equal(
        out_default["log_lambda_k_at_event"], out_explicit["log_lambda_k_at_event"]
    )
```

- [ ] **Step 1.3: Add test for MLP head constructor**

Append to `tests/test_mark_head_mlp.py`:

```python
def test_mlp_mark_head_constructs_with_expected_shape():
    """NeuralHawkes(mark_head='mlp') builds an nn.Sequential head with the
    correct (hidden_dim → hidden_dim // 2 → n_marks) shape."""
    model = NeuralHawkes(
        n_marks=8, hidden_dim=64, mark_emb_dim=8, spatial_emb_dim=8, n_mix=2,
        mark_head="mlp",
    )

    head = model.W_lambda_k
    assert isinstance(head, nn.Sequential), f"expected Sequential, got {type(head)}"
    assert len(head) == 3, f"expected 3 sub-modules (Linear, ReLU, Linear), got {len(head)}"
    assert isinstance(head[0], nn.Linear)
    assert head[0].in_features == 64
    assert head[0].out_features == 32  # hidden_dim // 2
    assert isinstance(head[1], nn.ReLU)
    assert isinstance(head[2], nn.Linear)
    assert head[2].in_features == 32
    assert head[2].out_features == 8

    assert model.mark_head == "mlp"
```

- [ ] **Step 1.4: Add test for finite log_likelihood under MLP head**

Append to `tests/test_mark_head_mlp.py`:

```python
def test_mlp_mark_head_log_likelihood_is_finite():
    """log_likelihood under the MLP head returns a finite scalar tensor."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    model.eval()

    times, lons, lats, marks = _small_inputs(n_marks=3, n_events=20, seed=0)
    ll = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0))
    assert ll.dim() == 0, f"expected scalar, got shape {tuple(ll.shape)}"
    assert torch.isfinite(ll), f"non-finite log_likelihood: {ll.item()}"
```

- [ ] **Step 1.5: Add test for invalid mark_head value**

Append to `tests/test_mark_head_mlp.py`:

```python
def test_invalid_mark_head_raises_value_error():
    """Unknown mark_head value raises a clear ValueError, not a silent fallback."""
    import pytest
    with pytest.raises(ValueError, match="unknown mark_head"):
        NeuralHawkes(
            n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
            mark_head="transformer",
        )
```

- [ ] **Step 1.6: Add test for state_dict round-trip preserving MLP head**

Append to `tests/test_mark_head_mlp.py`:

```python
def test_mlp_mark_head_state_dict_round_trip(tmp_path):
    """Save an MLP-head model + config dict, reload it, verify forward output
    matches bit-exactly. Mirrors the CLI's checkpoint save/load pattern."""
    torch.manual_seed(0)
    m_src = NeuralHawkes(
        n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    m_src.eval()

    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "state_dict": m_src.state_dict(),
            "mark_names": ["a", "b", "c"],
            "config": {
                "hidden_dim": 8,
                "mark_emb_dim": 4,
                "spatial_emb_dim": 4,
                "n_mix": 2,
                "n_marks": 3,
                "mark_head": "mlp",
            },
        },
        ckpt_path,
    )

    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["config"]
    assert cfg["mark_head"] == "mlp"

    m_dst = NeuralHawkes(
        n_marks=cfg["n_marks"],
        hidden_dim=cfg["hidden_dim"],
        mark_emb_dim=cfg["mark_emb_dim"],
        spatial_emb_dim=cfg["spatial_emb_dim"],
        n_mix=cfg["n_mix"],
        mark_head=cfg["mark_head"],
    )
    m_dst.load_state_dict(ckpt["state_dict"])
    m_dst.eval()

    times, lons, lats, marks = _small_inputs(n_marks=3, n_events=15, seed=1)
    out_src = m_src(times, lons, lats, marks)
    out_dst = m_dst(times, lons, lats, marks)
    assert torch.equal(out_src["log_lambda_k_at_event"], out_dst["log_lambda_k_at_event"])
```

- [ ] **Step 1.7: Run the tests; verify they all fail with the expected error**

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
export PATH="$HOME/.local/bin:$PATH"
unset DYLD_LIBRARY_PATH
uv run pytest tests/test_mark_head_mlp.py -v 2>&1 | tail -20
```

Expected output: 5 test FAILED, each with `TypeError: ... got an unexpected keyword argument 'mark_head'` (or similar). The bit-identical test (1.2) may actually PASS at this stage because the default `NeuralHawkes()` call works — but the explicit `NeuralHawkes(mark_head="linear")` call fails. So expect 4 failures + 1 pass on test_linear_mark_head_default_is_bit_identical, OR 5 failures if the explicit call short-circuits the test.

- [ ] **Step 1.8: Commit the failing tests**

```bash
git add tests/test_mark_head_mlp.py
git commit -m "test(neural_hawkes): failing tests for mark_head constructor arg

Will pass once mark_head is added to NeuralHawkes.__init__ in the
next commit. Covers backward-compat, MLP head shape, finite NLL,
ValueError on unknown values, and state_dict round-trip."
```

---

## Task 2: Implement `mark_head` in NeuralHawkes

**Files:**
- Modify: `src/eonet_cascades/models/neural_hawkes.py` (lines 31-50)

- [ ] **Step 2.1: Add the `mark_head` argument to `__init__` signature**

Edit `src/eonet_cascades/models/neural_hawkes.py`. Find:

```python
    def __init__(
        self,
        n_marks: int,
        hidden_dim: int = 64,
        mark_emb_dim: int = 16,
        spatial_emb_dim: int = 16,
        n_mix: int = 8,
    ) -> None:
```

Replace with:

```python
    def __init__(
        self,
        n_marks: int,
        hidden_dim: int = 64,
        mark_emb_dim: int = 16,
        spatial_emb_dim: int = 16,
        n_mix: int = 8,
        mark_head: str = "linear",
    ) -> None:
```

- [ ] **Step 2.2: Replace the W_lambda_k construction with a branch**

In `src/eonet_cascades/models/neural_hawkes.py`, find:

```python
        # Per-mark temporal intensity head — replaces the old W_lambda_t (scalar)
        # and W_mark (softmax) pair.
        self.W_lambda_k = nn.Linear(hidden_dim, n_marks)
```

Replace with:

```python
        # Per-mark temporal intensity head. The "linear" branch is the
        # original Tier 1 architecture (single nn.Linear). The "mlp" branch
        # (added 2026-05-26) is a 2-layer ReLU MLP that tests whether
        # non-linear capacity breaks the rank-1 mark-head collapse documented
        # in docs/notes/tier1_5-result.md.
        if mark_head == "linear":
            self.W_lambda_k = nn.Linear(hidden_dim, n_marks)
        elif mark_head == "mlp":
            self.W_lambda_k = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_marks),
            )
        else:
            raise ValueError(
                f"unknown mark_head: {mark_head!r} (expected 'linear' or 'mlp')"
            )
        self.mark_head = mark_head
```

- [ ] **Step 2.3: Run the new tests; verify all 5 pass**

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
uv run pytest tests/test_mark_head_mlp.py -v 2>&1 | tail -15
```

Expected: `5 passed`.

- [ ] **Step 2.4: Run the existing test suite; verify no regressions**

```bash
uv run pytest tests/ -v 2>&1 | tail -25
```

Expected: all tests pass. Specifically:
- `tests/test_neural_hawkes.py` (2 tests) — pass
- `tests/test_attribution.py` (1 test) — pass
- `tests/test_mark_rebalance.py` (6 tests) — pass
- `tests/test_mark_head_mlp.py` (5 tests) — pass

If `test_neural_training.py` (slow) is collected, it should also pass; if it's marked slow and skipped by default, that's fine.

- [ ] **Step 2.5: Lint check**

```bash
uv run ruff check src/ tests/test_mark_head_mlp.py 2>&1 | tail -5
```

Expected: `All checks passed!`. If any errors, fix them inline and re-run.

- [ ] **Step 2.6: Commit the model change**

```bash
git add src/eonet_cascades/models/neural_hawkes.py
git commit -m "feat(neural_hawkes): mark_head constructor arg (linear | mlp)

Adds a backward-compatible mark_head: str = 'linear' arg to
NeuralHawkes.__init__. The 'linear' branch is bit-identical to the
current default (existing Tier 1 / 1.5 checkpoints continue to load).
The 'mlp' branch constructs nn.Sequential(Linear(H, H//2), ReLU(),
Linear(H//2, K)) and is the experimental head for testing whether
non-linear capacity breaks the rank-1 mark-head collapse documented
in docs/notes/tier1_5-result.md.

Stores self.mark_head for checkpoint reload code to read.

All 5 tests in test_mark_head_mlp.py now pass; the existing test
suite (test_neural_hawkes, test_attribution, test_mark_rebalance) is
unaffected."
```

---

## Task 3: CLI integration — `--mark-head` flag

**Files:**
- Modify: `src/eonet_cascades/cli.py` (Tier 1 train command region, lines 162-225 area)

- [ ] **Step 3.1: Add `--mark-head` Typer option to the train command**

Edit `src/eonet_cascades/cli.py`. Find the `train-neural-hawkes` function signature; the last existing flag is `stratify_threshold`. Add `mark_head` AFTER `stratify_threshold` and BEFORE the closing `) -> None:`:

```python
    stratify_threshold: Annotated[
        float,
        typer.Option(
            help=(
                "Mark-frequency fraction below which a mark is forced into the "
                "training subsample whole. Only applies when --stratify-train is set."
            )
        ),
    ] = 0.01,
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

- [ ] **Step 3.2: Pass `mark_head` to the NeuralHawkes constructor**

In the same file, find:

```python
    model = NeuralHawkes(n_marks=n_marks, hidden_dim=hidden_dim).to(device)
```

Replace with:

```python
    model = NeuralHawkes(
        n_marks=n_marks, hidden_dim=hidden_dim, mark_head=mark_head
    ).to(device)
```

- [ ] **Step 3.3: Persist `mark_head` in the saved checkpoint config dict**

Still in `src/eonet_cascades/cli.py`. Find the `torch.save({...}, out / "checkpoint_best.pt")` block; specifically the `"config"` sub-dict. It currently looks like:

```python
            "config": {
                "since": since,
                "until": until,
                "val_until": val_until,
                "hidden_dim": hidden_dim,
                "n_marks": n_marks,
```

Add a `mark_head` entry inside this dict, immediately after `"hidden_dim": hidden_dim,`:

```python
            "config": {
                "since": since,
                "until": until,
                "val_until": val_until,
                "hidden_dim": hidden_dim,
                "mark_head": mark_head,
                "n_marks": n_marks,
```

If the file ALSO writes a `checkpoint_final.pt` with its own config dict, apply the same edit there. (Check by `grep -n '"config"' src/eonet_cascades/cli.py` after the edit — there should be 1 or 2 hits, all updated.)

- [ ] **Step 3.4: Confirm the `--mark-head` flag appears in `--help`**

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
uv run eonet model train-neural-hawkes --help 2>&1 | grep -A 4 "mark-head"
```

Expected output: a paragraph mentioning `'linear'` (original) and `'mlp'` (added 2026-05-26). If the flag doesn't appear, the Typer arg wasn't picked up — re-check Step 3.1.

- [ ] **Step 3.5: Lint check**

```bash
uv run ruff check src/eonet_cascades/cli.py 2>&1 | tail -3
```

Expected: `All checks passed!`.

- [ ] **Step 3.6: Commit the CLI change**

```bash
git add src/eonet_cascades/cli.py
git commit -m "feat(cli): --mark-head flag on train-neural-hawkes

Adds the --mark-head option (linear | mlp, default linear) and threads
it through to the NeuralHawkes constructor + the saved checkpoint
config dict. Existing checkpoints (Tier 1, Tier 1.5) continue to load
unchanged via the default. The mark_head string is now persisted on
save, so downstream loaders (probe script, attribution runners) can
reconstruct the matching architecture without out-of-band knowledge."
```

---

## Task 4: Probe script reads `mark_head` from checkpoint

**Files:**
- Modify: `scripts/probe_forward_sim.py`

- [ ] **Step 4.1: Find the model-construction line in the probe script**

```bash
grep -n "NeuralHawkes(\|model = \|ckpt\[" /Users/liamschmidt/Projects/eonet-cascades/scripts/probe_forward_sim.py | head -10
```

Expected output: a line like `model = NeuralHawkes(n_marks=n_marks, hidden_dim=...)` or similar. Note its line number for the next step.

- [ ] **Step 4.2: Update the construction to read `mark_head` from the checkpoint config**

In `scripts/probe_forward_sim.py`, find the model construction block (likely near where `ckpt = torch.load(...)` happens). It currently looks something like:

```python
    ckpt = torch.load(RUN_DIR / "checkpoint_best.pt", weights_only=False)
    mark_names = ckpt["mark_names"]
    n_marks = len(mark_names)
    model = NeuralHawkes(n_marks=n_marks, hidden_dim=64)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
```

(Inspect the actual code with `sed -n '40,60p' scripts/probe_forward_sim.py` if the structure differs.)

Replace the `model = NeuralHawkes(...)` line with:

```python
    cfg = ckpt.get("config", {})
    mark_head_str = cfg.get("mark_head", "linear")  # default for pre-mark_head checkpoints
    hidden_dim_val = cfg.get("hidden_dim", 64)
    model = NeuralHawkes(
        n_marks=n_marks, hidden_dim=hidden_dim_val, mark_head=mark_head_str
    )
```

The `.get("mark_head", "linear")` fallback handles older checkpoints (Tier 1, Tier 1.5) saved before this field existed.

- [ ] **Step 4.3: Re-run the probe against the existing Tier 1.5 checkpoint as regression check**

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
uv run python scripts/probe_forward_sim.py 2>&1 | tail -10
```

Expected: the probe runs to completion and prints the same "verdict" output as before (Tier 1.5 row-deviation = 0.0000). If it errors, the fallback didn't work — check Step 4.2.

- [ ] **Step 4.4: Commit the probe-script update**

```bash
git add scripts/probe_forward_sim.py
git commit -m "feat(probe): read mark_head from checkpoint config

probe_forward_sim.py now reads the mark_head field from the checkpoint
config (with a 'linear' fallback for pre-2026-05-26 checkpoints). This
lets the same probe script work against both the original Tier 1 / 1.5
checkpoints (linear head) and the new Tier 1 MLP-head checkpoints
without manual code changes per run."
```

---

## Task 5: Local smoke test on Jan 2024 slice

**Files:** none (transient artifacts cleaned up at end)

- [ ] **Step 5.1: Verify the source DuckDB is reachable**

```bash
ls -lh /Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb
```

Expected: a file in the ~1.1 GB range. If missing, the smoke test cannot run — fix mount or path before continuing.

- [ ] **Step 5.2: Run the CLI smoke test with `--mark-head mlp`**

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
  --out-dir runs/tier1_mlp/smoketest 2>&1 | tail -25
```

Expected output (will take ~1–2 min on CPU):
- "Loaded N train events and M val events" lines
- "K = N marks" with the Jan 2024 mark vocabulary
- A single epoch row `{'epoch': 0, 'train_nll': ..., 'val_nll': ..., 'elapsed_s': ...}`
- "Saved checkpoints + curves to runs/tier1_mlp/smoketest"

If any traceback appears, this is a real bug — stop and diagnose.

- [ ] **Step 5.3: Verify the checkpoint persists `mark_head="mlp"` in its config**

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
uv run python -c "
import torch
ckpt = torch.load('runs/tier1_mlp/smoketest/checkpoint_best.pt', weights_only=False)
print('mark_head:', ckpt['config'].get('mark_head'))
print('state_dict keys (mark head only):')
for k in ckpt['state_dict']:
    if 'W_lambda_k' in k:
        print(' ', k, tuple(ckpt['state_dict'][k].shape))
"
```

Expected output:
```
mark_head: mlp
state_dict keys (mark head only):
  W_lambda_k.0.weight  (16, 32)
  W_lambda_k.0.bias    (16,)
  W_lambda_k.2.weight  (N, 16)
  W_lambda_k.2.bias    (N,)
```

Where `N` is the number of marks in the Jan 2024 slice (the second dimension of `W_lambda_k.0.weight` is `hidden_dim // 2 = 16` since we ran with `--hidden-dim 32`).

- [ ] **Step 5.4: Run the probe against the smoke-test checkpoint**

Edit `scripts/probe_forward_sim.py` and temporarily change `RUN_DIR` to point at the smoke-test checkpoint:

```python
RUN_DIR = Path("runs/tier1_mlp/smoketest")
```

Then run:

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
uv run python scripts/probe_forward_sim.py 2>&1 | tail -10
```

Expected: the probe runs to completion and prints a "verdict" block. (The actual values are meaningless — 1 epoch on 5k events isn't a trained model — we're just confirming the load/probe pipeline works end-to-end on an MLP-head checkpoint.)

- [ ] **Step 5.5: Revert the probe RUN_DIR back to the Tier 1.5 checkpoint**

Edit `scripts/probe_forward_sim.py` and restore:

```python
RUN_DIR = Path("runs/tier1_5/20260526_043203")
```

Verify with:

```bash
grep -n "RUN_DIR = " /Users/liamschmidt/Projects/eonet-cascades/scripts/probe_forward_sim.py
```

Expected: `RUN_DIR = Path("runs/tier1_5/20260526_043203")`.

- [ ] **Step 5.6: Clean up the smoke-test artifacts**

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
rm -rf runs/tier1_mlp/smoketest
ls runs/tier1_mlp/ 2>&1 || echo "(empty)"
```

Expected: the directory is gone. (No commit — `runs/` is gitignored, so the smoke test left no traces in the repo.)

---

## Task 6: Update cloud runbook with Step 5″ section

**Files:**
- Modify: `docs/notes/tier1-cloud-runbook.md` (append after the Tier 1.5 section)

- [ ] **Step 6.1: Confirm the runbook structure**

```bash
grep -n "^##\|^### Step" /Users/liamschmidt/Projects/eonet-cascades/docs/notes/tier1-cloud-runbook.md | tail -20
```

Expected: the file has Steps 1–8, then a "Recovery notes" block, then a "Tier 1.5 retrain" section ending with `Step 5'`. Note where the file ends.

- [ ] **Step 6.2: Append the MLP-head section**

Append to `docs/notes/tier1-cloud-runbook.md`:

```markdown

---

## Tier 1 with MLP mark head — H3 experiment (added 2026-05-26)

Tier 1.5's class-rebalance failed to fix the mark-head rank collapse
(commit `d97ae60`). Hypothesis 3 from `docs/notes/tier1_5-result.md`:
the linear mark head `W_lambda_k` has insufficient capacity. This
experiment replaces it with an MLP (`64 → 32 → 8 ReLU`) via the
`--mark-head mlp` flag added in commit `<TASK_3_HASH>`.

Spec: `docs/superpowers/specs/2026-05-26-tier1-mlp-mark-head-design.md`.

**Workflow:** repeat Steps 1–3 (provision, bootstrap, transfer
DuckDB). Skip Step 5 and Step 5′. Use Step 5″ below.

### Step 5″ — Launch the MLP-head training run

```bash
# On the cloud instance:
cd ~/eonet-cascades
git pull   # ensure the --mark-head flag is present in this commit or later
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

Note: `--mark-rebalance` and `--stratify-train` are intentionally
OMITTED. This is a single-variable test of the mark-head architecture
in isolation; rebalance hurt likelihood last time (val NLL 4.20 → 6.80)
and we don't want to confound the H3 result.

Expected: same ~5 h wall time as Tier 1 and Tier 1.5 (15 epochs × ~21
min/epoch). Eval NLL is reported per-event and is directly comparable
to the original Tier 1's 4.20.

### Acceptance — what to check after pulling results back

```bash
# On the Mac, after Step 7 pulls runs/tier1_mlp/<ts>/ down:
LATEST_MLP=$(ls -t runs/tier1_mlp/ | head -1)

# 1) val NLL within ~5% of Tier 1's 4.20:
tail -n 1 runs/tier1_mlp/$LATEST_MLP/train_curves.csv
# Pass: val_nll <= 4.41

# 2) forward-sim probe row-deviation > 0.1:
# Edit scripts/probe_forward_sim.py to point RUN_DIR at the new
# tier1_mlp checkpoint, then:
uv run python scripts/probe_forward_sim.py
# Pass: total |row - row_mean| > 0.1 in either Seed A or Seed B
# (was 0.0023 for Tier 1, 0.0000 for Tier 1.5)

# 3) Re-render the cross-tier notebook (sanity check):
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/03_tier0_vs_tier1.ipynb
```

Decision table:

| outcome | next step |
|---------|----------|
| (1) + (2) both pass | H3 confirmed; pivot to writeup + headline figure rendering |
| (1) fails, (2) passes | Fix-with-cost; weaker but publishable; investigate optimizer / lr |
| (2) fails (whether (1) passes or not) | H3 ruled out; advance to H2 (mark-agnostic spatial head). Draft spec at `docs/superpowers/specs/2026-05-27-tier1-shared-mdn-design.md` before any further compute spend. |
```

(Replace `<TASK_3_HASH>` with the actual short hash of the Task 3 commit before pushing, OR leave as-is — the doc still parses; we'll patch the hash post-merge if needed.)

- [ ] **Step 6.3: Verify the runbook still parses cleanly**

```bash
wc -l /Users/liamschmidt/Projects/eonet-cascades/docs/notes/tier1-cloud-runbook.md
head -1 /Users/liamschmidt/Projects/eonet-cascades/docs/notes/tier1-cloud-runbook.md
tail -3 /Users/liamschmidt/Projects/eonet-cascades/docs/notes/tier1-cloud-runbook.md
```

Expected: file ends with the H2 fallback line; no truncation; total line count grew by ~60-70.

- [ ] **Step 6.4: Commit the runbook**

```bash
git add docs/notes/tier1-cloud-runbook.md
git commit -m "docs(runbook): Step 5\" — MLP mark head launch + acceptance

Adds a Step 5\" section for the H3 MLP-head experiment. Uses the new
--mark-head mlp flag, omits --mark-rebalance and --stratify-train for
a clean single-variable test, writes to runs/tier1_mlp/<ts>/. The
acceptance block lists the three post-run checks (val NLL within 5%
of 4.20, probe row-deviation > 0.1, notebook re-render) and the
decision table for fail modes (H2 spec is the pre-committed fallback)."
```

- [ ] **Step 6.5: Push the full sequence to origin/main**

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
git push origin main 2>&1 | tail -5
```

Expected: 5 commits pushed (failing tests, model change, CLI change, probe script, runbook). If the push is rejected for any reason (out-of-date branch, hook failure), diagnose before retrying — don't `--force`.

---

## Task 7: Hand off to user for cloud run

This task has no code; it's the gate where the engineer transitions back to the user.

- [ ] **Step 7.1: Confirm local-side state**

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
git log --oneline -7
git status
```

Expected: 5 new commits on top of `40cf6d3` (the spec commit). `git status` shows nothing modified except possibly `notebooks/01_data_exploration.ipynb` (which has been modified-but-not-staged throughout this session).

- [ ] **Step 7.2: Notify the user with the exact cloud-launch command**

Tell the user:

> Local-side work is complete. Five new commits on `origin/main`:
> - `test(neural_hawkes): failing tests for mark_head constructor arg`
> - `feat(neural_hawkes): mark_head constructor arg (linear | mlp)`
> - `feat(cli): --mark-head flag on train-neural-hawkes`
> - `feat(probe): read mark_head from checkpoint config`
> - `docs(runbook): Step 5″ — MLP mark head launch + acceptance`
>
> Ready for the cloud run when you are. Spin up a Lambda Labs A10, follow runbook Steps 1–3 (provision + bootstrap + DuckDB transfer) as before, then use the Step 5″ command. Cost: ~$5. Wall time: ~5 hours.
>
> When the checkpoint lands, I'll point the probe at it, decide the H3 verdict, and either pivot to writeup (pass) or draft the H2 spec (fail).

The implementation plan ends here. Subsequent work (probe + decide + writeup or H2 spec) is gated on the cloud-run outcome.

---

## Self-Review

(Run by the plan author after writing the plan; fix any issues inline.)

**1. Spec coverage:**
- §1 Motivation → covered as plan header context
- §2 Architecture → Task 2 implementation
- §3 CLI surface → Task 3
- §4 Training configuration → Task 6 runbook (Step 5″ command)
- §5 Acceptance criteria → Task 6 runbook acceptance block + Task 7 handoff notes
- §6 Implementation order → Tasks 1-7 (matches the spec's 12-step list)
- §7 Risks → mitigations baked into Tasks 1-5 (backward-compat tests, smoke test, time-box)
- §8 Out of scope → no tasks for those items (correct)
- §9 Definition of done → Tasks 1-7 hit every checkbox in the spec's DoD list except the post-cloud items (which are intentionally post-handoff)

**2. Placeholder scan:** No "TBD" / "TODO" / "fill in details" / "implement later" patterns. The one `<TASK_3_HASH>` placeholder is documented as a post-merge patch and the doc still parses without it.

**3. Type consistency:** `mark_head` is `str` everywhere it appears (Task 1 tests, Task 2 model, Task 3 CLI, Task 4 probe). State-dict keys (`W_lambda_k.0.weight` etc.) are consistent between Task 1.3 (shape check), Task 2.2 (construction), and Task 5.3 (verification).

**4. Step granularity:** All code-changing steps include the full text of the change. Test code is complete and runnable. Bash commands are exact with expected outputs.
