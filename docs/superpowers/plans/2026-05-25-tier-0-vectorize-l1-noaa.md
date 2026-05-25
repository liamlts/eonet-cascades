# Tier 0 — Vectorize Likelihood, L1 Regularization, NOAA Mark Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Tier 0 parametric Hawkes usable on real-data scale (≥5k events): vectorize the O(N²) Python likelihood into a single numpy block (50–100× speedup), add L1 regularization on α to prevent bound-pegging, and refine the NOAA mark registry so the sparse marks (tropical_cyclone, temperature_extreme, sea_lake_ice) are no longer structurally undercounted.

**Architecture:** Three independent improvements layered on the existing Tier 0 module: (a) a vectorized variant of `hawkes_log_likelihood` that builds the full (N, N) interaction matrix and reduces in one shot — kept side-by-side with the loop version for correctness verification; (b) an `l1_lambda` parameter threaded through `ParametricHawkes.fit` that adds `λ · Σ|α|` to the NLL objective; (c) expanded NOAA-side entries in the mark registry that promote storm-component event types into the umbrella marks.

**Tech Stack:** NumPy (the vectorization is the main piece), SciPy (existing L-BFGS-B), polars (existing), no new deps.

---

## Background — why this plan exists

The first Tier 0 real-data fit (Plan 2 Task 12) revealed three concrete issues:

1. **Speed.** N=500 / K=5 / 80 params took 33+ minutes and didn't fully converge. Profiling traced the hot path to the per-event Python loop in `hawkes_log_likelihood`. With N² Python iterations × O(N) intensity work each, the asymptote is brutal. The same fit at N=5000 would take many hours; at N=50000 it's infeasible.

2. **Bound-pegging.** The unregularized MLE pushed α[earthquake → wildfire] to its 0.95 upper bound on 200 events — a clear non-cascade. L1 regularization on α is the standard fix.

3. **Sparse marks.** The smell-test gate showed `tropical_cyclone: 2`, `temperature_extreme: 1`, `sea_lake_ice: 1` events over 26 years of CONUS+MX data. The NOAA Storm Events catalog records hurricane/tropical-system components under codes like `Storm Surge/Tide`, `Marine Hurricane/Typhoon`, etc. that our registry doesn't map. Auditable + fixable.

---

## File Structure

```
~/Projects/eonet-cascades/
├── src/eonet_cascades/
│   ├── models/
│   │   └── hawkes.py                    # MODIFY: add hawkes_log_likelihood_vectorized; thread l1_lambda
│   ├── data/
│   │   └── marks.py                     # MODIFY: extend NOAA mappings
├── tests/
│   ├── test_hawkes_likelihood.py        # MODIFY: assert vectorized == loop on small inputs
│   ├── test_hawkes_likelihood_perf.py   # NEW: numerical speedup benchmark
│   ├── test_hawkes_fit_l1.py            # NEW: L1 produces sparser alpha than λ=0
│   ├── test_marks.py                    # MODIFY: assert new NOAA mappings + coverage
│   └── test_synthetic_recovery.py       # MODIFY: tighter tolerance now that fits are fast
```

---

## Task 1: Vectorized hawkes_log_likelihood

**Files:**
- Modify: `src/eonet_cascades/models/hawkes.py` (rename existing function, add vectorized variant)
- Test: `tests/test_hawkes_likelihood.py` (add equivalence test)

The math is identical to the loop version. The implementation builds three (N, N) matrices — pairwise Δt, Δx², and (parent_mark, child_mark)-indexed (α, β, σ) lookups — then reduces in one numpy expression.

For N ≤ 5000, this fits comfortably in memory (32 MB for one (N, N) float64 matrix). Larger N requires chunking, which is out of scope here (called out as a future-work note in the function docstring).

- [ ] **Step 1: Write the equivalence test** — appended to `tests/test_hawkes_likelihood.py`

```python
def test_vectorized_matches_loop_on_small_synthetic():
    """Both formulations must agree to ~1e-9 on identical inputs."""
    import numpy as np

    from eonet_cascades.eval.synthetic import simulate_hawkes
    from eonet_cascades.models.hawkes import (
        HawkesParams,
        hawkes_log_likelihood,
        hawkes_log_likelihood_vectorized,
    )

    rng = np.random.default_rng(7)
    n_marks = 3
    p = HawkesParams(
        mu=np.array([0.4, 0.3, 0.2]),
        alpha=np.array([[0.25, 0.05, 0.0], [0.0, 0.30, 0.10], [0.05, 0.0, 0.20]]),
        beta=np.full((n_marks, n_marks), 1.0),
        sigma=np.full((n_marks, n_marks), 1.0),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    events = simulate_hawkes(p, bbox=bbox, t_end=80.0, rng=rng)

    def _uniform_pi(k, x, b):
        ml, mlat, Ml, Mlat = b
        return np.full(x.shape[0], 1.0 / ((Ml - ml) * (Mlat - mlat)))

    ll_loop = hawkes_log_likelihood(p, events, (0.0, 80.0), _uniform_pi, bbox)
    ll_vec = hawkes_log_likelihood_vectorized(p, events, (0.0, 80.0), _uniform_pi, bbox)
    assert abs(ll_loop - ll_vec) < 1e-7, f"loop {ll_loop} vs vectorized {ll_vec}"
```

- [ ] **Step 2: Run, confirm failure**

```bash
export PATH="$HOME/.local/bin:$PATH"
unset DYLD_LIBRARY_PATH
cd /Users/liamschmidt/Projects/eonet-cascades
uv run pytest tests/test_hawkes_likelihood.py::test_vectorized_matches_loop_on_small_synthetic -v
```

Expected: ImportError on `hawkes_log_likelihood_vectorized`.

- [ ] **Step 3: Append `hawkes_log_likelihood_vectorized` to `src/eonet_cascades/models/hawkes.py`**

```python
def hawkes_log_likelihood_vectorized(
    params: HawkesParams,
    events: dict[str, np.ndarray],
    window: tuple[float, float],
    pi_k: SpatialDensityFn,
    bbox: tuple[float, float, float, float],
    spatial_mass_approx_one: bool = True,
) -> float:
    """Vectorized form of `hawkes_log_likelihood`. Mathematically identical.

    Builds three (N, N) matrices — pairwise time delta, squared spatial delta,
    and pair-indexed (alpha, beta, sigma) lookups — then reduces in one
    numpy expression. ~50-100x faster than the loop version for N <= 5000.

    Memory: ~32 MB per (N, N) float64 matrix at N=2000. Up to ~5000 fits in
    a few GB. Larger N requires chunking (Plan 4+).
    """
    t0, t_end = window
    t_arr = events["time"]
    lon_arr = events["lon"]
    lat_arr = events["lat"]
    k_arr = events["mark"].astype(np.int64)
    n = t_arr.shape[0]
    n_marks = params.K

    if n == 0:
        return -float(np.sum(params.mu) * (t_end - t0))

    # Sort by time so the causal mask below is straightforward.
    order = np.argsort(t_arr, kind="stable")
    t_s = t_arr[order]
    lon_s = lon_arr[order]
    lat_s = lat_arr[order]
    k_s = k_arr[order]

    # Per-event baseline at the event's own (location, mark).
    x_all = np.column_stack([lon_s, lat_s])  # (n, 2)
    pi_vals = np.zeros((n, n_marks), dtype=np.float64)
    for kk in range(n_marks):
        pi_vals[:, kk] = pi_k(kk, x_all, bbox)
    baseline_at_event = params.mu[k_s] * pi_vals[np.arange(n), k_s]  # (n,)

    # Pairwise (n, n) deltas. dt[i, j] = t_s[i] - t_s[j].
    dt_mat = t_s[:, None] - t_s[None, :]
    dlon = lon_s[:, None] - lon_s[None, :]
    dlat = lat_s[:, None] - lat_s[None, :]
    d2_mat = dlon * dlon + dlat * dlat
    causal = dt_mat > 0  # j strictly before i

    # Pair-indexed params. For pair (i, j): parent=k_s[j], child=k_s[i].
    parent_idx = k_s[None, :]  # broadcasts to (n, n) along rows
    child_idx = k_s[:, None]   # broadcasts to (n, n) along cols
    alpha_pairs = params.alpha[parent_idx, child_idx]  # (n, n)
    beta_pairs = params.beta[parent_idx, child_idx]
    sigma_pairs = params.sigma[parent_idx, child_idx]

    # Triggering contribution per pair (zeroed on non-causal pairs).
    # Clip dt for the exp to avoid NaN on the non-causal half (gets zeroed anyway).
    dt_safe = np.where(causal, dt_mat, 0.0)
    temporal = beta_pairs * np.exp(-beta_pairs * dt_safe)
    spatial = np.exp(-d2_mat / (2.0 * sigma_pairs * sigma_pairs)) / (
        2.0 * np.pi * sigma_pairs * sigma_pairs
    )
    contrib = np.where(causal, alpha_pairs * temporal * spatial, 0.0)
    # Sum over parents j for each child i → per-event triggering intensity.
    trigger_at_event = contrib.sum(axis=1)

    lam_at_event = baseline_at_event + trigger_at_event
    if np.any(lam_at_event <= 0):
        return -np.inf
    sum_log = float(np.sum(np.log(lam_at_event)))

    # Integrated intensity. Baseline part: ∑_k μ_k * (T - t0).
    integral_baseline = float(np.sum(params.mu) * (t_end - t0))

    # Triggering part: same closed form as the loop version.
    if not spatial_mass_approx_one:
        raise NotImplementedError("Exact spatial mass not implemented in v1")
    decay = np.exp(-params.beta[k_s, :] * (t_end - t_s)[:, None])  # (n, K)
    per_event = params.alpha[k_s, :] * (1.0 - decay)  # (n, K)
    integral_trigger = float(np.sum(per_event))

    return sum_log - integral_baseline - integral_trigger
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
uv run pytest tests/test_hawkes_likelihood.py -v
uv run ruff check .
```

Expected: 4 passed (3 existing + the equivalence test); ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/eonet_cascades/models/hawkes.py tests/test_hawkes_likelihood.py
git commit -m "feat(models): add vectorized hawkes_log_likelihood (NxN matrix form)"
```

---

## Task 2: Performance benchmark test

**Files:**
- Test: `tests/test_hawkes_likelihood_perf.py` (NEW)

Lock in the vectorization benefit as a regression test: at N=300, the vectorized version must be at least 20x faster than the loop.

- [ ] **Step 1: Write the perf test**

`tests/test_hawkes_likelihood_perf.py`:

```python
"""Vectorized vs loop performance regression test."""

from __future__ import annotations

import time

import numpy as np
import pytest

from eonet_cascades.eval.synthetic import simulate_hawkes
from eonet_cascades.models.hawkes import (
    HawkesParams,
    hawkes_log_likelihood,
    hawkes_log_likelihood_vectorized,
)


def _uniform_pi(k, x, b):
    ml, mlat, Ml, Mlat = b
    return np.full(x.shape[0], 1.0 / ((Ml - ml) * (Mlat - mlat)))


@pytest.mark.slow
def test_vectorized_at_least_20x_faster_than_loop():
    rng = np.random.default_rng(3)
    n_marks = 3
    p = HawkesParams(
        mu=np.array([0.5, 0.3, 0.2]),
        alpha=np.array([[0.30, 0.10, 0.00], [0.00, 0.40, 0.15], [0.05, 0.00, 0.20]]),
        beta=np.full((n_marks, n_marks), 1.0),
        sigma=np.full((n_marks, n_marks), 1.0),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    events = simulate_hawkes(p, bbox=bbox, t_end=200.0, rng=rng)
    n = events["time"].shape[0]
    print(f"benchmark on {n} events")

    t0 = time.perf_counter()
    ll_loop = hawkes_log_likelihood(p, events, (0.0, 200.0), _uniform_pi, bbox)
    dt_loop = time.perf_counter() - t0

    t1 = time.perf_counter()
    ll_vec = hawkes_log_likelihood_vectorized(p, events, (0.0, 200.0), _uniform_pi, bbox)
    dt_vec = time.perf_counter() - t1

    print(f"  loop:       {dt_loop:.3f} s   (ll={ll_loop:.4f})")
    print(f"  vectorized: {dt_vec:.3f} s   (ll={ll_vec:.4f})")
    print(f"  speedup:    {dt_loop / dt_vec:.1f}x")
    assert abs(ll_loop - ll_vec) < 1e-6
    assert dt_loop / dt_vec >= 20.0, f"only {dt_loop / dt_vec:.1f}x speedup, expected >=20x"
```

- [ ] **Step 2: Run, confirm pass**

```bash
uv run pytest tests/test_hawkes_likelihood_perf.py -v -m slow -s
```

Expected: PASS with a printed speedup of 30-100x. If the speedup is <20x, the vectorization has a Python-loop hidden somewhere — go look. **Do not lower the 20x threshold without understanding why.**

- [ ] **Step 3: Commit**

```bash
git add tests/test_hawkes_likelihood_perf.py
git commit -m "test(perf): lock in 20x speedup of vectorized log-likelihood"
```

---

## Task 3: Switch ParametricHawkes.fit to use the vectorized likelihood

**Files:**
- Modify: `src/eonet_cascades/models/hawkes.py` (change the `nll` closure in `ParametricHawkes.fit` to call the vectorized version)
- The existing `hawkes_log_likelihood` (loop form) stays for testing only — useful as a reference implementation.

- [ ] **Step 1: Read the existing `fit` method** to find the line that calls `hawkes_log_likelihood`:

```bash
grep -n "hawkes_log_likelihood" src/eonet_cascades/models/hawkes.py
```

Expected: a hit inside `ParametricHawkes.fit`'s `nll` closure (the `_inside_fit` line near the bottom of the file).

- [ ] **Step 2: Swap loop → vectorized in `fit`**

In `src/eonet_cascades/models/hawkes.py`, the `fit` method has a closure like:

```python
def nll(theta: np.ndarray) -> float:
    params = unpack(theta)
    ll = hawkes_log_likelihood(params, events, window, self.pi_k, self.bbox)
    if not np.isfinite(ll):
        return 1e20
    return -ll
```

Change `hawkes_log_likelihood` to `hawkes_log_likelihood_vectorized` on that one line.

- [ ] **Step 3: Also swap the `log_likelihood` instance method**

The class method `ParametricHawkes.log_likelihood(events, window)` calls the loop version. Change it to the vectorized version too. The instance method is what consumers (notebooks, CLI) call.

- [ ] **Step 4: Run all existing tests to confirm nothing regressed**

```bash
uv run pytest -v
uv run ruff check .
```

Expected: all tests pass. The slow `test_synthetic_recovery_within_tolerance` should now run in well under a minute (was 7.5 min).

- [ ] **Step 5: Commit**

```bash
git add src/eonet_cascades/models/hawkes.py
git commit -m "perf(hawkes): use vectorized likelihood inside ParametricHawkes.fit + log_likelihood"
```

---

## Task 4: L1 regularization on α

**Files:**
- Modify: `src/eonet_cascades/models/hawkes.py` (add `l1_lambda` kwarg to `ParametricHawkes.fit`)
- Test: `tests/test_hawkes_fit_l1.py` (NEW)

L1 penalty: `nll_l1(θ) = nll(θ) + λ · ‖α‖₁`. Tested by generating data from a sparse α (many true zeros), then verifying that the L1 fit recovers more zeros than the λ=0 fit.

- [ ] **Step 1: Write the L1 sparsity test**

`tests/test_hawkes_fit_l1.py`:

```python
"""L1 regularization on alpha — sparsity recovery test."""

from __future__ import annotations

import numpy as np
import pytest

from eonet_cascades.eval.synthetic import simulate_hawkes
from eonet_cascades.models.hawkes import HawkesParams, ParametricHawkes


def _uniform_pi(k, x, b):
    ml, mlat, Ml, Mlat = b
    return np.full(x.shape[0], 1.0 / ((Ml - ml) * (Mlat - mlat)))


@pytest.mark.slow
def test_l1_produces_sparser_alpha_than_unregularized():
    """Sparse ground-truth alpha (5 of 9 entries are zero).
    L1 fit should recover more of those zeros than the unregularized fit.
    """
    n_marks = 3
    bbox = (-10.0, -10.0, 10.0, 10.0)
    alpha_true = np.array(
        [
            [0.30, 0.00, 0.00],
            [0.00, 0.40, 0.15],
            [0.00, 0.00, 0.20],
        ]
    )
    truth = HawkesParams(
        mu=np.array([0.4, 0.3, 0.2]),
        alpha=alpha_true,
        beta=np.full((n_marks, n_marks), 1.0),
        sigma=np.full((n_marks, n_marks), 1.0),
    )
    rng = np.random.default_rng(0)
    events = simulate_hawkes(truth, bbox=bbox, t_end=200.0, rng=rng)
    print(f"Generated {events['time'].shape[0]} events")

    threshold = 0.02  # an alpha entry below this counts as a "recovered zero"
    zero_mask = alpha_true < 1e-3
    n_true_zeros = int(zero_mask.sum())

    # Unregularized fit.
    m0 = ParametricHawkes(K=n_marks, bbox=bbox, pi_k=_uniform_pi)
    m0.fit(events, (0.0, 200.0), max_iter=200)
    n_recovered_zeros_0 = int(np.sum(m0.params.alpha[zero_mask] < threshold))
    print(f"Unregularized:  recovered {n_recovered_zeros_0}/{n_true_zeros} zeros")
    print(f"  alpha:\n{m0.params.alpha}")

    # L1 fit.
    m1 = ParametricHawkes(K=n_marks, bbox=bbox, pi_k=_uniform_pi)
    m1.fit(events, (0.0, 200.0), max_iter=200, l1_lambda=0.5)
    n_recovered_zeros_1 = int(np.sum(m1.params.alpha[zero_mask] < threshold))
    print(f"L1 (lambda=0.5): recovered {n_recovered_zeros_1}/{n_true_zeros} zeros")
    print(f"  alpha:\n{m1.params.alpha}")

    assert n_recovered_zeros_1 >= n_recovered_zeros_0, (
        f"L1 recovered {n_recovered_zeros_1} zeros vs unregularized {n_recovered_zeros_0}"
    )
    assert n_recovered_zeros_1 >= 3, (
        f"L1 should recover at least 3/{n_true_zeros} true zeros, got {n_recovered_zeros_1}"
    )
```

- [ ] **Step 2: Run, confirm failure**

```bash
uv run pytest tests/test_hawkes_fit_l1.py -v -m slow -s
```

Expected: `TypeError: ... got an unexpected keyword argument 'l1_lambda'` — the `fit` method doesn't accept it yet.

- [ ] **Step 3: Add `l1_lambda` to `ParametricHawkes.fit` signature and the NLL closure**

In `src/eonet_cascades/models/hawkes.py`, modify the `fit` method:

Change the signature from:
```python
def fit(
    self,
    events: dict[str, np.ndarray] | pl.DataFrame,
    window: tuple[float, float],
    *,
    fix_alpha_zero: bool = False,
    max_iter: int = 200,
) -> dict[str, Any]:
```

to:
```python
def fit(
    self,
    events: dict[str, np.ndarray] | pl.DataFrame,
    window: tuple[float, float],
    *,
    fix_alpha_zero: bool = False,
    max_iter: int = 200,
    l1_lambda: float = 0.0,
) -> dict[str, Any]:
```

Update the `nll` closure (inside `fit`) to add the L1 penalty:

```python
def nll(theta: np.ndarray) -> float:
    params = unpack(theta)
    ll = hawkes_log_likelihood_vectorized(params, events, window, self.pi_k, self.bbox)
    if not np.isfinite(ll):
        return 1e20
    penalty = l1_lambda * float(np.sum(np.abs(params.alpha))) if l1_lambda > 0 else 0.0
    return -ll + penalty
```

And include `l1_lambda` in the returned summary dict:

```python
return {
    "nll_final": float(res.fun),
    "n_iter": int(res.nit),
    "status": "success" if res.success else "failed",
    "message": res.message if isinstance(res.message, str) else res.message.decode("utf-8", "ignore"),
    "spectral_radius": self.params.spectral_radius(),
    "l1_lambda": float(l1_lambda),
}
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
uv run pytest tests/test_hawkes_fit_l1.py -v -m slow -s
uv run pytest tests/test_hawkes_fit.py -v
uv run ruff check .
```

Expected: L1 test passes; existing unregularized fit tests still pass; ruff clean. The L1 fit should recover at least one more zero than the unregularized fit, and at least 3 of 5 true zeros.

- [ ] **Step 5: Commit**

```bash
git add src/eonet_cascades/models/hawkes.py tests/test_hawkes_fit_l1.py
git commit -m "feat(hawkes): add L1 regularization on alpha via l1_lambda kwarg"
```

---

## Task 5: Add l1-lambda flag to train-hawkes CLI

**Files:**
- Modify: `src/eonet_cascades/cli.py` (add `--l1-lambda` option)

- [ ] **Step 1: Add the CLI option**

In `src/eonet_cascades/cli.py`, modify the `model_train_hawkes` signature to add an `l1_lambda` parameter. Find the existing `max_iter` line:

```python
    max_iter: Annotated[int, typer.Option(help="L-BFGS-B max iterations")] = 300,
```

and add a line right after:

```python
    l1_lambda: Annotated[float, typer.Option(help="L1 regularization on alpha (0 = none)")] = 0.0,
```

Then in the function body, find the `model.fit(...)` call:

```python
    result = model.fit(events_dict, (0.0, t_end_days), max_iter=max_iter)
```

and change it to:

```python
    result = model.fit(events_dict, (0.0, t_end_days), max_iter=max_iter, l1_lambda=l1_lambda)
```

- [ ] **Step 2: Verify the CLI registers + tests still pass**

```bash
uv run eonet model train-hawkes --help | grep -E "l1|lambda"
uv run pytest tests/test_cli.py -v
uv run ruff check .
```

Expected: `--l1-lambda FLOAT` shown in help; CLI tests pass; ruff clean.

- [ ] **Step 3: Commit**

```bash
git add src/eonet_cascades/cli.py
git commit -m "feat(cli): add --l1-lambda flag to train-hawkes"
```

---

## Task 6: Audit NOAA mark mappings against the live store

**Files:**
- None — this is a read-only diagnostic step.

Before extending the registry, find the actual `EVENT_TYPE` strings present in NOAA data that currently fall through. We want a data-driven list.

- [ ] **Step 1: Query the store for the top-K unmapped NOAA event_types**

```bash
export PATH="$HOME/.local/bin:$PATH"
unset DYLD_LIBRARY_PATH
cd /Users/liamschmidt/Projects/eonet-cascades

cp /Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb /tmp/events_audit.duckdb
uv run python <<'EOF'
import json
import duckdb
import polars as pl

conn = duckdb.connect("/tmp/events_audit.duckdb", read_only=True)
df = conn.execute("SELECT mark, metadata_json FROM events WHERE source_catalog='noaa'").pl()
# Pull NOAA EVENT_TYPE strings from the metadata_json
types: dict[str, int] = {}
for row in df.iter_rows(named=True):
    meta = json.loads(row["metadata_json"])
    et = (meta.get("event_type") or "").strip()
    types[et] = types.get(et, 0) + 1
# Print the top 40 mapped types so we can see what's already covered
from eonet_cascades.data.marks import _REGISTRY
mapped = set(_REGISTRY.get("noaa", {}).keys())
unmapped_present = sorted(((t, c) for t, c in types.items() if t.lower() not in mapped), key=lambda x: -x[1])[:30]
print("=== UNMAPPED NOAA event_types currently in store, by frequency ===")
for t, c in unmapped_present:
    print(f"  {c:>7}  {t}")
print()
mapped_present = sorted(((t, c) for t, c in types.items() if t.lower() in mapped), key=lambda x: -x[1])[:15]
print("=== Currently-mapped event_types (top 15) ===")
for t, c in mapped_present:
    print(f"  {c:>7}  {t}")
conn.close()
EOF
```

Expected: a table of the most frequent NOAA event types that currently get dropped. Look for plausible additions:

- `Hurricane (Typhoon)`, `Marine Hurricane/Typhoon`, `Marine Tropical Storm`, `Tropical Depression`, `Storm Surge/Tide` → `tropical_cyclone`
- `Heat`, `Excessive Heat`, `Marine Heat Wave` → `temperature_extreme`
- `Cold/Wind Chill`, `Extreme Cold/Wind Chill`, `Frost/Freeze` → `temperature_extreme`
- `Heavy Snow`, `Lake-Effect Snow`, `Ice Storm`, `Sleet`, `Winter Weather` → `sea_lake_ice` (umbrella usage) or → `severe_storm`?

Record what's actually there (the count column is your evidence) and pick which ones to map. **Don't map things that are not actually in the data** — wasted effort.

This step has no commit; the output informs Task 7.

---

## Task 7: Extend the NOAA mark registry

**Files:**
- Modify: `src/eonet_cascades/data/marks.py`
- Test: `tests/test_marks.py` (add per-mapping tests)

Based on the Task 6 audit, add entries that bring the sparse umbrella marks (tropical_cyclone, temperature_extreme, sea_lake_ice) above zero population at full ingest scale.

- [ ] **Step 1: Append tests for the new mappings**

Append to `tests/test_marks.py`:

```python
def test_noaa_hurricane_typhoon_maps_to_tropical_cyclone():
    assert harmonize_mark("noaa", "Hurricane (Typhoon)") == Mark.TROPICAL_CYCLONE
    assert harmonize_mark("noaa", "Marine Hurricane/Typhoon") == Mark.TROPICAL_CYCLONE
    assert harmonize_mark("noaa", "Marine Tropical Storm") == Mark.TROPICAL_CYCLONE
    assert harmonize_mark("noaa", "Storm Surge/Tide") == Mark.TROPICAL_CYCLONE


def test_noaa_temperature_extremes():
    assert harmonize_mark("noaa", "Frost/Freeze") == Mark.TEMPERATURE_EXTREME
    assert harmonize_mark("noaa", "Marine Heat Wave") == Mark.TEMPERATURE_EXTREME


def test_noaa_winter_storms_to_severe_storm():
    # Winter weather components belong under severe_storm umbrella.
    assert harmonize_mark("noaa", "Heavy Snow") == Mark.SEVERE_STORM
    assert harmonize_mark("noaa", "Ice Storm") == Mark.SEVERE_STORM
    assert harmonize_mark("noaa", "Sleet") == Mark.SEVERE_STORM
    assert harmonize_mark("noaa", "Winter Weather") == Mark.SEVERE_STORM


def test_noaa_lake_effect_snow_to_sea_lake_ice():
    assert harmonize_mark("noaa", "Lake-Effect Snow") == Mark.SEA_LAKE_ICE
```

- [ ] **Step 2: Run, confirm failure**

```bash
uv run pytest tests/test_marks.py -v
```

Expected: the new tests fail (the mappings don't exist yet). The existing 8 tests should still pass.

- [ ] **Step 3: Extend `_REGISTRY["noaa"]` in `src/eonet_cascades/data/marks.py`**

Find the `"noaa": {...}` block and add these entries (placement among existing entries is fine):

```python
        # tropical cyclone variants
        "hurricane (typhoon)": Mark.TROPICAL_CYCLONE,
        "marine hurricane/typhoon": Mark.TROPICAL_CYCLONE,
        "marine tropical storm": Mark.TROPICAL_CYCLONE,
        "marine tropical depression": Mark.TROPICAL_CYCLONE,
        "storm surge/tide": Mark.TROPICAL_CYCLONE,
        # temperature extreme variants
        "frost/freeze": Mark.TEMPERATURE_EXTREME,
        "marine heat wave": Mark.TEMPERATURE_EXTREME,
        # winter storm variants → severe_storm
        "heavy snow": Mark.SEVERE_STORM,
        "ice storm": Mark.SEVERE_STORM,
        "sleet": Mark.SEVERE_STORM,
        "winter weather": Mark.SEVERE_STORM,
        # sea/lake ice variants
        "lake-effect snow": Mark.SEA_LAKE_ICE,
```

(Note: `"hurricane (typhoon)"` already exists in the registry from Plan 1, so be careful not to double-add it. Check first with `grep -n "hurricane.typhoon" src/eonet_cascades/data/marks.py`. The new ones are the marine/storm-surge/winter variants.)

- [ ] **Step 4: Run all marks tests, confirm pass**

```bash
uv run pytest tests/test_marks.py -v
uv run ruff check .
```

Expected: all marks tests pass (old + new); ruff clean. The full-coverage sentinel test (`test_all_unified_marks_have_at_least_one_source_mapping`) should still pass — we haven't removed any mappings.

- [ ] **Step 5: Commit**

```bash
git add src/eonet_cascades/data/marks.py tests/test_marks.py
git commit -m "feat(marks): extend NOAA registry — tropical cyclone, temperature extreme, winter, lake-effect coverage"
```

---

## Task 8: Re-ingest NOAA with the new registry (operational)

**Files:**
- None (operational task)

The mark registry only takes effect on **fresh** harmonization. Existing rows in the store keep their old mark assignments until re-ingested. We need to clear NOAA from the store and re-ingest.

- [ ] **Step 1: Wipe NOAA rows from the store + reset its manifest**

Make sure no notebook kernel is holding the DB open first (`pgrep -fl ipykernel`); if so, kill those kernels.

```bash
cd /Users/liamschmidt/Projects/eonet-cascades
export PATH="$HOME/.local/bin:$PATH"
unset DYLD_LIBRARY_PATH

uv run python <<'EOF'
import duckdb
conn = duckdb.connect("/Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb")
n_before = conn.execute("SELECT COUNT(*) FROM events WHERE source_catalog='noaa'").fetchone()[0]
conn.execute("DELETE FROM events WHERE source_catalog='noaa'")
n_after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
print(f"Deleted NOAA: {n_before} rows; store now has {n_after} events total")
conn.close()
EOF

rm /Volumes/Seagate_Ext/eonet-cascades-data/manifests/noaa_state.json
ls /Volumes/Seagate_Ext/eonet-cascades-data/manifests/
```

Expected: NOAA rows deleted; total count drops by ~870k; `noaa_state.json` removed.

- [ ] **Step 2: Re-ingest NOAA only**

```bash
uv run eonet ingest --catalogs noaa --since 2000-01-01
```

Expected: ~5-10 minutes. Final log line should report ~860k–890k NOAA events written (a bit more than before, since we now catch the previously-dropped storm-surge / winter-weather / etc. types).

- [ ] **Step 3: Verify the previously-sparse marks now have meaningful population**

```bash
cp /Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb /tmp/events_check.duckdb
uv run python <<'EOF'
import duckdb
conn = duckdb.connect("/tmp/events_check.duckdb", read_only=True)
df = conn.execute("SELECT mark, COUNT(*) AS n FROM events GROUP BY mark ORDER BY n DESC").pl()
print(df)
EOF
```

Expected: `tropical_cyclone`, `temperature_extreme`, `sea_lake_ice` move from single-digit counts to hundreds or thousands of events. Top of table should still be dominated by wildfire / severe_storm / earthquake / flood.

- [ ] **Step 4: No commit (operational)**

---

## Task 9: Re-fit Tier 0 at usable scale and inspect cascade graph

**Files:**
- None (operational task — produces `runs/tier0/<ts>/`)

With vectorized likelihood + L1 regularization + a fuller NOAA registry, re-run Tier 0 on a meaningful sample.

- [ ] **Step 1: Run the fit**

```bash
export PATH="$HOME/.local/bin:$PATH"
unset DYLD_LIBRARY_PATH
cd /Users/liamschmidt/Projects/eonet-cascades

uv run eonet model train-hawkes \
  --since 2023-01-01 --until 2024-07-01 \
  --sample 5000 \
  --max-iter 300 \
  --l1-lambda 0.1 \
  --seed 0
```

Expected wall time: ~5–15 minutes. With the vectorized likelihood, 5000 events × K marks should be comfortable.

The `--l1-lambda 0.1` is a starting point; revisit if too many entries collapse to zero or none do. Doubling/halving the lambda is the right tuning knob.

- [ ] **Step 2: Inspect the cascade graph**

```bash
LATEST=$(ls -t runs/tier0/ | head -1)
echo "Latest: $LATEST"
cat runs/tier0/$LATEST/alpha.csv | head -25
open runs/tier0/$LATEST/alpha.png   # macOS
```

What to look for:
- Diagonal entries (self-excitation) positive and plausible: wildfire→wildfire (FIRMS clustering), earthquake→earthquake (aftershocks), severe_storm→severe_storm (outbreak clustering).
- Off-diagonal cross-mark triggers: severe_storm → tornado, severe_storm → flood, tropical_cyclone → flood. Each should be physically plausible.
- α[earthquake → wildfire] should now be near 0 (we deliberately avoided sample=200 + unregularized; L1 should kill the spurious peg).

Add 5-10 bullets of observations to `docs/notes/tier0-second-fit.md` (create if needed). What's confirmed? What's surprising? What looks wrong?

- [ ] **Step 3 (optional): Commit the notes file**

```bash
git add docs/notes/tier0-second-fit.md
git commit -m "docs(tier0): observations from L1-regularized vectorized fit"
```

---

## Self-Review

**Spec coverage** (against Plan 2 §"What's next" + observations):

- Item 1 ("Vectorize the likelihood — biggest win") → Tasks 1, 2, 3
- Item 2 ("L1 regularization on α") → Tasks 4, 5
- Item 3 ("Fix sparse-mark NOAA registry") → Tasks 6, 7, 8
- Validation at scale → Task 9

**Placeholder scan:** No `TBD`, `TODO`, "implement later". Two intentional human-judgment steps:
- Task 6 (audit) produces a list whose specific entries depend on what's actually in the store. Task 7 spells out concrete mappings to add; if the audit shows additional types worth mapping, append them. This is a known-and-bounded extension, not a placeholder.
- Task 9 tuning (l1_lambda = 0.1 vs 0.05 vs 0.2): documented as "doubling/halving is the knob," with criteria for what "right" looks like.

**Type/name consistency**: `hawkes_log_likelihood` (loop), `hawkes_log_likelihood_vectorized`, `ParametricHawkes.fit(..., l1_lambda=...)`, `harmonize_mark(catalog, native)`, `Mark.TROPICAL_CYCLONE`, etc. — all match existing signatures.

**Memory consideration documented**: vectorized version uses O(N²) memory; out of scope for chunking is documented in its own docstring with the N=5000 / few-GB number.

**Out of scope, called out for Plan 4+:**
- Chunked likelihood for N > 5000 events.
- Tier 1 — Neural Hawkes — has its own plan after this one lands.
- Cross-catalog dedup as a separate post-ingest pass (carried from Plan 1).
- Spatial Gaussian-mass integral over bbox (Assumption A1 remains).
