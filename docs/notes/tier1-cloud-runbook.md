# Tier 1 cloud GPU runbook (Plan 4 Task 12)

Self-contained execution sequence for the Lambda Labs A10 training run.
All commands are quoted verbatim from `docs/superpowers/plans/2026-05-25-tier-1-neural-hawkes.md`
§Task 12 Steps 2–8.

Placeholders — fill in at runtime:
- `<INSTANCE_IP>` — IPv4 of the Lambda Labs instance
- `<YOUR_USERNAME>` — GitHub username that owns the `eonet-cascades` repo

Estimated wall time: ~15–20 hr on a 1× A10.
Estimated cost: ~$15 (≈$0.75 / hr).

---

## Step 1 — Provision the instance (web UI)

1. Sign in to <https://cloud.lambdalabs.com/instances>.
2. Spin up a **GPU 1× A10**, image **Ubuntu 22.04**.
3. Note the public IPv4 → this is `<INSTANCE_IP>`.

---

## Step 2 — Bootstrap the cloud machine

SSH in, install uv + gh, authenticate gh, clone the (private) repo, sync the dev + ml extras:

```bash
ssh ubuntu@<INSTANCE_IP>

# On the cloud instance:
curl -LsSf https://astral.sh/uv/install.sh | sh

# gh CLI for the private clone (Lambda Labs Ubuntu 22.04 doesn't ship gh)
(type -p gh >/dev/null) || (
  sudo mkdir -p -m 755 /etc/apt/keyrings
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
  sudo apt-get update && sudo apt-get install -y gh
)

# Cartopy needs proj + geos C libs; uv wheel install fails without them
sudo apt-get install -y libproj-dev libgeos-dev proj-bin

source ~/.bashrc

gh auth login --hostname github.com --git-protocol https
# Prompts: choose "GitHub.com" → "HTTPS" → "Login with a web browser".
# gh prints a one-time code; open https://github.com/login/device on
# your Mac and paste it. Approves in seconds.

gh repo clone <YOUR_USERNAME>/eonet-cascades
cd eonet-cascades
uv sync --extra dev --extra ml
```

If the cartopy install still fails despite the apt deps, drop it from the
`ml` extra for this run — only the training command itself is needed, not
the plotting stack. Plotting (Task 13 attribution heatmap) runs locally on
the Mac after results are pulled back.

---

## Step 3 — Transfer the DuckDB snapshot from the Mac

In a separate shell on the local Mac:

```bash
ssh ubuntu@<INSTANCE_IP> 'mkdir -p eonet-cascades/data-snapshot'

scp /Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb \
    ubuntu@<INSTANCE_IP>:eonet-cascades/data-snapshot/events.duckdb
```

~200 MB, ~30 s on a decent uplink.

---

## Step 4 — Pre-flight short run

Back on the cloud instance, confirm the model + data load OK with a tiny
1-month / 1-epoch run before committing to the real one:

```bash
# On the cloud instance:
cd ~/eonet-cascades
export EONET_DATA_ROOT=$(pwd)/data-snapshot
mkdir -p data-snapshot/manifests data-snapshot/raw

uv run eonet model train-neural-hawkes \
  --since 2024-01-01 --until 2024-02-01 \
  --val-until 2024-02-15 \
  --sample 10000 \
  --n-epochs 1 \
  --hidden-dim 32 \
  --device cuda
```

Expected: one epoch row printed; checkpoint saved under `runs/tier1/<ts>/`.

If the pre-flight crashes, fix the root cause before launching the
full run — do NOT burn the full $15 budget on a broken run.

---

## Step 5 — Launch the scale-B training run

```bash
# On the cloud instance:
cd ~/eonet-cascades
nohup uv run eonet model train-neural-hawkes \
  --since 2022-01-01 --until 2024-06-30 \
  --val-until 2024-12-31 \
  --sample 200000 \
  --n-epochs 15 \
  --hidden-dim 64 \
  --lr 1e-3 \
  --device cuda \
  > train.log 2>&1 &
```

Expected: ~15–20 hr wall. Final log line announces saved checkpoints under
`runs/tier1/<ts>/`.

---

## Step 6 — Monitor

```bash
# On the cloud instance:
tail -f train.log
```

The instance bills hourly — terminate promptly after the final log line.

---

## Step 7 — Pull results back to the Mac

From the local Mac:

```bash
LATEST_REMOTE=$(ssh ubuntu@<INSTANCE_IP> 'ls -t /home/ubuntu/eonet-cascades/runs/tier1/ | head -1')
mkdir -p ~/Projects/eonet-cascades/runs/tier1
scp -r ubuntu@<INSTANCE_IP>:/home/ubuntu/eonet-cascades/runs/tier1/$LATEST_REMOTE \
       ~/Projects/eonet-cascades/runs/tier1/$LATEST_REMOTE
```

---

## Step 8 — Terminate the Lambda Labs instance

In the Lambda Labs web UI, terminate the instance. The bill keeps running
until you do.

Confirm the final invoice is under $20 before closing the tab.

---

## Recovery notes

- If the pre-flight (Step 4) hits `KeyError` on a rare mark, the train+val
  vocabulary union fix from commit `6cb9925` should have addressed it.
  Confirm that commit is present on the cloud clone (`git log --oneline -5`).
- If MDN training diverges (loss spikes), lower `--lr` to `5e-4` and relaunch.
- If VRAM is tight, reduce `--hidden-dim` to 32 or `--sample` to 100000.

---

## Tier 1.5 retrain — class-rebalanced (added 2026-05-26)

Tier 1's mark head collapsed to outputting the empirical marginal `P(k)`
regardless of input (diagnosed in commit `420d5a3`). Tier 1.5 retrains
with stratified subsampling + inverse-sqrt mark weights (commit
`befb33d`). Same hardware, same hyperparams, three new flags.

**Workflow:** repeat Steps 1–3 (provision, bootstrap, transfer DuckDB)
verbatim, then run the new launch command below. Skip Step 5; use this
in its place. Steps 6–8 (monitor, pull, terminate) work unchanged.

### Step 5′ — Launch the Tier 1.5 training run

```bash
# On the cloud instance:
cd ~/eonet-cascades
git pull   # ensure commit befb33d (Tier 1.5 plumbing) is present
nohup uv run eonet model train-neural-hawkes \
  --since 2022-01-01 --until 2024-06-30 \
  --val-until 2024-12-31 \
  --sample 200000 \
  --n-epochs 15 \
  --hidden-dim 64 \
  --lr 1e-3 \
  --device cuda \
  --mark-rebalance \
  --rebalance-mode inverse-sqrt \
  --stratify-train \
  --stratify-threshold 0.01 \
  --out-dir runs/tier1_5/$(date -u +%Y%m%d_%H%M%S) \
  > train_tier1_5.log 2>&1 &
```

Expected: same ~15–20 hr wall as Tier 1. Eval NLL/event prints alongside
train each epoch; eval is **unweighted** (the rebalance only affects
training), so the printed val NLL is directly comparable to the Tier 1
4.20 from `runs/tier1/20260525_162056/train_curves.csv`.

### Acceptance — what to check after pulling results back

```bash
# On the Mac, after Step 7 pulls runs/tier1_5/<ts>/ down:
LATEST_T15=$(ls -t runs/tier1_5/ | head -1)

# 1) val NLL within ~5% of Tier 1's 4.20:
tail -n 1 runs/tier1_5/$LATEST_T15/train_curves.csv

# 2) forward-sim degeneracy actually broken:
# Edit scripts/probe_forward_sim.py to point RUN_DIR at the new
# tier1_5 checkpoint, then:
uv run python scripts/probe_forward_sim.py
# Pass: total |row - row_mean| > 0.1 (currently ~0.001 on Tier 1).

# 3) Re-render the cross-tier notebook so the headline figure
# reflects the retrained checkpoint:
uv run python scripts/run_task13_v2.py     # regenerate attribution at n=5000
uv run python /tmp/make_tier1_nb.py        # if you still have the generator
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/03_tier0_vs_tier1.ipynb
```

If (1) or (2) fails:
- (1) fails (val NLL much worse): the rebalance is too aggressive. Drop
  `--rebalance-mode` to a softer mode (TODO: add `inverse-quartic` =
  `1/count^0.25`) or skip `--stratify-train` and rely on weights only.
- (2) fails (rows still don't differentiate): the mark-head collapse is
  not purely a class-imbalance artifact. Hypothesis space narrows to (b)
  under-training (try `--n-epochs 30`) or an architecture issue
  (separated mark head — out of this runbook's scope).


---

## Tier 1 with MLP mark head — H3 experiment (added 2026-05-26)

Tier 1.5's class-rebalance failed to fix the mark-head rank collapse
(commit `d97ae60`). Hypothesis 3 from `docs/notes/tier1_5-result.md`:
the linear mark head `W_lambda_k` has insufficient capacity. This
experiment replaces it with an MLP (`64 → 32 → 8 ReLU`) via the
`--mark-head mlp` flag added in commit `a413af1`.

Spec: `docs/superpowers/specs/2026-05-26-tier1-mlp-mark-head-design.md`.

**Workflow:** repeat Steps 1–3 (provision, bootstrap, transfer
DuckDB). Skip Step 5 and Step 5′. Use Step 5″ below. After training,
use Step 7″ (not Step 7) to pull results — Step 7 hardcodes
`runs/tier1/` and would pull the wrong directory.

### Step 5″ — Launch the MLP-head training run

**Pre-flight (cheap; catches CUDA shape errors before the long run).**
Before launching the nohup line below, do a quick 1-epoch sanity run
on the cloud GPU with --mark-head mlp:

```bash
# On the cloud instance:
cd ~/eonet-cascades
git pull   # ensure commit a413af1 (the --mark-head flag) is present
uv run eonet model train-neural-hawkes \
  --since 2024-01-01 --until 2024-02-01 \
  --val-until 2024-02-15 \
  --sample 10000 \
  --n-epochs 1 \
  --hidden-dim 64 \
  --device cuda \
  --mark-head mlp
```

If the pre-flight prints one `{'epoch': 0, ...}` row and `Saved checkpoints
+ curves to runs/tier1/<ts>/` without a traceback, you're cleared. The
local-CPU smoke test caught this same path in commit f892f52, but A10
CUDA can occasionally surface shape errors that CPU doesn't.

Now the full run:

```bash
# On the cloud instance (assuming pre-flight passed):
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

Expected: ~5 h 20 m wall time, matching the Tier 1.5 run (15 epochs ×
~21 min/epoch on the A10). NOTE: the top-of-file header and the
original Tier 1 section quote 15–20 hr — that was a conservative
estimate from before we had real timing data; the actual Tier 1 and
Tier 1.5 runs both completed in ~5–6 hr. Plan for ~$5 cost (~$0.75/hr
× ~5.5 hr). Eval NLL is reported per-event and is directly comparable
to the original Tier 1's 4.20.

### Step 7″ — Pull results back to the Mac

From the local Mac:

```bash
LATEST_REMOTE=$(ssh ubuntu@<INSTANCE_IP> 'ls -t /home/ubuntu/eonet-cascades/runs/tier1_mlp/ | head -1')
mkdir -p ~/Projects/eonet-cascades/runs/tier1_mlp
scp -r ubuntu@<INSTANCE_IP>:/home/ubuntu/eonet-cascades/runs/tier1_mlp/$LATEST_REMOTE \
       ~/Projects/eonet-cascades/runs/tier1_mlp/$LATEST_REMOTE
```

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
# Pass: nbconvert exits 0 (no Python errors during execution).
```

Decision table:

| outcome | next step |
|---------|----------|
| (1) + (2) both pass | H3 confirmed; pivot to writeup + headline figure rendering |
| (1) fails, (2) passes | Fix-with-cost; weaker but publishable; investigate optimizer / lr |
| (2) fails (whether (1) passes or not) | H3 ruled out; advance to H2 (mark-agnostic spatial head). Draft spec at `docs/superpowers/specs/2026-05-27-tier1-shared-mdn-design.md` before any further compute spend. |

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

Expected: same ~5h 20m wall time as Tier 1-MLP. **Val NLL** reported in
`train_curves.csv` is pure Hawkes NLL, directly comparable to Tier 1's
4.20. **train_nll** is the BLENDED loss (Hawkes + 1.0 × cross-entropy)
and is NOT directly comparable to prior runs — a yellow warning prints
to the training log when --aux-lambda > 0.

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
| (1) + (2) both pass | **H4 confirmed**; the model works; pivot to writeup with a clean positive story. |
| (1) fails, (2) passes | Aux loss is too strong; re-run with `--aux-lambda 0.1`. ~$5 follow-up. |
| (2) fails | **H4 ruled out**; H5 (joint-Hawkes loss is the wrong objective on this data) essentially proved by exclusion. Pivot to writeup of the negative chain as a methodological finding. Draft `docs/notes/mark-head-collapse-chain.md` consolidating all four runs. |
