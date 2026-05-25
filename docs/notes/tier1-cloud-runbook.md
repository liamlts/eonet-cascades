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

SSH in, install uv, clone the repo, sync the dev + ml extras:

```bash
ssh ubuntu@<INSTANCE_IP>

# On the cloud instance:
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

git clone https://github.com/<YOUR_USERNAME>/eonet-cascades.git
cd eonet-cascades
uv sync --extra dev --extra ml
```

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
