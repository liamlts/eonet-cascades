# eonet-cascades

Spatio-temporal point process benchmark suite for natural-hazard event cascades over CONUS + Mexico, 2000–present.

Three model tiers — parametric multivariate Hawkes, Neural Hawkes (continuous-time LSTM), and Transformer Hawkes — share a common likelihood interface and evaluation harness. The learned cross-mark triggering structure is the headline interpretable output: a cascade graph of natural hazards.

## Status

Phase 0 + 1 — bootstrap and data layer. See `docs/superpowers/specs/2026-05-24-eonet-cascade-benchmark-design.md` for the full design.

## Quick start

```bash
uv sync --extra dev
uv run eonet --help
uv run pytest
```

## Data location

Raw catalogs and the harmonized DuckDB store live on an external drive by default:

```
/Volumes/Seagate_Ext/eonet-cascades-data/
```

Override with the `EONET_DATA_ROOT` environment variable or `--data-root` CLI flag.

## Reproduce the headline figure

```bash
make headline
```

(Stub until Phase 6.)
