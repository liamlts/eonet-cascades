.PHONY: install test lint format headline ingest

install:
	uv sync --extra dev --extra ml

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

ingest:
	uv run eonet ingest --catalogs eonet,usgs,noaa,firms --since 2000-01-01

headline:
	@echo "Stub — headline figure regeneration lands in Phase 6."
