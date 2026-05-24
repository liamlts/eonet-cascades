"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def eonet_payload() -> dict:
    return json.loads((FIXTURE_DIR / "eonet_sample.json").read_text())
