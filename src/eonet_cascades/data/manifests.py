"""Per-catalog ingestion state stored as JSON in the manifests directory."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ManifestStore:
    """Tracks the last successful fetch timestamp per catalog."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, catalog: str) -> Path:
        return self.root / f"{catalog}_state.json"

    def last_fetched(self, catalog: str) -> datetime | None:
        p = self._path(catalog)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        ts = data.get("last_fetched")
        return datetime.fromisoformat(ts) if ts else None

    def set_last_fetched(self, catalog: str, ts: datetime) -> None:
        p = self._path(catalog)
        p.write_text(json.dumps({"last_fetched": ts.isoformat()}))
