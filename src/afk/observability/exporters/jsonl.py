"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

JSONL exporter for append-only run metrics envelopes.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from ..models import RunMetrics
from ..projectors.run_metrics import run_metrics_schema_version


class JSONLRunMetricsExporter:
    """Append run metrics envelopes to JSONL file with process-local lock."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def export(self, metrics: RunMetrics) -> None:
        payload = {
            "schema_version": run_metrics_schema_version(),
            "reported_at": time.time(),
            "metrics": metrics.to_dict(),
        }
        line = json.dumps(payload, ensure_ascii=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """Read all exported JSONL envelopes from exporter file path."""

        if not self._path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip()
                if not row:
                    continue
                out.append(json.loads(row))
        return out
