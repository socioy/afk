from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Keep unit tests on in-memory backend by default to avoid sqlite worker-thread
# teardown races when many short-lived event loops are created in test helpers.
os.environ.setdefault("AFK_MEMORY_BACKEND", "in_memory")
