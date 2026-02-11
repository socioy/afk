from __future__ import annotations

import math
from typing import Iterable, Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity and return 0.0 for zero-norm vectors."""
    if len(a) != len(b):
        raise ValueError(f"Embedding dim mismatch: {len(a)} != {len(b)}")

    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)

    if na <= 0.0 or nb <= 0.0:
        return 0.0

    return dot / (math.sqrt(na) * math.sqrt(nb))


def format_pgvector(vec: Sequence[float]) -> str:
    """Format a Python vector as a pgvector literal string."""
    return "[" + ",".join(f"{float(x):.10g}" for x in vec) + "]"
