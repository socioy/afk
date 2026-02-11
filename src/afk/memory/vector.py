from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module provides vector similarity utilities and formatting helpers for the AFK memory subsystem.
"""

from typing import Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity and return 0.0 for zero-norm vectors."""
    a_values = np.asarray(a, dtype=np.float64)
    b_values = np.asarray(b, dtype=np.float64)
    if a_values.ndim != 1 or b_values.ndim != 1:
        raise ValueError("Embeddings must be 1D vectors.")
    if a_values.shape[0] != b_values.shape[0]:
        raise ValueError(
            f"Embedding dim mismatch: {a_values.shape[0]} != {b_values.shape[0]}"
        )

    denominator = np.linalg.norm(a_values) * np.linalg.norm(b_values)
    if denominator <= 0.0:
        return 0.0

    return float(np.dot(a_values, b_values) / denominator)


def format_pgvector(vec: Sequence[float]) -> str:
    """Format a Python vector as a pgvector literal string."""
    return "[" + ",".join(f"{float(x):.10g}" for x in vec) + "]"
