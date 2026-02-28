"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Pluggable assertion and scoring contracts for eval execution.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from .models import EvalAssertionResult, EvalCase, EvalCaseResult


class EvalAssertion(Protocol):
    """Contract for one eval assertion callable."""

    name: str

    def __call__(self, case: EvalCase, result: EvalCaseResult) -> EvalAssertionResult:
        """Return assertion result for one case/result tuple."""
        ...


class AsyncEvalAssertion(Protocol):
    """Contract for one async eval assertion callable."""

    name: str

    def __call__(
        self,
        case: EvalCase,
        result: EvalCaseResult,
    ) -> Awaitable[EvalAssertionResult]:
        """Return awaitable assertion result for one case/result tuple."""
        ...


class EvalScorer(Protocol):
    """Contract for one case-level score producer."""

    name: str

    def __call__(self, case: EvalCase, result: EvalCaseResult) -> float:
        """Return normalized case score within implementation-defined range."""
        ...
