"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Core models for deterministic eval case/suite execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ..agents import BaseAgent
from ..agents.types import AgentState, JSONValue
from ..observability.models import RunMetrics

if TYPE_CHECKING:
    from .budgets import EvalBudget
    from .contracts import AsyncEvalAssertion, EvalAssertion, EvalScorer


ExecutionMode = Literal["adaptive", "sequential", "parallel"]


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One deterministic eval case executed against a runner and agent."""

    name: str
    agent: BaseAgent
    user_message: str | None = None
    context: dict[str, JSONValue] = field(default_factory=dict)
    thread_id: str | None = None
    tags: tuple[str, ...] = ()
    budget: EvalBudget | None = None


@dataclass(frozen=True, slots=True)
class EvalAssertionResult:
    """Assertion/scoring output row attached to each eval case result."""

    name: str
    passed: bool
    details: str | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    """Terminal output for one eval case."""

    case: str
    state: AgentState
    final_text: str
    run_id: str
    thread_id: str
    event_types: list[str] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    assertions: list[EvalAssertionResult] = field(default_factory=list)
    budget_violations: list[str] = field(default_factory=list)
    passed: bool = True


@dataclass(frozen=True, slots=True)
class EvalSuiteConfig:
    """Configuration for running an eval suite."""

    execution_mode: ExecutionMode = "adaptive"
    max_concurrency: int = 4
    fail_fast: bool = False
    assertions: tuple[EvalAssertion | AsyncEvalAssertion, ...] = ()
    scorers: tuple[EvalScorer, ...] = ()
    budget: EvalBudget | None = None


@dataclass(frozen=True, slots=True)
class EvalSuiteResult:
    """Aggregated output for multi-case eval execution."""

    results: list[EvalCaseResult]
    execution_mode: ExecutionMode

    @property
    def total(self) -> int:
        """Total executed case count."""

        return len(self.results)

    @property
    def passed(self) -> int:
        """Count of cases that passed all configured checks."""

        return sum(1 for row in self.results if row.passed)

    @property
    def failed(self) -> int:
        """Count of cases that failed state/assertion/budget checks."""

        return sum(1 for row in self.results if not row.passed)
