"""
Rule-based policy engine for agent/runtime controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Iterable, Literal

from .types import JSONValue, PolicyAction, PolicyDecision, PolicyEvent, json_value_from_tool_result


PolicySubject = Literal["llm_call", "tool_call", "subagent_call", "interaction", "any"]


@dataclass(frozen=True, slots=True)
class PolicyRuleCondition:
    """Match conditions used by policy rules."""

    event_type: str | None = None
    tool_name: str | None = None
    tool_name_pattern: str | None = None
    subagent_name: str | None = None
    context_equals: dict[str, JSONValue] = field(default_factory=dict)
    context_has_keys: list[str] = field(default_factory=list)
    metadata_equals: dict[str, JSONValue] = field(default_factory=dict)

    def matches(self, event: PolicyEvent) -> bool:
        """Return `True` when this condition matches the given event."""
        if self.event_type is not None and event.event_type != self.event_type:
            return False

        if self.tool_name is not None and event.tool_name != self.tool_name:
            return False

        if self.tool_name_pattern is not None:
            candidate = event.tool_name or ""
            if not fnmatch(candidate, self.tool_name_pattern):
                return False

        if self.subagent_name is not None and event.subagent_name != self.subagent_name:
            return False

        for key in self.context_has_keys:
            if key not in event.context:
                return False

        for key, expected in self.context_equals.items():
            if event.context.get(key) != expected:
                return False

        for key, expected in self.metadata_equals.items():
            if event.metadata.get(key) != expected:
                return False

        return True


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """Single policy rule with deterministic priority ordering."""

    rule_id: str
    action: PolicyAction
    priority: int = 100
    enabled: bool = True
    subjects: list[PolicySubject] = field(default_factory=lambda: ["any"])
    reason: str | None = None
    request_payload: dict[str, JSONValue] = field(default_factory=dict)
    updated_tool_args: dict[str, JSONValue] | None = None
    condition: PolicyRuleCondition = field(default_factory=PolicyRuleCondition)

    def applies_to(self, event: PolicyEvent) -> bool:
        """Return `True` when this rule applies to the given event."""
        if not self.enabled:
            return False
        subjects = self.subjects or ["any"]
        if "any" in subjects:
            return self.condition.matches(event)
        subject = infer_policy_subject(event.event_type)
        if subject not in subjects:
            return False
        return self.condition.matches(event)


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Evaluation output containing final decision and matched rules."""

    decision: PolicyDecision
    matched_rule_ids: list[str] = field(default_factory=list)


class PolicyEngine:
    """
    Deterministic rule evaluator.

    Rule selection:
    1. Only enabled + matching rules are considered.
    2. Rules are sorted by priority DESC, then rule_id ASC.
    3. The highest-priority matching rule decides the action.
    """

    def __init__(self, rules: Iterable[PolicyRule] | None = None) -> None:
        self._rules: list[PolicyRule] = sorted(
            list(rules or []),
            key=lambda rule: (-rule.priority, rule.rule_id),
        )

    @property
    def rules(self) -> list[PolicyRule]:
        """Return configured rules in evaluation order."""
        return list(self._rules)

    def evaluate(self, event: PolicyEvent) -> PolicyEvaluation:
        """Evaluate rules and return the selected policy decision."""
        matches = [rule for rule in self._rules if rule.applies_to(event)]
        if not matches:
            return PolicyEvaluation(decision=PolicyDecision(action="allow"))

        chosen = matches[0]
        decision = PolicyDecision(
            action=chosen.action,
            reason=chosen.reason,
            updated_tool_args=(
                {
                    str(k): json_value_from_tool_result(v)
                    for k, v in chosen.updated_tool_args.items()
                }
                if isinstance(chosen.updated_tool_args, dict)
                else None
            ),
            request_payload={
                str(k): json_value_from_tool_result(v)
                for k, v in chosen.request_payload.items()
            },
            policy_id=chosen.rule_id,
            matched_rules=[rule.rule_id for rule in matches],
        )
        return PolicyEvaluation(
            decision=decision,
            matched_rule_ids=[rule.rule_id for rule in matches],
        )


def infer_policy_subject(event_type: str) -> PolicySubject:
    """Infer policy subject channel from event type name."""
    normalized = (event_type or "").strip().lower()
    if normalized.startswith("tool_"):
        return "tool_call"
    if normalized.startswith("llm_"):
        return "llm_call"
    if normalized.startswith("subagent_"):
        return "subagent_call"
    if "approval" in normalized or "user_input" in normalized or "interaction" in normalized:
        return "interaction"
    return "any"


def normalize_policy_payload(payload: dict[str, Any] | None) -> dict[str, JSONValue]:
    """Normalize arbitrary payload values into JSON-safe policy payload."""
    if not isinstance(payload, dict):
        return {}
    return {str(k): json_value_from_tool_result(v) for k, v in payload.items()}
