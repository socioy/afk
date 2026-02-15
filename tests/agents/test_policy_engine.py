from __future__ import annotations

from afk.agents.policy import PolicyEngine, PolicyRule, PolicyRuleCondition, infer_policy_subject
from afk.agents.types import PolicyEvent


def _event(**kwargs):
    base = {
        "event_type": "tool_before_execute",
        "run_id": "run_1",
        "thread_id": "thread_1",
        "step": 1,
        "context": {"user_id": "u1"},
        "tool_name": "search_web",
        "tool_args": {"q": "hello"},
        "metadata": {"source": "test"},
    }
    base.update(kwargs)
    return PolicyEvent(**base)


def test_policy_engine_priority_is_deterministic():
    engine = PolicyEngine(
        rules=[
            PolicyRule(
                rule_id="allow-default",
                action="allow",
                priority=10,
                condition=PolicyRuleCondition(event_type="tool_before_execute"),
            ),
            PolicyRule(
                rule_id="deny-specific",
                action="deny",
                priority=100,
                reason="blocked",
                condition=PolicyRuleCondition(tool_name="search_web"),
            ),
        ]
    )

    evaluation = engine.evaluate(_event())
    assert evaluation.decision.action == "deny"
    assert evaluation.decision.policy_id == "deny-specific"
    assert evaluation.decision.matched_rules == ["deny-specific", "allow-default"]


def test_policy_engine_condition_matches_context_and_metadata():
    engine = PolicyEngine(
        rules=[
            PolicyRule(
                rule_id="ask-approval",
                action="request_approval",
                priority=50,
                condition=PolicyRuleCondition(
                    context_equals={"user_id": "u1"},
                    metadata_equals={"source": "test"},
                ),
            )
        ]
    )

    evaluation = engine.evaluate(_event())
    assert evaluation.decision.action == "request_approval"
    assert evaluation.decision.policy_id == "ask-approval"


def test_infer_policy_subject_mapping():
    assert infer_policy_subject("tool_before_execute") == "tool_call"
    assert infer_policy_subject("llm_before_execute") == "llm_call"
    assert infer_policy_subject("subagent_before_execute") == "subagent_call"
    assert infer_policy_subject("request_user_input") == "interaction"
