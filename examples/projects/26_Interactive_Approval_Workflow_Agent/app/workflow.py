"""Workflow entry for interactive mode and deferred input decisions."""

import asyncio

from afk.agents import Agent, PolicyDecision, PolicyEvent
from afk.core import Runner, RunnerConfig

from .complexity_chain import run_chain
from .interaction import AutoResolveInteractiveProvider
from .llm_stub import InputGatedLLM
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario
from .tools import echo_change_plan


def policy(event: PolicyEvent) -> PolicyDecision:
    if event.event_type == "tool_before_execute" and event.tool_name == "echo_change_plan":
        return PolicyDecision(
            action="request_user_input",
            request_payload={
                "prompt": "Provide the approved change window",
                "target_arg": "text",
            },
        )
    return PolicyDecision(action="allow")


async def _run() -> None:
    scenario = build_scenario("interactive-policy-gate")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    provider = AutoResolveInteractiveProvider()
    runner = Runner(
        interaction_provider=provider,
        config=RunnerConfig(
            interaction_mode="interactive",
            approval_timeout_s=1.0,
            input_timeout_s=1.0,
        ),
    )

    agent = Agent(
        model=InputGatedLLM(),
        tools=[echo_change_plan],
        instructions="Collect approved change plan details.",
        policy_roles=[policy],
    )

    result = await runner.run(agent, user_message="Prepare tonight's change plan")

    feature_payload: dict[str, object] = {
        "kind": "interactive_mode",
        "status": "ok",
        "tool_calls": len(result.tool_executions),
        "notifications": len(provider.notifications()),
    }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[interactive] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
