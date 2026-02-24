"""
---
name: Policy Governed Change Agent
description: Gate production changes with policy rules and approval analytics.
tags: [agent, runner, policy, approvals, analytics]
---
---
This example demonstrates how to enforce change-management controls in AFK.
A policy rule requests approval for production deployments and we measure policy outcomes.
---
"""

import asyncio

from afk.agents import Agent, PolicyEngine, PolicyRule, PolicyRuleCondition
from afk.core import Runner, RunnerConfig
from afk.tools import tool
from pydantic import BaseModel, Field

MODEL = "ollama_chat/gpt-oss:20b"


class DeployArgs(BaseModel):
    service: str = Field(description="Service name being deployed.")
    environment: str = Field(description="Target environment, for example prod or staging.")
    change_ticket: str = Field(description="Approved change ticket ID.")


@tool(
    args_model=DeployArgs,
    name="deploy_change",
    description="Deploy a service version to an environment. Mutating operation.",
)
def deploy_change(args: DeployArgs) -> dict:
    """Mock deploy tool for change-control demonstration."""
    return {
        "service": args.service,
        "environment": args.environment,
        "change_ticket": args.change_ticket,
        "status": "deployed",
    }


policy = PolicyEngine(
    rules=[
        PolicyRule(
            rule_id="prod-deploy-approval",
            action="request_approval",
            priority=200,
            reason="Production deploys require explicit human approval.",
            condition=PolicyRuleCondition(
                event_type="tool_before_execute",
                tool_name="deploy_change",
            ),
        )
    ]
)

change_agent = Agent(
    name="policy_governed_change_agent",
    model=MODEL,
    instructions="""
    You are a release manager.
    Use deploy_change whenever the user asks for a deployment.
    Explain the final release status and any blocked actions.
    """,
    tools=[deploy_change],
)

runner = Runner(
    policy_engine=policy,
    config=RunnerConfig(
        interaction_mode="headless",
        approval_fallback="deny",
    ),
)


async def main() -> None:
    user_input = input("[] > ")

    handle = await runner.run_handle(
        change_agent,
        user_message=user_input,
    )

    policy_events = []
    async for event in handle.events:
        if event.type == "policy_decision":
            policy_events.append(event)

    result = await handle.await_result()
    if result is None:
        raise RuntimeError("Run was cancelled before a terminal result was produced.")

    denied_count = sum(
        1
        for event in policy_events
        if event.data.get("action") in {"deny", "request_approval"}
    )

    print(f"[policy_governed_change_agent] > {result.final_text}")

    print("\n--- Policy Analytics ---")
    print(f"state: {result.state}")
    print(f"policy_events: {len(policy_events)}")
    print(f"gated_actions: {denied_count}")
    print(f"tool_calls_recorded: {len(result.tool_executions)}")
    print(f"total_tokens: {result.usage_aggregate.total_tokens}")

    if policy_events:
        print("\nPolicy decision detail:")
        for event in policy_events:
            print(
                "- "
                f"event_type={event.data.get('event_type')} | "
                f"action={event.data.get('action')} | "
                f"reason={event.data.get('reason')}"
            )


if __name__ == "__main__":
    asyncio.run(main())


"""
---
Tl;dr: This example applies a deterministic policy rule for deployment gating and inspects policy decision events for compliance analytics.
---
---
What's next?
- Switch to interactive mode with an InteractionProvider for real approvals.
- Persist policy_decision events to your audit store.
- Add separate rules for staging vs production environments.
---
"""
