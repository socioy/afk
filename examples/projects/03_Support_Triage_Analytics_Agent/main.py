"""
---
name: Support Triage Analytics Agent
description: Route support tickets with multi-tool triage and execution quality metrics.
tags: [agent, runner, tool, analytics]
---
---
This example demonstrates a support operations agent that triages customer incidents,
uses multiple tools, and reports execution quality metrics after each run.
---
"""

from afk.agents import Agent
from afk.core import Runner
from afk.tools import tool
from pydantic import BaseModel, Field

MODEL = "ollama_chat/gpt-oss:20b"


class CustomerArgs(BaseModel):
    customer_id: str = Field(description="Unique customer account identifier.")


@tool(
    args_model=CustomerArgs,
    name="get_customer_health",
    description="Return customer health signals (ARR tier, churn risk, SLA class).",
)
def get_customer_health(args: CustomerArgs) -> dict:
    """Mock customer success signals used during triage."""
    high_value = args.customer_id.upper().endswith("ENT")
    return {
        "customer_id": args.customer_id,
        "arr_tier": "enterprise" if high_value else "growth",
        "churn_risk": "high" if args.customer_id.endswith("7") else "medium",
        "sla_class": "P1" if high_value else "P2",
    }


@tool(
    args_model=CustomerArgs,
    name="get_recent_incidents",
    description="Return recent production incidents for this customer.",
)
def get_recent_incidents(args: CustomerArgs) -> dict:
    """Mock incident history for assignment and escalation decisions."""
    incident_count = (sum(ord(ch) for ch in args.customer_id) % 4) + 1
    return {
        "customer_id": args.customer_id,
        "incident_count_last_30d": incident_count,
        "dominant_issue": "api_timeouts" if incident_count >= 3 else "auth_errors",
    }


triage_agent = Agent(
    name="support_triage_analytics_agent",
    model=MODEL,
    instructions="""
    You are a support triage lead.

    Always call both tools before producing your answer.
    Return:
    1) Priority (P1/P2/P3)
    2) Owning team assignment
    3) Escalation recommendation
    4) Customer-facing response summary (3-5 lines)

    Keep your recommendation concrete and operational.
    """,
    tools=[get_customer_health, get_recent_incidents],
)

runner = Runner()

if __name__ == "__main__":
    user_input = input("[] > ")

    result = runner.run_sync(
        triage_agent,
        user_message=user_input,
    )

    print(f"[support_triage_analytics_agent] > {result.final_text}")

    tool_calls = result.tool_executions
    total_calls = len(tool_calls)
    failed_calls = sum(1 for record in tool_calls if not record.success)
    latency_values = [
        record.latency_ms for record in tool_calls if record.latency_ms is not None
    ]
    avg_latency = (
        round(sum(latency_values) / len(latency_values), 2) if latency_values else None
    )

    print("\n--- Run Analytics ---")
    print(f"state: {result.state}")
    print(f"total_tokens: {result.usage_aggregate.total_tokens}")
    print(f"tool_calls: {total_calls}")
    print(f"failed_tool_calls: {failed_calls}")
    print(f"avg_tool_latency_ms: {avg_latency}")

    if tool_calls:
        print("\nTool execution detail:")
        for record in tool_calls:
            print(
                "- "
                f"{record.tool_name} | success={record.success} | "
                f"latency_ms={record.latency_ms} | error={record.error}"
            )


"""
---
Tl;dr: This example adds multi-tool triage and computes quality metrics like failure count and average tool latency.
---
---
What's next?
- Persist triage metrics to a warehouse for weekly SRE support reporting.
- Add policy rules so VIP customers always require human approval before downgrade.
- Build alerting when tool failure rate exceeds your threshold.
---
"""
