"""
---
name: Incident Command Center Agent
description: Coordinate incident response with specialist subagents and delegation analytics.
tags: [agent, runner, subagents, analytics]
---
---
This example demonstrates multi-agent incident response orchestration.
A lead agent delegates to specialists and then synthesizes an incident command update.
---
"""

from afk.agents import Agent, FailSafeConfig
from afk.core import Runner

MODEL = "ollama_chat/gpt-oss:20b"

triage_specialist = Agent(
    name="triage_specialist",
    model=MODEL,
    instructions="""
    You classify incidents by severity and blast radius.
    Return: severity level, impacted systems, and initial containment steps.
    """,
)

rca_specialist = Agent(
    name="rca_specialist",
    model=MODEL,
    instructions="""
    You produce likely root causes and confidence for each hypothesis.
    Return the top 3 causes in descending probability.
    """,
)

comms_specialist = Agent(
    name="comms_specialist",
    model=MODEL,
    instructions="""
    You draft internal and external stakeholder communication updates.
    Keep updates calm, factual, and timestamp-aware.
    """,
)

incident_commander = Agent(
    name="incident_commander",
    model=MODEL,
    instructions="""
    You are the Incident Commander.
    You MUST invoke all three subagents exactly once before finalizing:
    - triage_specialist
    - rca_specialist
    - comms_specialist

    Final response format:
    1) Incident summary
    2) Technical action plan
    3) Stakeholder communication block
    4) Next 30-minute command plan
    """,
    subagents=[triage_specialist, rca_specialist, comms_specialist],
    fail_safe=FailSafeConfig(
        max_steps=25,
        subagent_failure_policy="retry_then_degrade",
    ),
)

runner = Runner()

if __name__ == "__main__":
    user_input = input("[] > ")

    result = runner.run_sync(
        incident_commander,
        user_message=user_input,
    )

    print(f"[incident_commander] > {result.final_text}")

    subagent_runs = result.subagent_executions
    successful_subagents = sum(1 for row in subagent_runs if row.success)
    latencies = [row.latency_ms for row in subagent_runs if row.latency_ms is not None]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None

    print("\n--- Delegation Analytics ---")
    print(f"state: {result.state}")
    print(f"subagent_calls: {len(subagent_runs)}")
    print(f"successful_subagent_calls: {successful_subagents}")
    print(f"avg_subagent_latency_ms: {avg_latency}")
    print(f"total_tokens: {result.usage_aggregate.total_tokens}")

    if subagent_runs:
        print("\nSubagent execution detail:")
        for row in subagent_runs:
            print(
                "- "
                f"{row.subagent_name} | success={row.success} | "
                f"latency_ms={row.latency_ms} | error={row.error}"
            )


"""
---
Tl;dr: This example introduces a real incident response workflow where a lead agent delegates to multiple specialists and reports delegation performance analytics.
---
---
What's next?
- Add policy rules to require approval before customer-facing comms are finalized.
- Log subagent outputs into your incident timeline system.
- Compare latency and token usage between single-agent and multi-agent incident handling.
---
"""
