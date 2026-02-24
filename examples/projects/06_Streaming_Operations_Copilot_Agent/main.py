"""
---
name: Streaming Operations Copilot Agent
description: Stream operational responses in real time while tracking live event analytics.
tags: [agent, runner, streaming, memory, analytics]
---
---
This example demonstrates a production-style streaming chat loop.
It keeps a stable thread_id for memory continuity and computes per-turn stream analytics.
---
"""

import asyncio

from afk.agents import Agent, FailSafeConfig
from afk.core import Runner
from afk.tools import tool
from pydantic import BaseModel, Field

MODEL = "ollama_chat/gpt-oss:20b"


class ServiceArgs(BaseModel):
    service: str = Field(description="Service name, for example payments-api.")


@tool(
    args_model=ServiceArgs,
    name="get_service_health",
    description="Return synthetic health metrics for an internal service.",
)
def get_service_health(args: ServiceArgs) -> dict:
    """Mock service health signal for streaming demo."""
    degraded = args.service.lower().startswith("pay")
    return {
        "service": args.service,
        "status": "degraded" if degraded else "healthy",
        "error_rate_pct": 4.2 if degraded else 0.3,
        "p95_latency_ms": 780 if degraded else 140,
    }


streaming_copilot = Agent(
    name="streaming_operations_copilot_agent",
    model=MODEL,
    instructions="""
    You are an operations copilot.
    Always call get_service_health when diagnosing service issues.
    Give concise, actionable responses and include immediate next actions.
    """,
    tools=[get_service_health],
    fail_safe=FailSafeConfig(max_steps=12),
)

runner = Runner()


async def stream_turn(user_message: str, thread_id: str) -> tuple[dict, str]:
    """Stream one user turn and return analytics plus the terminal state."""
    handle = await runner.run_stream(
        streaming_copilot,
        user_message=user_message,
        thread_id=thread_id,
    )

    counters = {
        "text_delta_chunks": 0,
        "tool_started": 0,
        "tool_completed": 0,
        "errors": 0,
    }

    async for event in handle:
        if event.type == "text_delta" and event.text_delta:
            counters["text_delta_chunks"] += 1
            print(event.text_delta, end="", flush=True)
        elif event.type == "tool_started":
            counters["tool_started"] += 1
            print(f"\n[tool_started] {event.tool_name}")
        elif event.type == "tool_completed":
            counters["tool_completed"] += 1
            print(f"\n[tool_completed] {event.tool_name} success={event.tool_success}")
        elif event.type == "error":
            counters["errors"] += 1
            print(f"\n[stream_error] {event.error}")

    result = handle.result
    if result is None:
        raise RuntimeError("Stream ended without a terminal AgentResult.")

    counters["total_tokens"] = result.usage_aggregate.total_tokens
    return counters, result.state


async def main() -> None:
    thread_id = "ops-streaming-demo-001"
    turns = [
        "Investigate payments-api latency spikes in production.",
        "Given your last answer, what is the fastest rollback plan?",
        "Summarize the operator update I should send to leadership.",
    ]

    all_turn_metrics: list[dict] = []

    for index, turn in enumerate(turns, start=1):
        print(f"\n\nUser[{index}]: {turn}\n")
        print("Assistant: ", end="")
        turn_metrics, state = await stream_turn(turn, thread_id)
        turn_metrics["state"] = state
        all_turn_metrics.append(turn_metrics)

    print("\n\n--- Streaming Analytics ---")
    for idx, metrics in enumerate(all_turn_metrics, start=1):
        print(
            "- "
            f"turn={idx} | state={metrics['state']} | "
            f"text_chunks={metrics['text_delta_chunks']} | "
            f"tool_started={metrics['tool_started']} | "
            f"tool_completed={metrics['tool_completed']} | "
            f"errors={metrics['errors']} | "
            f"total_tokens={metrics['total_tokens']}"
        )


if __name__ == "__main__":
    asyncio.run(main())


"""
---
Tl;dr: This example streams responses turn-by-turn on a shared thread and computes real-time event analytics for each streamed interaction.
---
---
What's next?
- Connect these stream counters to a live dashboard in your UI.
- Track moving averages of chunk throughput and tool latency by turn.
- Add cancellation controls so operators can interrupt long runs.
---
"""
