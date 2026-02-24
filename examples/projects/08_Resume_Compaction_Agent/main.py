"""
---
name: Resume + Compaction Agent
description: Resume checkpointed runs and compact thread memory with retention analytics.
tags: [agent, runner, memory, resume, compaction, analytics]
---
---
This example demonstrates operational lifecycle controls for long-running systems:
resuming from checkpoints and compacting thread memory with explicit retention policies.
---
"""

import asyncio

from afk.agents import Agent, FailSafeConfig
from afk.core import Runner, RunnerConfig
from afk.memory import RetentionPolicy, StateRetentionPolicy

MODEL = "ollama_chat/gpt-oss:20b"

research_agent = Agent(
    name="resume_compaction_agent",
    model=MODEL,
    instructions="""
    You are a research operations assistant.
    Return concise but complete plans with milestones, risks, and owners.
    """,
    fail_safe=FailSafeConfig(
        max_steps=15,
        max_wall_time_s=120.0,
    ),
)

runner = Runner(config=RunnerConfig(interaction_mode="headless"))


async def main() -> None:
    thread_id = "resume-compaction-thread-77"

    first_result = await runner.run(
        research_agent,
        user_message="Create a 14-day launch plan for a new onboarding flow.",
        thread_id=thread_id,
    )

    resumed_result = await runner.resume(
        research_agent,
        run_id=first_result.run_id,
        thread_id=first_result.thread_id,
    )

    compaction = await runner.compact_thread(
        thread_id=thread_id,
        event_policy=RetentionPolicy(max_age_ms=60_000),
        state_policy=StateRetentionPolicy(max_entries=20),
    )

    print("--- Resume Output ---")
    print(first_result.final_text)

    print("\n--- Lifecycle Analytics ---")
    print(f"initial_run_id: {first_result.run_id}")
    print(f"resumed_run_id: {resumed_result.run_id}")
    print(f"thread_id: {thread_id}")
    print(f"initial_state: {first_result.state}")
    print(f"resumed_state: {resumed_result.state}")
    print(f"initial_tokens: {first_result.usage_aggregate.total_tokens}")
    print(f"resumed_tokens: {resumed_result.usage_aggregate.total_tokens}")
    print(f"events_removed_by_compaction: {compaction.events_removed}")
    print(f"states_removed_by_compaction: {compaction.states_removed}")


if __name__ == "__main__":
    asyncio.run(main())


"""
---
Tl;dr: This example shows how to resume a run using run_id + thread_id and then compact stored thread memory with measurable retention results.
---
---
What's next?
- Resume only interrupted runs and skip already terminal runs in your scheduler.
- Tune retention limits by workload criticality and compliance requirements.
- Emit compaction metrics into your observability pipeline.
---
"""
