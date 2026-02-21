# 04_resume_and_compact

Compact reference for 04_resume_and_compact.

Source: `docs/library/snippets/04_resume_and_compact.mdx`

````python 04_resume_and_compact.py
"""
Example 04: Resume and memory compaction flow.

Run:
    python 04_resume_and_compact.py
"""

from __future__ import annotations

from afk.agents import Agent
from afk.core import Runner
from afk.memory import RetentionPolicy, StateRetentionPolicy

async def main() -> None:
    model_name = os.getenv("AFK_LLM_MODEL", "gpt-4.1-mini")
    runner = Runner()

    agent = Agent(
        name="ResumeDemo",
        model=model_name,
        instructions="Answer user questions in exactly two short sentences.",
    )

    first = await runner.run(
        agent,
        user_message="What is checkpoint/resume in this runtime?",
        context={"user_id": "demo-user"},
    )

    resumed = await runner.resume(
        agent,
        run_id=first.run_id,
        thread_id=first.thread_id,
    )

    compaction = await runner.compact_thread(
        thread_id=first.thread_id,
````

> Code block truncated to 40 lines. Source: `docs/library/snippets/04_resume_and_compact.mdx`
