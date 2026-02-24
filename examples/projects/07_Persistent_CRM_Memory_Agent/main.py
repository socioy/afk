"""
---
name: Persistent CRM Memory Agent
description: Maintain customer context across turns with SQLite-backed memory and continuity metrics.
tags: [agent, runner, memory, sqlite, analytics]
---
---
This example demonstrates persistent thread memory with SQLite.
The same thread_id is reused across turns, then memory statistics are inspected directly.
---
"""

import asyncio

from afk.agents import Agent
from afk.core import Runner
from afk.memory import SQLiteMemoryStore
from afk.tools import tool
from pydantic import BaseModel, Field

MODEL = "ollama_chat/gpt-oss:20b"
SQLITE_PATH = "examples/projects/07_Persistent_CRM_Memory_Agent/crm_memory.sqlite3"


class AccountArgs(BaseModel):
    account_id: str = Field(description="Account identifier from CRM.")


@tool(
    args_model=AccountArgs,
    name="lookup_account_profile",
    description="Return CRM profile data for an account.",
)
def lookup_account_profile(args: AccountArgs) -> dict:
    """Mock account profile used for memory continuity testing."""
    premium = args.account_id.upper().startswith("ENT")
    return {
        "account_id": args.account_id,
        "segment": "enterprise" if premium else "growth",
        "renewal_window_days": 45 if premium else 21,
        "assigned_csm": "Nia Patel" if premium else "Alex Kim",
    }


crm_agent = Agent(
    name="persistent_crm_memory_agent",
    model=MODEL,
    instructions="""
    You are a CRM operations assistant.
    Always call lookup_account_profile for account-specific questions.
    Use prior conversation context from the same thread when available.
    """,
    tools=[lookup_account_profile],
)


async def main() -> None:
    memory_store = SQLiteMemoryStore(path=SQLITE_PATH)
    runner = Runner(memory_store=memory_store)

    thread_id = "crm-thread-1001"
    turns = [
        "Account ENT-441 asked for renewal risk details.",
        "Who is the assigned CSM and what should I do next?",
        "Write a concise follow-up note for the account owner.",
    ]

    per_turn_tokens: list[int] = []

    for idx, message in enumerate(turns, start=1):
        result = await runner.run(
            crm_agent,
            user_message=message,
            thread_id=thread_id,
        )
        per_turn_tokens.append(result.usage_aggregate.total_tokens)

        print(f"\nTurn {idx} response:\n{result.final_text}\n")

    events = await memory_store.get_recent_events(thread_id, limit=500)
    state_rows = await memory_store.list_state(thread_id)

    print("--- Memory Analytics ---")
    print(f"thread_id: {thread_id}")
    print(f"turns: {len(turns)}")
    print(f"tokens_per_turn: {per_turn_tokens}")
    print(f"persisted_events: {len(events)}")
    print(f"persisted_state_keys: {len(state_rows)}")

    await memory_store.close()


if __name__ == "__main__":
    asyncio.run(main())


"""
---
Tl;dr: This example wires Runner to a SQLite memory backend, reuses one thread across turns, and inspects persisted events/state for continuity analytics.
---
---
What's next?
- Switch to Redis or Postgres memory for shared multi-instance deployments.
- Add periodic compact_thread calls for long-lived customer threads.
- Build retention policies by account tier to control storage growth.
---
"""
