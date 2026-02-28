"""
---
name: Habit Tracker
description: A habit tracker agent that uses SQLiteMemoryStore for persistent data across sessions.
tags: [agent, runner, memory, sqlite, persistence, async]
---
---
This example demonstrates how to use the SQLiteMemoryStore to persist agent state (habits, streaks,
completions) across sessions. Unlike InMemoryMemoryStore (which loses data when the process exits),
SQLiteMemoryStore writes to a local database file so your data survives restarts. This pattern is
essential for building agents that maintain long-term state — personal assistants, trackers, learning
systems, and more. Swapping between memory backends (InMemory, SQLite, Redis, Postgres) requires
changing only a single line.
---
"""

import asyncio  # <- Async is required because memory store operations (get_state, put_state) are async methods.
import json  # <- For serializing habit data to JSON for storage.
from datetime import datetime, timezone  # <- For tracking completion timestamps.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner executes agents and manages their state. Accepts an optional memory_store parameter.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool, ToolContext  # <- @tool decorator and ToolContext for accessing runtime context including memory.
from afk.memory import SQLiteMemoryStore  # <- SQLiteMemoryStore persists data to a local .db file. Swap this import to InMemoryMemoryStore, RedisMemoryStore, or PostgresMemoryStore for different backends.


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class AddHabitArgs(BaseModel):  # <- Schema for creating a new habit to track.
    name: str = Field(description="Name of the habit (e.g., 'Exercise', 'Read', 'Meditate')")
    goal: str = Field(description="Daily goal description (e.g., '30 minutes', '20 pages', '10 minutes')")


class HabitNameArgs(BaseModel):  # <- Schema for tools that operate on a specific habit by name.
    name: str = Field(description="Name of the habit to operate on")


class EmptyArgs(BaseModel):  # <- Schema for tools that take no input.
    pass


# ===========================================================================
# Memory keys — these define how data is organized in the memory store
# ===========================================================================

HABITS_KEY = "habits"  # <- State key for the habits dictionary. Stored under thread_id scope in the memory store.
COMPLETIONS_KEY = "completions"  # <- State key for the completions log.


# ===========================================================================
# Tool definitions — all tools use ToolContext to access the memory store
# ===========================================================================

@tool(args_model=AddHabitArgs, name="add_habit", description="Add a new habit to track")
async def add_habit(args: AddHabitArgs, ctx: ToolContext) -> str:  # <- ToolContext is injected by the runner. ctx.memory gives access to the memory store bound to the current thread_id.
    memory = ctx.memory  # <- Access the memory store from the tool context. This is the same SQLiteMemoryStore passed to the Runner.
    thread_id = ctx.thread_id  # <- The thread_id scopes all state. Different thread_ids have independent state.

    # --- Load existing habits from persistent storage ---
    habits_raw = await memory.get_state(thread_id, HABITS_KEY)  # <- get_state retrieves a JSON value from the store. Returns None if the key doesn't exist yet.
    habits = json.loads(habits_raw) if habits_raw else {}  # <- Deserialize from JSON string.

    if args.name.lower() in habits:
        return f"Habit '{args.name}' already exists! Use complete_habit to log progress."

    habits[args.name.lower()] = {
        "name": args.name,
        "goal": args.goal,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "streak": 0,
        "total_completions": 0,
    }

    await memory.put_state(thread_id, HABITS_KEY, json.dumps(habits))  # <- put_state writes a JSON value to the store. This persists to the SQLite database file.
    return f"Habit '{args.name}' added with goal: {args.goal}. Start tracking today!"


@tool(args_model=HabitNameArgs, name="complete_habit", description="Mark a habit as completed for today")
async def complete_habit(args: HabitNameArgs, ctx: ToolContext) -> str:
    memory = ctx.memory
    thread_id = ctx.thread_id

    habits_raw = await memory.get_state(thread_id, HABITS_KEY)
    habits = json.loads(habits_raw) if habits_raw else {}
    key = args.name.lower()

    if key not in habits:
        available = ", ".join(habits.keys()) if habits else "none"
        return f"Habit '{args.name}' not found. Available habits: {available}"

    # --- Update the habit's streak and completion count ---
    habits[key]["streak"] += 1  # <- Increment the streak. A real app would check if the last completion was yesterday.
    habits[key]["total_completions"] += 1
    await memory.put_state(thread_id, HABITS_KEY, json.dumps(habits))

    # --- Log the completion event ---
    completions_raw = await memory.get_state(thread_id, COMPLETIONS_KEY)
    completions = json.loads(completions_raw) if completions_raw else []
    completions.append({
        "habit": args.name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    await memory.put_state(thread_id, COMPLETIONS_KEY, json.dumps(completions))  # <- Store the completion log as a separate state key. This separation keeps data organized and queryable.

    streak = habits[key]["streak"]
    total = habits[key]["total_completions"]
    return f"'{args.name}' completed! Streak: {streak} days, Total: {total} times."


@tool(args_model=EmptyArgs, name="list_habits", description="List all tracked habits with their current streaks and goals")
async def list_habits(args: EmptyArgs, ctx: ToolContext) -> str:
    memory = ctx.memory
    thread_id = ctx.thread_id

    habits_raw = await memory.get_state(thread_id, HABITS_KEY)
    habits = json.loads(habits_raw) if habits_raw else {}

    if not habits:
        return "No habits being tracked yet. Use add_habit to start!"

    lines = []
    for key, habit in habits.items():
        lines.append(
            f"  - {habit['name']}: goal={habit['goal']}, streak={habit['streak']} days, total={habit['total_completions']}"
        )
    return "Your Habits:\n" + "\n".join(lines)


@tool(args_model=EmptyArgs, name="get_progress_report", description="Get a detailed progress report for all habits")
async def get_progress_report(args: EmptyArgs, ctx: ToolContext) -> str:
    memory = ctx.memory
    thread_id = ctx.thread_id

    habits_raw = await memory.get_state(thread_id, HABITS_KEY)
    habits = json.loads(habits_raw) if habits_raw else {}
    completions_raw = await memory.get_state(thread_id, COMPLETIONS_KEY)
    completions = json.loads(completions_raw) if completions_raw else []

    if not habits:
        return "No habits to report on. Add some habits first!"

    # --- Build report ---
    total_habits = len(habits)
    total_completions = len(completions)
    best_streak = max((h["streak"] for h in habits.values()), default=0)
    best_habit = max(habits.values(), key=lambda h: h["streak"])["name"] if habits else "N/A"

    report = (
        f"Progress Report\n"
        f"{'=' * 30}\n"
        f"Total habits tracked: {total_habits}\n"
        f"Total completions logged: {total_completions}\n"
        f"Best current streak: {best_streak} days ({best_habit})\n"
        f"\nPer-habit breakdown:\n"
    )
    for habit in habits.values():
        report += f"  {habit['name']}: {habit['streak']} day streak, {habit['total_completions']} total, goal: {habit['goal']}\n"

    return report


@tool(args_model=HabitNameArgs, name="remove_habit", description="Stop tracking a habit")
async def remove_habit(args: HabitNameArgs, ctx: ToolContext) -> str:
    memory = ctx.memory
    thread_id = ctx.thread_id

    habits_raw = await memory.get_state(thread_id, HABITS_KEY)
    habits = json.loads(habits_raw) if habits_raw else {}
    key = args.name.lower()

    if key not in habits:
        return f"Habit '{args.name}' not found."

    removed = habits.pop(key)
    await memory.put_state(thread_id, HABITS_KEY, json.dumps(habits))
    return f"Habit '{removed['name']}' removed. It had a {removed['streak']} day streak and {removed['total_completions']} total completions."


# ===========================================================================
# Agent and runner setup
# ===========================================================================

habit_agent = Agent(
    name="habit-tracker",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a motivational habit tracker assistant. You help users build and maintain daily habits.

    When the user wants to track a new habit:
    1. Use add_habit with a clear name and measurable goal.

    When the user completes a habit:
    1. Use complete_habit to log it and show their streak.
    2. Celebrate their progress! Be encouraging.

    When the user wants to see their progress:
    1. Use list_habits for a quick overview or get_progress_report for detailed stats.

    Be positive and motivating. Remind users that consistency is key — even small streaks matter!

    **NOTE**: Data persists between sessions thanks to SQLite storage. Remind users of this!
    """,
    tools=[add_habit, complete_habit, list_habits, get_progress_report, remove_habit],
)

THREAD_ID = "habit-tracker-main"  # <- A fixed thread_id so data persists across restarts. Change this to create separate tracking sessions.
DB_PATH = "habit_tracker.db"  # <- The SQLite database file path. Data persists here between runs.


async def main():
    memory = SQLiteMemoryStore(db_path=DB_PATH)  # <- Create a SQLiteMemoryStore pointing to a local file. This is the only line you change to switch backends (e.g., InMemoryMemoryStore() for testing, RedisMemoryStore(url=...) for distributed).
    await memory.setup()  # <- Initialize the store (creates tables if they don't exist). Always call setup() before using the store.

    runner = Runner(memory_store=memory)  # <- Pass the memory store to the Runner. All agents executed by this runner will have access to this store via ToolContext.memory.

    print("Habit Tracker Agent (type 'quit' to exit)")
    print(f"Data persists in: {DB_PATH}")
    print("=" * 45)
    print("Try: 'I want to track exercise', 'I did my reading today', 'show my progress'\n")

    try:
        while True:
            user_input = input("[] > ")

            if user_input.strip().lower() in ("quit", "exit", "q"):
                print("Keep up the great habits! Goodbye!")
                break

            response = await runner.run(
                habit_agent,
                user_message=user_input,
                thread_id=THREAD_ID,  # <- The thread_id ensures all state operations are scoped to this session. Using the same thread_id across restarts means the agent picks up where it left off.
            )

            print(f"[habit-tracker] > {response.final_text}\n")
    finally:
        await memory.close()  # <- Always close the memory store cleanly. This flushes any pending writes and closes the SQLite connection.


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() is required because memory operations are async.



"""
---
Tl;dr: This example creates a habit tracker agent backed by SQLiteMemoryStore for persistent data that
survives process restarts. Tools use ToolContext to access the memory store (ctx.memory) and thread-scoped
state operations (get_state/put_state). The same thread_id across sessions means the agent remembers your
habits, streaks, and completion history. Swapping to a different backend (InMemory, Redis, Postgres) requires
changing only the memory store constructor — all tool code stays the same.
---
---
What's next?
- Restart the program and verify your habits are still there — that's SQLite persistence in action.
- Try swapping SQLiteMemoryStore for InMemoryMemoryStore to see data disappear on restart.
- Add date-aware streak tracking (checking if the last completion was yesterday vs. today).
- Experiment with multiple thread_ids to create separate tracking sessions (e.g., work habits vs. personal habits).
- Explore the memory store's event logging with append_event/get_recent_events for an activity feed.
- Check out the Vector Search example for semantic memory queries using long_term_memory operations!
---
"""
