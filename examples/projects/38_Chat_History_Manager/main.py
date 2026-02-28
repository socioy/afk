"""
---
name: Chat History Manager
description: A chat agent that manages multiple conversation threads using AFK's memory system for history, events, and state.
tags: [agent, runner, tools, memory, state, events]
---
---
This example goes deep into AFK's memory system by building a multi-thread chat history manager. It demonstrates the full lifecycle of InMemoryMemoryStore: append_event for logging conversation events, get_recent_events for retrieving history, put_state/get_state for thread metadata (like names and timestamps), and list_state for discovering all stored keys. Multiple thread_ids show how memory is fully isolated per thread -- each conversation is a separate namespace with its own events and state.
---
"""

import asyncio  # <- We use asyncio because all memory operations are async. This is the standard pattern for AFK agents that use memory.

from pydantic import BaseModel, Field  # <- Pydantic is used to define structured argument models for tools. This lets you specify exactly what inputs each tool expects, with types, descriptions, and validation built in.
from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior.
from afk.tools import tool  # <- The @tool decorator turns a plain Python function into a tool that an agent can call. You give it a name, description, and an args_model so the LLM knows when and how to use it.
from afk.memory import InMemoryMemoryStore, MemoryEvent, now_ms, new_id  # <- InMemoryMemoryStore is a fast, in-process memory backend. MemoryEvent records things that happened. now_ms() and new_id() are helpers for timestamps and unique IDs.


# ===========================================================================
# Memory setup
# ===========================================================================

memory = InMemoryMemoryStore()  # <- In-memory store for development. For production, swap in SQLiteMemoryStore or RedisMemoryStore -- same API, just change the class.

# Global state to track the currently active thread
current_thread_id = "default"  # <- The currently active thread. Tools read and write this to know which thread's memory to access. We start with a "default" thread.
known_threads: list[str] = ["default"]  # <- A simple list tracking all thread names the user has created. In production, you'd store this in memory too.


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class ListThreadsArgs(BaseModel):  # <- No-argument tool: lists all known conversation threads.
    pass


class SwitchThreadArgs(BaseModel):  # <- Takes a thread name to switch to. Creates the thread if it doesn't exist.
    thread_name: str = Field(description="The name of the conversation thread to switch to")


class GetHistoryArgs(BaseModel):  # <- Optional limit parameter for how many recent events to retrieve.
    limit: int = Field(default=10, description="Maximum number of recent messages to retrieve")


class ClearHistoryArgs(BaseModel):  # <- No-argument tool: clears the current thread's event history.
    pass


class GetThreadInfoArgs(BaseModel):  # <- No-argument tool: shows metadata about the current thread.
    pass


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=ListThreadsArgs, name="list_threads", description="List all available conversation threads")  # <- Lists all threads the user has created. Shows which one is currently active.
async def list_threads(args: ListThreadsArgs) -> str:
    global current_thread_id
    if not known_threads:
        return "No threads exist yet. Start chatting to create the default thread."

    thread_details = []
    for thread_name in known_threads:
        # Check if thread has any metadata stored
        msg_count = await memory.get_state(thread_id=thread_name, key="message_count")  # <- get_state returns the stored value or None. We use it here to show how many messages each thread has.
        count_str = f" ({msg_count} messages)" if msg_count is not None else " (empty)"
        active = " [ACTIVE]" if thread_name == current_thread_id else ""
        thread_details.append(f"  - {thread_name}{count_str}{active}")

    return "Conversation threads:\n" + "\n".join(thread_details)


@tool(args_model=SwitchThreadArgs, name="switch_thread", description="Switch to a different conversation thread, creating it if it doesn't exist")  # <- Switches the active thread. Creates a new thread with initial metadata if it doesn't exist yet.
async def switch_thread(args: SwitchThreadArgs) -> str:
    global current_thread_id, known_threads
    thread_name = args.thread_name.strip().lower().replace(" ", "-")  # <- Normalize the thread name: lowercase, hyphens instead of spaces. This prevents "Work" and "work" from being different threads.

    if thread_name not in known_threads:
        # New thread -- initialize its metadata via put_state
        known_threads.append(thread_name)
        await memory.put_state(  # <- put_state stores a key-value pair scoped to the thread. Here we store creation metadata so we can show it later with get_state.
            thread_id=thread_name,
            key="created_at",
            value=now_ms(),  # <- now_ms() returns the current time in milliseconds. Useful for timestamps in memory.
        )
        await memory.put_state(
            thread_id=thread_name,
            key="message_count",
            value=0,  # <- Initialize message count to 0. We'll increment this each time the user sends a message.
        )
        current_thread_id = thread_name
        return f"Created new thread '{thread_name}' and switched to it. This thread has no history yet."

    current_thread_id = thread_name

    # Show recent history from the thread we're switching to
    events = await memory.get_recent_events(thread_id=thread_name, limit=3)  # <- get_recent_events returns the most recent events for this thread. We show a preview so the user remembers what was discussed.
    if not events:
        return f"Switched to thread '{thread_name}'. No conversation history yet."

    preview_lines = []
    for evt in events:
        if evt.type == "trace" and "role" in evt.payload:
            role = evt.payload.get("role", "unknown")
            text = evt.payload.get("text", "")
            preview_lines.append(f"  [{role}]: {text[:80]}...")  # <- Show first 80 chars of each message as a preview.

    preview = "\n".join(preview_lines) if preview_lines else "  (no readable messages)"
    return f"Switched to thread '{thread_name}'. Recent messages:\n{preview}"


@tool(args_model=GetHistoryArgs, name="get_history", description="Retrieve the conversation history for the current thread")  # <- Reads the event log for the current thread and formats it as a readable conversation transcript.
async def get_history(args: GetHistoryArgs) -> str:
    global current_thread_id

    events = await memory.get_recent_events(  # <- get_recent_events returns up to `limit` events in chronological order (oldest first). Each event has a type, timestamp, and payload.
        thread_id=current_thread_id,
        limit=args.limit,
    )

    if not events:
        return f"Thread '{current_thread_id}' has no conversation history yet."

    # Format events into a readable transcript
    lines = [f"History for thread '{current_thread_id}' (last {args.limit} messages):"]
    for evt in events:
        if evt.type == "trace" and "role" in evt.payload:
            role = evt.payload.get("role", "unknown")
            text = evt.payload.get("text", "")
            lines.append(f"  [{role}]: {text}")

    # Also show thread metadata using list_state
    all_state = await memory.list_state(thread_id=current_thread_id)  # <- list_state returns ALL key-value pairs for this thread as a dict. Useful for debugging and for seeing everything the thread knows.
    if all_state:
        lines.append(f"\n  Thread metadata keys: {', '.join(all_state.keys())}")  # <- Shows all state keys stored for this thread. Helps the user understand what metadata is being tracked.

    return "\n".join(lines)


@tool(args_model=ClearHistoryArgs, name="clear_history", description="Clear the conversation history for the current thread")  # <- Replaces the event log with an empty list and resets the message count.
async def clear_history(args: ClearHistoryArgs) -> str:
    global current_thread_id

    await memory.replace_thread_events(  # <- replace_thread_events atomically replaces all events for a thread. Passing an empty list effectively clears the history.
        thread_id=current_thread_id,
        events=[],
    )
    await memory.put_state(  # <- Reset the message count to 0 since we just cleared all events.
        thread_id=current_thread_id,
        key="message_count",
        value=0,
    )

    return f"Cleared all conversation history for thread '{current_thread_id}'."


@tool(args_model=GetThreadInfoArgs, name="get_thread_info", description="Show detailed metadata about the current thread including all stored state keys")  # <- Demonstrates list_state for inspecting all metadata stored in a thread.
async def get_thread_info(args: GetThreadInfoArgs) -> str:
    global current_thread_id

    all_state = await memory.list_state(thread_id=current_thread_id)  # <- list_state returns a dict of all key-value pairs for this thread. No prefix filter means we get everything.

    if not all_state:
        return f"Thread '{current_thread_id}' has no stored metadata."

    lines = [f"Thread info for '{current_thread_id}':"]
    for key, value in all_state.items():
        lines.append(f"  {key}: {value}")  # <- Each key-value pair shows what metadata is stored. Keys like "created_at", "message_count", "last_topic" are all thread-scoped.

    event_count = len(await memory.get_recent_events(thread_id=current_thread_id, limit=1000))
    lines.append(f"  total_events: {event_count}")

    return "\n".join(lines)


# ===========================================================================
# Helper: log a message as a MemoryEvent
# ===========================================================================

async def log_message(thread_id: str, role: str, text: str) -> None:
    """Record a chat message as a MemoryEvent in the given thread.

    This helper is called after each user message and agent response so
    that the conversation history is persisted in memory. Each message
    becomes a trace event with the role and text in the payload.
    """
    event = MemoryEvent(  # <- MemoryEvent is a frozen dataclass. Every field is required except tags.
        id=new_id("msg"),  # <- new_id("msg") generates a unique ID like "msg_a1b2c3d4...". The prefix helps you identify event types at a glance.
        thread_id=thread_id,  # <- Events are scoped to a thread. This is how different conversations stay isolated.
        user_id=role,  # <- We use the role ("user" or "assistant") as user_id for simplicity. In production, this would be a real user ID.
        type="trace",  # <- "trace" is the right event type for application-level tracking.
        timestamp=now_ms(),  # <- Millisecond timestamp.
        payload={  # <- Arbitrary JSON-serializable data. We store the role and text for easy retrieval.
            "role": role,
            "text": text,
        },
    )
    await memory.append_event(event)  # <- Appends to the thread's event log in chronological order.

    # Increment message count in state
    count = await memory.get_state(thread_id=thread_id, key="message_count")  # <- Read current count (or None if not set).
    new_count = (count or 0) + 1
    await memory.put_state(thread_id=thread_id, key="message_count", value=new_count)  # <- Overwrite with incremented count.


# ===========================================================================
# Agent setup
# ===========================================================================

chat_manager = Agent(
    name="chat-history-manager",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a helpful chat assistant that manages multiple conversation threads.

    You can:
    - Chat normally with the user (respond to their messages)
    - List all conversation threads with list_threads
    - Switch between threads with switch_thread
    - View conversation history with get_history
    - Clear history with clear_history
    - View thread metadata with get_thread_info

    When the user wants to create or switch threads, use the switch_thread tool.
    When they ask about history or previous messages, use get_history.
    When they want to see all threads, use list_threads.
    When they ask about thread details or metadata, use get_thread_info.

    Always confirm which thread is active after switching.
    Keep your responses concise and helpful.
    """,  # <- Instructions guide the agent to use the right tool based on the user's intent.
    tools=[list_threads, switch_thread, get_history, clear_history, get_thread_info],  # <- All five tools are available. The LLM picks the right one based on the user's message and the instructions.
)

runner = Runner()  # <- A single Runner instance handles all agent executions.


# ===========================================================================
# Main loop (async because memory operations require it)
# ===========================================================================

async def main():
    await memory.setup()  # <- Initialize the memory store. MUST be called before any memory operations. Forgetting this raises RuntimeError.

    # Initialize default thread metadata
    await memory.put_state(thread_id="default", key="created_at", value=now_ms())  # <- Store creation time for the default thread.
    await memory.put_state(thread_id="default", key="message_count", value=0)  # <- Initialize message count.

    print("[chat-history-manager] > Welcome! I'm your chat history manager.")
    print("[chat-history-manager] > You can chat normally, or manage threads:")
    print("[chat-history-manager] >   - 'list threads' to see all threads")
    print("[chat-history-manager] >   - 'switch to <name>' to change threads")
    print("[chat-history-manager] >   - 'show history' to see conversation history")
    print("[chat-history-manager] >   - 'clear history' to clear current thread")
    print("[chat-history-manager] >   - 'thread info' to see thread metadata")
    print("[chat-history-manager] > Type 'quit' to exit.\n")

    while True:  # <- A conversation loop. Each iteration handles one user message. Memory persists across iterations because the MemoryStore holds state for the entire session.
        user_input = input(f"[{current_thread_id}] > ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        # Log the user's message to the current thread
        await log_message(current_thread_id, "user", user_input)  # <- Record every user message as a MemoryEvent before running the agent. This builds the conversation history.

        # Store the last topic as thread-scoped state
        await memory.put_state(  # <- put_state overwrites any previous value for the same key. Here we track what the user last talked about in this thread.
            thread_id=current_thread_id,
            key="last_topic",
            value=user_input[:100],  # <- Store first 100 chars as the topic summary.
        )

        response = await runner.run(
            chat_manager, user_message=user_input
        )  # <- Run the agent asynchronously using the Runner. The agent may call tools (list_threads, switch_thread, etc.) and the Runner handles it all.

        # Log the agent's response to the current thread
        await log_message(current_thread_id, "assistant", response.final_text)  # <- Record the agent's response as a MemoryEvent too. Now both sides of the conversation are in the event log.

        print(f"[chat-history-manager] > {response.final_text}\n")

    # --- Show final summary before exiting ---
    print("\n--- Session Summary ---")
    for thread_name in known_threads:
        count = await memory.get_state(thread_id=thread_name, key="message_count")
        print(f"  {thread_name}: {count or 0} messages")  # <- Show message counts per thread as a summary.

    await memory.close()  # <- Clean up the memory store. Always pair setup() with close().


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() is the standard way to start an async main function. Required because memory operations are all async.



"""
---
Tl;dr: This example builds a multi-thread chat history manager using AFK's InMemoryMemoryStore. It demonstrates the full memory API: append_event to record conversation messages as MemoryEvent objects, get_recent_events to retrieve chat history, put_state/get_state for thread metadata (creation time, message count, last topic), list_state to inspect all stored keys, and replace_thread_events to clear history. Multiple thread_ids show how memory is fully isolated -- switching threads gives a completely separate conversation namespace. The agent uses five tools (list_threads, switch_thread, get_history, clear_history, get_thread_info) that all communicate through memory rather than Python globals.
---
---
What's next?
- Swap InMemoryMemoryStore for SQLiteMemoryStore to persist conversation history across program restarts. The API is identical -- just change the class name and pass a file path.
- Add a "search_history" tool that uses get_events_since() to find messages from a specific time range.
- Implement thread deletion by clearing both events (replace_thread_events) and state (delete_state for each key).
- Try using LongTermMemory to store summaries of old conversations that persist beyond individual threads.
- Build a "merge_threads" tool that copies events from one thread to another using get_recent_events + append_event.
- Check out the other examples in the library to see how to use long-term memory, vector search, and cross-session personalization!
---
"""
