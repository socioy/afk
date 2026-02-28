"""
---
name: Flashcard Tutor
description: A flashcard tutor agent that remembers your study progress using AFK's memory system.
tags: [agent, runner, tools, memory, state]
---
---
This example introduces AFK's memory system -- InMemoryMemoryStore for persisting state, events, and long-term memory across sessions. The agent is a flashcard tutor that draws cards, checks answers, and tracks your score. It shows how to use put_state/get_state for thread-scoped state, append_event/get_recent_events for recording study history, and how memory differs from simple Python globals (thread isolation, async safety, swappable backends).
---
"""

import asyncio  # <- We use asyncio because all memory operations are async. This is the standard pattern for AFK agents that use memory.

from pydantic import BaseModel, Field  # <- Pydantic is used to define structured argument models for tools. This lets you specify exactly what inputs each tool expects, with types, descriptions, and validation built in.
from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior. Tl;dr: you create an Agent to define what your agent is and how it should behave, and then you use the Runner to execute it.
from afk.tools import tool  # <- The @tool decorator turns a plain Python function into a tool that an agent can call. You give it a name, description, and an args_model so the LLM knows when and how to use it.
from afk.memory import InMemoryMemoryStore, MemoryEvent, now_ms, new_id  # <- InMemoryMemoryStore is a fast, in-process memory backend for development. MemoryEvent records things that happened. now_ms() and new_id() are helpers for timestamps and unique IDs. For production, swap in SQLiteMemoryStore or RedisMemoryStore with zero code changes.


# --- Memory setup ---

THREAD_ID = "flashcard-session-1"  # <- Thread ID scopes all memory to this session. Different threads are fully isolated -- like separate browser tabs. This is how memory differs from simple Python globals: it's scoped, async-safe, and backend-swappable.

memory = InMemoryMemoryStore()  # <- In-memory store for development. For production, use SQLiteMemoryStore or RedisMemoryStore -- same API, just swap the class. All state lives in this object and is isolated by thread_id.


# --- Flashcard deck ---

flashcards = [  # <- A simple list of flashcard dicts. In a real app you might load these from a file or database.
    {"front": "What is the capital of France?", "back": "Paris"},
    {"front": "What year did World War II end?", "back": "1945"},
    {"front": "What is the chemical symbol for gold?", "back": "Au"},
    {"front": "Who wrote Romeo and Juliet?", "back": "William Shakespeare"},
    {"front": "What is the largest planet in our solar system?", "back": "Jupiter"},
    {"front": "What is the speed of light (approx)?", "back": "300,000 km/s"},
    {"front": "What element has atomic number 1?", "back": "Hydrogen"},
    {"front": "In what year was the first iPhone released?", "back": "2007"},
]


# --- Tool argument schemas ---

class DrawCardArgs(BaseModel):  # <- Even tools with no user-facing arguments need an args_model. This is an empty model that tells the LLM "this tool takes no parameters".
    pass


class CheckAnswerArgs(BaseModel):  # <- The check_flashcard tool needs the user's answer as input. Field(description=...) helps the LLM understand what to pass.
    answer: str = Field(description="The user's answer to the current flashcard question")


class GetProgressArgs(BaseModel):  # <- Another no-argument tool -- it reads everything it needs from memory.
    pass


# --- Tool definitions ---

@tool(args_model=DrawCardArgs, name="draw_card", description="Draw the next flashcard for the user to answer")  # <- This tool reads the current card index from memory, picks the next card, and saves the updated index back. All state lives in MemoryStore, not in Python globals.
async def draw_card(args: DrawCardArgs) -> str:
    # Get current card index from memory (returns the raw value, or None if not set)
    index = await memory.get_state(thread_id=THREAD_ID, key="card_index")  # <- get_state returns JsonValue | None. If the key doesn't exist yet, you get None.
    if index is None:
        index = 0  # <- First time drawing a card -- start at the beginning of the deck.

    if index >= len(flashcards):
        index = 0  # <- Wrap around to the start when we've gone through all cards.

    card = flashcards[index]

    await memory.put_state(thread_id=THREAD_ID, key="card_index", value=index + 1)  # <- Save the next index so the next draw_card picks up where we left off. put_state overwrites any previous value for the same key.
    await memory.put_state(thread_id=THREAD_ID, key="current_card", value=card)  # <- Save the current card so check_flashcard knows what question was asked. This is how tools communicate through memory instead of global variables.

    return f"Question: {card['front']}"


@tool(args_model=CheckAnswerArgs, name="check_flashcard", description="Check the user's answer against the current flashcard")  # <- This tool reads the current card from memory, compares the answer, records the attempt as a MemoryEvent, and updates the running score.
async def check_flashcard(args: CheckAnswerArgs) -> str:
    card = await memory.get_state(thread_id=THREAD_ID, key="current_card")  # <- Read which card is currently active. If the user hasn't drawn a card yet, this will be None.
    if not card:
        return "No card is active right now. Draw a card first!"

    correct = args.answer.lower().strip() in card["back"].lower()  # <- Simple substring match. "paris" matches "Paris", "shakespeare" matches "William Shakespeare". Good enough for a tutorial!

    # Record the attempt as a memory event -- this creates an audit trail of every answer
    event = MemoryEvent(  # <- MemoryEvent is a frozen dataclass. Every field is required except tags. The type must be one of: "tool_call", "tool_result", "message", "system", "trace".
        id=new_id("evt"),  # <- new_id("evt") generates a unique ID like "evt_a1b2c3d4...". Guarantees no collisions.
        thread_id=THREAD_ID,  # <- Events are scoped to a thread, just like state.
        user_id="student",  # <- Identifies who triggered this event. Could be a real user ID in production.
        type="trace",  # <- "trace" is the right event type for application-level tracking. Other types like "tool_call" and "message" are used internally by the runner.
        timestamp=now_ms(),  # <- Millisecond timestamp. now_ms() is a convenience helper from afk.memory.
        payload={  # <- Arbitrary JSON-serializable data. Store whatever you need for analytics, debugging, or replay.
            "question": card["front"],
            "correct_answer": card["back"],
            "user_answer": args.answer,
            "correct": correct,
        },
    )
    await memory.append_event(event)  # <- Appends to the thread's event log. Events are stored in chronological order and can be retrieved later with get_recent_events.

    # Update the running score in state
    score = await memory.get_state(thread_id=THREAD_ID, key="score")  # <- Read the current score dict, or None if this is the first attempt.
    if score is None:
        score = {"correct": 0, "total": 0}

    score["total"] += 1
    if correct:
        score["correct"] += 1

    await memory.put_state(thread_id=THREAD_ID, key="score", value=score)  # <- Overwrite the score with the updated values.

    if correct:
        return f"Correct! The answer is: {card['back']}. Score: {score['correct']}/{score['total']}"
    return f"Not quite. The correct answer was: {card['back']}. Score: {score['correct']}/{score['total']}"


@tool(args_model=GetProgressArgs, name="get_progress", description="Show the user's study progress and recent answer history")  # <- This tool reads from both state (score) and events (recent attempts) to give a full progress report. It shows the difference between state (current snapshot) and events (historical log).
async def get_progress(args: GetProgressArgs) -> str:
    score = await memory.get_state(thread_id=THREAD_ID, key="score")  # <- State gives us the current score snapshot.
    if score is None:
        return "No progress yet! Draw a card and start studying."

    events = await memory.get_recent_events(thread_id=THREAD_ID, limit=10)  # <- get_recent_events returns up to `limit` events in chronological order (oldest first). We use this to build a recent answer history.

    # Build a summary of recent attempts from the event log
    history_lines = []
    for evt in events:
        if evt.type == "trace" and "question" in evt.payload:  # <- Filter to only our flashcard trace events. Other event types (tool_call, message, etc.) might be mixed in if the runner records them too.
            status = "correct" if evt.payload.get("correct") else "wrong"
            history_lines.append(f"  - {evt.payload['question']} -> {status}")

    history = "\n".join(history_lines) if history_lines else "  (no attempts recorded yet)"

    return (
        f"Score: {score['correct']}/{score['total']} "
        f"({score['correct'] / score['total'] * 100:.0f}% accuracy)\n"
        f"Recent attempts:\n{history}"
    )


# --- Agent setup ---

tutor = Agent(
    name="flashcard_tutor",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a friendly flashcard tutor. Help the user study by drawing flashcards and checking their answers.

    How to interact:
    - When the user wants to study or says "quiz", use the draw_card tool to show them a question.
    - When the user gives an answer, use the check_flashcard tool to verify it.
    - When the user asks about their progress or says "progress"/"score", use the get_progress tool.
    - Be encouraging! Celebrate correct answers and gently guide them when wrong.
    - After checking an answer, ask if they want to continue with another card.

    Keep your responses concise and friendly.
    """,  # <- Instructions that guide the agent's behavior. The agent will choose the right tool based on these instructions and the user's message.
    tools=[draw_card, check_flashcard, get_progress],  # <- Pass all three tools to the agent. The LLM will automatically pick the right one based on what the user asks. draw_card presents questions, check_flashcard verifies answers, get_progress shows stats.
)
runner = Runner()


# --- Main loop (async because memory operations require it) ---

async def main():
    await memory.setup()  # <- Initialize the memory store. This MUST be called before any get_state/put_state/append_event calls. Forgetting this raises RuntimeError. You can also use `async with memory:` as a context manager.

    print("[flashcard_tutor] > Welcome! I'm your flashcard tutor.")
    print("[flashcard_tutor] > Type 'quiz' to draw a card, answer questions, or type 'progress' to see your stats.")
    print("[flashcard_tutor] > Type 'quit' to exit.\n")

    while True:  # <- A simple conversation loop. Each iteration takes user input, runs the agent, and prints the response. The agent's memory persists across iterations because the MemoryStore holds state for the entire session.
        user_input = input("[] > ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        response = await runner.run(
            tutor, user_message=user_input
        )  # <- Run the agent asynchronously using the Runner. We use `await runner.run(...)` instead of `runner.run_sync(...)` because we're already inside an async function (memory requires async).

        print(
            f"[flashcard_tutor] > {response.final_text}\n"
        )  # <- Print the agent's response to the console.

    # Show final score before exiting
    score = await memory.get_state(thread_id=THREAD_ID, key="score")
    if score:
        print(f"\n[flashcard_tutor] > Final score: {score['correct']}/{score['total']}. Great studying!")
    else:
        print("\n[flashcard_tutor] > See you next time!")

    await memory.close()  # <- Clean up the memory store. Always pair setup() with close(). For production stores (SQLite, Redis, Postgres) this releases connections and flushes buffers.


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() is the standard way to start an async main function. This is required because memory operations (get_state, put_state, append_event) are all async.



"""
---
Tl;dr: This example creates a flashcard tutor agent that uses AFK's InMemoryMemoryStore to persist study progress across a conversation. It demonstrates three core memory operations: put_state/get_state for thread-scoped key-value state (card index, current card, score), append_event for recording a chronological log of study attempts, and get_recent_events for reading that history back. All state is isolated by thread_id, making it safe for multi-session use. The agent uses three tools (draw_card, check_flashcard, get_progress) that communicate through memory rather than Python globals. The async main loop shows the setup/close lifecycle and how to use runner.run() in an async context.
---
---
What's next?
- Try changing THREAD_ID to see how different threads have completely isolated state -- start two sessions and watch their scores stay separate.
- Swap InMemoryMemoryStore for SQLiteMemoryStore to persist progress across program restarts. The API is identical -- just change the class name.
- Add a "hint" tool that uses get_state to read the current card and gives a partial answer.
- Use get_events_since() to build a spaced-repetition algorithm that prioritizes cards you got wrong.
- Check out the other examples in the library to see how to use long-term memory (LongTermMemory) for cross-session user profiles and personalization!
---
"""
