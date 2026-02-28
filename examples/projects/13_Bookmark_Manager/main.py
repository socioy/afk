"""
---
name: Bookmark Manager
description: A bookmark manager agent that demonstrates RunnerConfig and RunnerDebugConfig for controlling runner behavior and debug output.
tags: [agent, runner, runner-config, tools, debug]
---
---
This example introduces **RunnerConfig** -- how to configure the Runner with debug mode,
interaction settings, and other runtime options. The agent manages a bookmark collection
with full CRUD operations (add, list, search, remove, export), while RunnerDebugConfig
controls how much detail you see during agent execution.
---
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field  # <- Pydantic is used to define the shape of tool arguments. Each tool gets its own args model so the LLM knows exactly what parameters to pass.

from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior. Tl;dr: you create an Agent to define what your agent is and how it should behave, and then you use the Runner to execute it.
from afk.core import Runner, RunnerConfig, RunnerDebugConfig  # <- Runner executes agents. RunnerConfig lets you customize runner behavior (debug mode, interaction settings, etc.). RunnerDebugConfig fine-tunes *what* debug information you see.
from afk.tools import tool  # <- The @tool decorator turns a plain Python function into an AFK Tool. You provide an args_model (a Pydantic BaseModel) so the framework knows how to validate and pass arguments from the LLM.

# ---------------------------------------------------------------------------
# Debug & Runner configuration -- the key concept for this example
# ---------------------------------------------------------------------------
debug_config = RunnerDebugConfig(
    enabled=True,          # <- Turn on debug output. When True, the runner enriches run events with debug metadata.
    verbosity="detailed",  # <- "basic", "detailed", or "trace". Controls how much information appears in debug output. "detailed" is a good middle ground.
)

config = RunnerConfig(
    debug=True,                  # <- Enable debug mode on the runner. This is the master switch -- without it, debug_config is ignored.
    debug_config=debug_config,   # <- Attach the debug settings. These control the format and detail level of debug output during agent execution.
)

# ---------------------------------------------------------------------------
# Shared state -- this list persists across tool calls within a session
# ---------------------------------------------------------------------------
bookmarks: list[dict] = []  # <- Each bookmark is {"id": int, "url": str, "title": str, "tags": list[str], "created_at": str}
_next_id = 1  # <- Auto-incrementing ID counter for new bookmarks


# ---------------------------------------------------------------------------
# Tool 1: Add a bookmark
# ---------------------------------------------------------------------------
class AddBookmarkArgs(BaseModel):  # <- Every tool needs an args model. This one takes a URL, a title, and optional tags for the bookmark.
    url: str = Field(description="The URL of the bookmark")
    title: str = Field(description="A short title for the bookmark")
    tags: list[str] = Field(default_factory=list, description="Optional tags to categorize the bookmark (e.g. ['python', 'tutorial'])")


@tool(args_model=AddBookmarkArgs, name="add_bookmark", description="Add a new bookmark with a URL, title, and optional tags")  # <- The @tool decorator registers this function as a tool the agent can call. We give it a name, description (shown to the LLM), and the args model that defines its parameters.
def add_bookmark(args: AddBookmarkArgs) -> str:  # <- The function receives a validated instance of AddBookmarkArgs. Returning a string sends the result back to the agent as the tool output.
    global _next_id  # <- We use a global counter to assign unique IDs to each bookmark. This is a simple approach for an example -- in production you might use a database.
    bookmark = {
        "id": _next_id,
        "url": args.url,
        "title": args.title,
        "tags": args.tags,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),  # <- Timestamp when the bookmark was created, stored as a human-readable string.
    }
    bookmarks.append(bookmark)
    _next_id += 1
    tag_str = f" [{', '.join(bookmark['tags'])}]" if bookmark["tags"] else ""  # <- Format tags for display, or show nothing if there are no tags.
    return f"Added bookmark #{bookmark['id']}: \"{bookmark['title']}\" ({bookmark['url']}){tag_str}"


# ---------------------------------------------------------------------------
# Tool 2: List all bookmarks
# ---------------------------------------------------------------------------
class EmptyArgs(BaseModel):  # <- Some tools don't need any arguments. We still need an args model, so we use an empty one. This tells the LLM "just call this tool, no parameters needed."
    pass


@tool(args_model=EmptyArgs, name="list_bookmarks", description="List all saved bookmarks")
def list_bookmarks(args: EmptyArgs) -> str:
    if not bookmarks:
        return "Your bookmark collection is empty. Try adding one!"
    lines = []
    for b in bookmarks:
        tag_str = f" [{', '.join(b['tags'])}]" if b["tags"] else ""  # <- Show tags inline if present.
        lines.append(f"  {b['id']}. {b['title']} - {b['url']}{tag_str} (added {b['created_at']})")
    return "Here are your bookmarks:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: Search bookmarks by query or tag
# ---------------------------------------------------------------------------
class SearchBookmarksArgs(BaseModel):  # <- This args model takes a query string. The search matches against titles, URLs, and tags.
    query: str = Field(description="Search term to match against bookmark titles, URLs, or tags")


@tool(args_model=SearchBookmarksArgs, name="search_bookmarks", description="Search bookmarks by title, URL, or tag")
def search_bookmarks(args: SearchBookmarksArgs) -> str:
    query_lower = args.query.lower()  # <- Case-insensitive matching so "Python" finds "python".
    matches = []
    for b in bookmarks:
        if (
            query_lower in b["title"].lower()
            or query_lower in b["url"].lower()
            or any(query_lower in tag.lower() for tag in b["tags"])  # <- Also match against any of the bookmark's tags.
        ):
            matches.append(b)
    if not matches:
        return f"No bookmarks found matching \"{args.query}\"."
    lines = []
    for b in matches:
        tag_str = f" [{', '.join(b['tags'])}]" if b["tags"] else ""
        lines.append(f"  {b['id']}. {b['title']} - {b['url']}{tag_str}")
    return f"Found {len(matches)} bookmark(s) matching \"{args.query}\":\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: Remove a bookmark
# ---------------------------------------------------------------------------
class RemoveBookmarkArgs(BaseModel):  # <- This args model takes an integer ID. The Field description helps the LLM understand what value to pass.
    bookmark_id: int = Field(description="The ID of the bookmark to remove")


@tool(args_model=RemoveBookmarkArgs, name="remove_bookmark", description="Remove a bookmark by its ID")
def remove_bookmark(args: RemoveBookmarkArgs) -> str:
    for i, b in enumerate(bookmarks):
        if b["id"] == args.bookmark_id:
            removed = bookmarks.pop(i)  # <- Removing from the shared list. The bookmark is gone for all future tool calls.
            return f"Removed bookmark #{removed['id']}: \"{removed['title']}\" ({removed['url']})"
    return f"No bookmark found with ID {args.bookmark_id}."


# ---------------------------------------------------------------------------
# Tool 5: Export all bookmarks
# ---------------------------------------------------------------------------
@tool(args_model=EmptyArgs, name="export_bookmarks", description="Export all bookmarks as a formatted summary")
def export_bookmarks(args: EmptyArgs) -> str:
    if not bookmarks:
        return "Nothing to export -- your bookmark collection is empty."
    lines = ["# Bookmarks Export", ""]  # <- Build a simple markdown-style export.
    for b in bookmarks:
        tag_str = f"  Tags: {', '.join(b['tags'])}" if b["tags"] else ""
        lines.append(f"- **{b['title']}**")
        lines.append(f"  URL: {b['url']}")
        if tag_str:
            lines.append(tag_str)
        lines.append(f"  Added: {b['created_at']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------
bookmark_manager = Agent(
    name="bookmark-manager",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a helpful bookmark manager assistant. You help users save, organize,
    find, and manage their web bookmarks.

    You have five tools available:
    - add_bookmark: Save a new bookmark with a URL, title, and optional tags
    - list_bookmarks: Show all saved bookmarks
    - search_bookmarks: Search bookmarks by title, URL, or tag
    - remove_bookmark: Delete a bookmark by its ID
    - export_bookmarks: Export all bookmarks as a formatted summary

    When the user asks to save or add a bookmark, use the add_bookmark tool.
    When the user wants to see or list bookmarks, use the list_bookmarks tool.
    When the user wants to find a specific bookmark, use the search_bookmarks tool.
    When the user wants to delete a bookmark, use the remove_bookmark tool.
    When the user wants an export or summary, use the export_bookmarks tool.

    Always respond concisely. After performing an action, report what you did.
    If the user provides a URL without a title, suggest a reasonable title.
    If the user's request is ambiguous, ask for clarification.

    """,  # <- Instructions that guide the agent's behavior. This is where you can specify the agent's goals, constraints, and any other information that will help it perform its task.
    tools=[add_bookmark, list_bookmarks, search_bookmarks, remove_bookmark, export_bookmarks],  # <- Attach all five tools to the agent. The agent can call any of these tools during a conversation to manage bookmarks.
)
runner = Runner(config=config)  # <- Pass config to the Runner. This is the key difference from earlier examples -- we configure the runner with debug mode and custom settings instead of using defaults.

if __name__ == "__main__":
    print("[bookmark-manager] > Hi! I'm your bookmark manager. I can save, search, and organize your web bookmarks.")
    print("[bookmark-manager] > Try saying things like 'Save https://docs.python.org as Python Docs with tags python, reference' or 'Search for python'")
    print("[bookmark-manager] > Type 'quit' to exit.\n")

    while True:  # <- Conversation loop -- because this is a stateful app, we keep taking input so the user can manage their bookmarks across multiple turns.
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agent.

        if user_input.lower() in ("quit", "exit", "q"):
            print("[bookmark-manager] > Goodbye! Happy browsing!")
            break

        response = runner.run_sync(
            bookmark_manager, user_message=user_input
        )  # <- Run the agent synchronously using the Runner. Because we passed config with debug=True, the runner will emit debug metadata during execution. We pass the user's input as a message to the agent. The agent may call one or more tools before responding.

        print(
            f"[bookmark-manager] > {response.final_text}\n"
        )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.



"""
---
Tl;dr: This example creates a bookmark manager agent that uses the "ollama_chat/gpt-oss:20b" model and five tools (add, list, search, remove, export) to manage a bookmark collection. The key new concept is **RunnerConfig** -- instead of using a bare `Runner()`, we create a `RunnerConfig` with `debug=True` and a `RunnerDebugConfig` that controls verbosity. This means the runner emits detailed debug information during agent execution, which is invaluable for understanding what the agent is doing and troubleshooting issues. The agent maintains shared state through a Python list that persists across calls within a session.
---
---
What's next?
- Try changing `verbosity` in RunnerDebugConfig from "detailed" to "basic" or "trace" and compare the output to see how it affects what you see during agent execution.
- Experiment with other RunnerConfig options like `interaction_mode` or `sanitize_tool_output` to see how they change runner behavior.
- Try turning off `debug` in RunnerConfig and notice how the output becomes cleaner -- debug mode is great for development but you may want it off in production.
- Add more bookmark features like editing a bookmark's title or tags, or organizing bookmarks into folders.
- Check out the other examples in the library to see how to use tool hooks, middleware, and multi-agent systems!
---
"""
