"""
---
name: File Organizer
description: An async-tools agent that organizes files by category in a simulated filesystem.
tags: [agent, runner, tools, async-tools]
---
---
This example demonstrates how to define **async tools** -- tools declared with `async def` that can
perform I/O-bound operations (like reading directories or moving files on disk) without blocking.
The agent organizes files from a downloads folder into category folders (documents, images, music,
videos, other) using a simulated in-memory filesystem.

Key concepts:
- Async tool functions with `async def` (the framework detects and awaits them automatically)
- When to use async vs sync tools (async for I/O-bound work like file/network ops; sync for CPU-bound work)
- Using `asyncio.run()` with the async Runner API
- Practical file organization use case with stateful tools
---
"""

import asyncio  # <- asyncio is Python's built-in library for writing asynchronous code. We use it here to run the async Runner API and to simulate I/O delays in our tools.
import os  # <- os.path.splitext is used to extract file extensions for categorization.
from pydantic import BaseModel, Field  # <- Pydantic is used to define structured argument models for tools. This lets you specify exactly what inputs each tool expects, with types, descriptions, and validation built in.
from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior. Tl;dr: you create an Agent to define what your agent is and how it should behave, and then you use the Runner to execute it.
from afk.tools import tool  # <- The @tool decorator turns a plain Python function into an AFK Tool. You provide an args_model (a Pydantic BaseModel) so the framework knows how to validate and pass arguments from the LLM.


# ---------------------------------------------------------------------------
# Simulated filesystem -- a dict of folder names to file lists
# ---------------------------------------------------------------------------
filesystem: dict[str, list[str]] = {  # <- This dict acts as our in-memory filesystem. Each key is a folder name, each value is a list of filenames. Tools read from and write to this shared state.
    "downloads": [
        "report.pdf",
        "photo.jpg",
        "song.mp3",
        "notes.txt",
        "video.mp4",
        "spreadsheet.xlsx",
        "archive.zip",
        "presentation.pptx",
    ],
    "documents": [],
    "images": [],
    "music": [],
    "videos": [],
    "other": [],
}

FILE_CATEGORIES: dict[str, str] = {  # <- Maps file extensions to their destination folder. When organizing, the agent looks up a file's extension here to decide where it goes.
    ".pdf": "documents",
    ".doc": "documents",
    ".docx": "documents",
    ".txt": "documents",
    ".xlsx": "documents",
    ".pptx": "documents",
    ".jpg": "images",
    ".jpeg": "images",
    ".png": "images",
    ".gif": "images",
    ".svg": "images",
    ".mp3": "music",
    ".wav": "music",
    ".flac": "music",
    ".mp4": "videos",
    ".avi": "videos",
    ".mov": "videos",
    ".mkv": "videos",
}


# ---------------------------------------------------------------------------
# Tool 1: List files in a folder
# ---------------------------------------------------------------------------
class ListFilesArgs(BaseModel):  # <- A Pydantic model that defines the arguments for the list_files tool. The LLM will populate these fields based on the user's request.
    folder: str = Field(description="Folder name to list files from")


@tool(args_model=ListFilesArgs, name="list_files", description="List all files in a given folder")  # <- The @tool decorator registers this function as a tool the agent can call. The name and description help the LLM decide which tool to use.
async def list_files(args: ListFilesArgs) -> str:  # <- async def makes this an async tool. The framework detects the coroutine and awaits it automatically. Use async tools for I/O-bound work (file ops, network calls, database queries) so other tasks aren't blocked while waiting.
    await asyncio.sleep(0.1)  # <- Simulates I/O delay (e.g., reading a directory from disk). In a real app, this would be an actual async file or network call.
    if args.folder not in filesystem:
        return f"Error: Folder '{args.folder}' does not exist."
    files = filesystem[args.folder]
    if not files:
        return f"Folder '{args.folder}' is empty."
    file_list = "\n".join(f"  - {f}" for f in files)
    return f"Files in '{args.folder}' ({len(files)} files):\n{file_list}"


# ---------------------------------------------------------------------------
# Tool 2: Move a file between folders
# ---------------------------------------------------------------------------
class MoveFileArgs(BaseModel):  # <- Each tool gets its own args model. This one takes three fields: the filename, the source folder, and the destination folder.
    filename: str = Field(description="Name of the file to move")
    from_folder: str = Field(description="Source folder")
    to_folder: str = Field(description="Destination folder")


@tool(args_model=MoveFileArgs, name="move_file", description="Move a file from one folder to another")
async def move_file(args: MoveFileArgs) -> str:  # <- Another async tool. The await asyncio.sleep simulates the I/O cost of actually moving a file on disk.
    await asyncio.sleep(0.05)  # <- Simulates I/O delay for a file move operation.
    if args.from_folder not in filesystem:
        return f"Error: Source folder '{args.from_folder}' does not exist."
    if args.to_folder not in filesystem:
        return f"Error: Destination folder '{args.to_folder}' does not exist."
    if args.filename not in filesystem[args.from_folder]:
        return f"Error: File '{args.filename}' not found in '{args.from_folder}'."
    filesystem[args.from_folder].remove(args.filename)  # <- Remove the file from the source folder.
    filesystem[args.to_folder].append(args.filename)  # <- Add the file to the destination folder.
    return f"Moved '{args.filename}' from '{args.from_folder}' to '{args.to_folder}'."


# ---------------------------------------------------------------------------
# Tool 3: Auto-organize all files from a folder by category
# ---------------------------------------------------------------------------
class AutoOrganizeArgs(BaseModel):  # <- This tool takes a single folder name and automatically categorizes every file in it based on extension.
    folder: str = Field(description="Folder to organize files from (files will be moved to category folders)")


@tool(args_model=AutoOrganizeArgs, name="auto_organize", description="Automatically organize all files from a folder into category folders based on file extension")
async def auto_organize(args: AutoOrganizeArgs) -> str:  # <- This is the most powerful tool: it iterates over all files, looks up each extension in FILE_CATEGORIES, and moves files to the right folder. Uncategorized extensions go to "other".
    await asyncio.sleep(0.1)  # <- Simulates I/O delay for scanning and moving multiple files.
    if args.folder not in filesystem:
        return f"Error: Folder '{args.folder}' does not exist."
    files = list(filesystem[args.folder])  # <- Copy the list so we can modify the original safely while iterating.
    if not files:
        return f"Folder '{args.folder}' is already empty. Nothing to organize."
    moved: list[str] = []
    for filename in files:
        _, ext = os.path.splitext(filename)  # <- Extract the file extension (e.g., ".pdf" from "report.pdf").
        ext = ext.lower()
        dest = FILE_CATEGORIES.get(ext, "other")  # <- Look up the destination folder. Defaults to "other" for unrecognized extensions.
        if dest not in filesystem:
            filesystem[dest] = []  # <- Create the destination folder if it doesn't exist yet.
        filesystem[args.folder].remove(filename)
        filesystem[dest].append(filename)
        moved.append(f"  - {filename} -> {dest}")
    summary = "\n".join(moved)
    return f"Organized {len(moved)} files from '{args.folder}':\n{summary}"


# ---------------------------------------------------------------------------
# Tool 4: Create a new folder
# ---------------------------------------------------------------------------
class CreateFolderArgs(BaseModel):  # <- Simple args model with a single field for the new folder name.
    folder_name: str = Field(description="Name of the new folder to create")


@tool(args_model=CreateFolderArgs, name="create_folder", description="Create a new empty folder")
async def create_folder(args: CreateFolderArgs) -> str:  # <- Async even for a simple operation, because in a real app creating a directory is an I/O call to the OS.
    await asyncio.sleep(0.05)  # <- Simulates I/O delay for creating a directory.
    if args.folder_name in filesystem:
        return f"Folder '{args.folder_name}' already exists."
    filesystem[args.folder_name] = []  # <- Add a new empty folder to our simulated filesystem.
    return f"Created folder '{args.folder_name}'."


# ---------------------------------------------------------------------------
# Tool 5: Get statistics about a folder's contents
# ---------------------------------------------------------------------------
class GetFolderStatsArgs(BaseModel):  # <- Args model for the stats tool. Takes a single folder name.
    folder: str = Field(description="Folder name to get statistics for")


@tool(args_model=GetFolderStatsArgs, name="get_folder_stats", description="Get statistics about files in a folder including count and breakdown by type")
async def get_folder_stats(args: GetFolderStatsArgs) -> str:  # <- Async tool that reads the folder contents and produces a summary with counts by extension.
    await asyncio.sleep(0.05)  # <- Simulates I/O delay for stating files.
    if args.folder not in filesystem:
        return f"Error: Folder '{args.folder}' does not exist."
    files = filesystem[args.folder]
    if not files:
        return f"Folder '{args.folder}' is empty (0 files)."
    ext_counts: dict[str, int] = {}  # <- Count files by extension to give the user a breakdown.
    for filename in files:
        _, ext = os.path.splitext(filename)
        ext = ext.lower() if ext else "(no extension)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    breakdown = "\n".join(f"  {ext}: {count} file(s)" for ext, count in sorted(ext_counts.items()))
    return f"Folder '{args.folder}' -- {len(files)} file(s):\n{breakdown}"


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------
organizer = Agent(
    name="file-organizer",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a helpful file organizer assistant. You help users manage and organize their files
    across folders in their filesystem.

    You have five tools available:
    - list_files: List all files in a folder
    - move_file: Move a single file from one folder to another
    - auto_organize: Automatically sort all files from a folder into category folders (documents, images, music, videos, other) based on file extension
    - create_folder: Create a new empty folder
    - get_folder_stats: Get statistics about a folder (file count and breakdown by type)

    The filesystem has these folders: downloads, documents, images, music, videos, other.
    The downloads folder starts with a mix of files that need organizing.

    When the user asks to organize files, prefer using auto_organize for bulk operations.
    When the user asks about a specific file, use move_file for targeted moves.
    After organizing, show the user what changed by listing the affected folders or using get_folder_stats.

    Be concise and report what actions you took.
    """,  # <- Instructions that guide the agent's behavior. This is where you can specify the agent's goals, constraints, and any other information that will help it perform its task.
    tools=[list_files, move_file, auto_organize, create_folder, get_folder_stats],  # <- Attach all five async tools to the agent. The LLM will automatically pick the right one based on the user's request. All tools are async, so they won't block the event loop during I/O.
)
runner = Runner()  # <- Create a Runner instance to execute the agent. The runner handles the request-response loop, tool dispatch, and state management.


async def main():  # <- Main async entry point. We use an async function so we can call runner.run() with await.
    """Main async entry point for the file organizer."""
    print("[file-organizer] > Hi! I'm your file organizer. I can list, move, organize, and analyze files for you.")
    print("[file-organizer] > The downloads folder has files waiting to be organized.")
    print("[file-organizer] > Try saying things like 'What files are in downloads?' or 'Organize my downloads folder'")
    print("[file-organizer] > Type 'quit' to exit.\n")

    while True:  # <- Conversation loop -- because this is a stateful app, we keep taking input so the user can manage their filesystem across multiple turns.
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agent.

        if user_input.lower() in ("quit", "exit", "q"):
            print("[file-organizer] > Goodbye! Your files are in good hands.")
            break

        response = await runner.run(
            organizer, user_message=user_input
        )  # <- Run the agent asynchronously using the Runner. This is the async version of run_sync() -- same API, but uses `await`. Because our tools are async, using the async runner lets everything run on the same event loop without blocking.

        print(
            f"\n[file-organizer] > {response.final_text}\n"
        )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.


if __name__ == "__main__":
    asyncio.run(
        main()
    )  # <- asyncio.run() is the standard way to run async code from a synchronous entry point. It creates an event loop, runs the coroutine, and cleans up when done. This is the recommended pattern for scripts that use async tools.



"""
---
Tl;dr: This example creates a file organizer agent that uses the "ollama_chat/gpt-oss:20b" model and five **async tools** (list_files, move_file, auto_organize, create_folder, get_folder_stats) to manage files in a simulated in-memory filesystem. All tools are defined with `async def`, which means they can perform I/O-bound work (simulated here with `asyncio.sleep`) without blocking the event loop. The agent categorizes files by extension (documents, images, music, videos, other) and can organize an entire folder in one command. The script uses `asyncio.run()` as the entry point and `await runner.run()` for async agent execution.
---
---
What's next?
- Try converting some tools to sync (`def` instead of `async def`) and observe that the framework handles both seamlessly -- sync tools are automatically wrapped with `asyncio.to_thread` so they don't block the event loop.
- Experiment with adding real filesystem I/O using `aiofiles` or `asyncio.to_thread(os.listdir, ...)` to make the tools work with actual files on disk.
- Add new category folders (e.g., "code" for .py/.js/.ts files) by extending the FILE_CATEGORIES dict and see how auto_organize adapts.
- Check out the other examples in the library to see how to use tool hooks, middleware, and multi-agent systems where one agent could delegate file operations to another!
---
"""
