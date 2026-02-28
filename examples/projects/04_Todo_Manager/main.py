"""
---
name: Todo Manager
description: A stateful todo manager agent that can add, list, complete, and remove tasks.
tags: [agent, runner, tools, state]
---
---
This example demonstrates how tools can maintain **state** using Python globals.
The agent manages a todo list with full CRUD operations -- add, list, complete, and remove.
Each tool reads from and writes to a shared list that persists across tool calls within a session.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic is used to define the shape of tool arguments. Each tool gets its own args model so the LLM knows exactly what parameters to pass.

from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior. Tl;dr: you create an Agent to define what your agent is and how it should behave, and then you use the Runner to execute it.
from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.tools import tool  # <- The @tool decorator turns a plain Python function into an AFK Tool. You provide an args_model (a Pydantic BaseModel) so the framework knows how to validate and pass arguments from the LLM.

# ---------------------------------------------------------------------------
# Shared state -- this list persists across tool calls within a session
# ---------------------------------------------------------------------------
todos: list[dict] = []  # <- Each todo is {"id": int, "task": str, "done": bool}
_next_id = 1  # <- Auto-incrementing ID counter for new todos


# ---------------------------------------------------------------------------
# Tool 1: Add a todo
# ---------------------------------------------------------------------------
class AddTodoArgs(BaseModel):  # <- Every tool needs an args model. This one takes a single string field -- the task description.
    task: str = Field(description="The task description to add")


@tool(args_model=AddTodoArgs, name="add_todo", description="Add a new todo item to the list")  # <- The @tool decorator registers this function as a tool the agent can call. We give it a name, description (shown to the LLM), and the args model that defines its parameters.
def add_todo(args: AddTodoArgs) -> str:  # <- The function receives a validated instance of AddTodoArgs. Returning a string sends the result back to the agent as the tool output.
    global _next_id  # <- We use a global counter to assign unique IDs to each todo. This is a simple approach for an example -- in production you might use a database.
    todo = {"id": _next_id, "task": args.task, "done": False}
    todos.append(todo)
    _next_id += 1
    return f"Added todo #{todo['id']}: \"{todo['task']}\""


# ---------------------------------------------------------------------------
# Tool 2: List all todos
# ---------------------------------------------------------------------------
class EmptyArgs(BaseModel):  # <- Some tools don't need any arguments. We still need an args model, so we use an empty one. This tells the LLM "just call this tool, no parameters needed."
    pass


@tool(args_model=EmptyArgs, name="list_todos", description="List all current todo items")
def list_todos(args: EmptyArgs) -> str:
    if not todos:
        return "Your todo list is empty. Try adding a task!"
    lines = []
    for t in todos:
        status = "x" if t["done"] else " "  # <- Show [x] for completed, [ ] for pending
        lines.append(f"  {t['id']}. [{status}] {t['task']}")
    return "Here are your todos:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: Complete a todo
# ---------------------------------------------------------------------------
class CompleteTodoArgs(BaseModel):  # <- This args model takes an integer ID. The Field description helps the LLM understand what value to pass.
    todo_id: int = Field(description="The ID of the todo to mark as complete")


@tool(args_model=CompleteTodoArgs, name="complete_todo", description="Mark a todo item as done")
def complete_todo(args: CompleteTodoArgs) -> str:
    for t in todos:
        if t["id"] == args.todo_id:
            t["done"] = True  # <- Mutating the shared state -- this change is visible to all subsequent tool calls in the session.
            return f"Marked todo #{t['id']} \"{t['task']}\" as done!"
    return f"No todo found with ID {args.todo_id}."


# ---------------------------------------------------------------------------
# Tool 4: Remove a todo
# ---------------------------------------------------------------------------
class RemoveTodoArgs(BaseModel):
    todo_id: int = Field(description="The ID of the todo to remove")


@tool(args_model=RemoveTodoArgs, name="remove_todo", description="Remove a todo item from the list")
def remove_todo(args: RemoveTodoArgs) -> str:
    for i, t in enumerate(todos):
        if t["id"] == args.todo_id:
            removed = todos.pop(i)  # <- Removing from the shared list. The todo is gone for all future tool calls.
            return f"Removed todo #{removed['id']}: \"{removed['task']}\""
    return f"No todo found with ID {args.todo_id}."


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------
todo_manager = Agent(
    name="todo-manager",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a helpful todo manager assistant. You help users manage their task list.

    You have four tools available:
    - add_todo: Add a new task to the list
    - list_todos: Show all current tasks
    - complete_todo: Mark a task as done by its ID
    - remove_todo: Remove a task from the list by its ID

    When the user asks to add a task, use the add_todo tool.
    When the user asks to see, show, or list their tasks, use the list_todos tool.
    When the user asks to complete, finish, or mark a task as done, use the complete_todo tool.
    When the user asks to remove or delete a task, use the remove_todo tool.

    Always respond concisely. After performing an action, report what you did.
    If the user's request is ambiguous, ask for clarification.

    """,  # <- Instructions that guide the agent's behavior. This is where you can specify the agent's goals, constraints, and any other information that will help it perform its task.
    tools=[add_todo, list_todos, complete_todo, remove_todo],  # <- Attach all four tools to the agent. The agent can call any of these tools during a conversation to manage the todo list.
)
runner = Runner()  # <- Create a Runner instance to execute the agent. The runner handles the request-response loop, tool dispatch, and state management.

if __name__ == "__main__":
    print("[todo-manager] > Hi! I'm your todo manager. I can add, list, complete, and remove tasks for you.")
    print("[todo-manager] > Try saying things like 'Add buy groceries' or 'Show my tasks'")
    print("[todo-manager] > Type 'quit' to exit.\n")

    while True:  # <- Conversation loop -- because this is a stateful app, we keep taking input so the user can manage their list across multiple turns.
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agent.

        if user_input.lower() in ("quit", "exit", "q"):
            print("[todo-manager] > Goodbye! Stay productive!")
            break

        response = runner.run_sync(
            todo_manager, user_message=user_input
        )  # <- Run the agent synchronously using the Runner. We pass the user's input as a message to the agent. The agent may call one or more tools before responding.

        print(
            f"[todo-manager] > {response.final_text}\n"
        )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.



"""
---
Tl;dr: This example creates a stateful todo manager agent that uses the "ollama_chat/gpt-oss:20b" model and four tools (add, list, complete, remove) to manage a task list. The tools share state through a Python list that persists across calls within a session. The agent is guided by instructions that tell it when to use each tool. A conversation loop lets the user interact with the agent over multiple turns, building up and managing their todo list.
---
---
What's next?
- Try adding more tool features like editing a task's description, setting priorities, or filtering by completion status.
- Experiment with the agent's instructions to change how it interprets ambiguous requests -- for example, should "done with groceries" complete the task or remove it?
- Notice how state is stored in a Python global -- consider how you might persist state to a file or database for a production application.
- Check out the other examples in the library to see how to use tool hooks, middleware, and multi-agent systems!
---
"""
