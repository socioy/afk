"""
---
name: Expense Tracker
description: An expense tracking agent with fail-safe limits that prevent runaway loops and excessive API calls.
tags: [agent, runner, tools, fail-safe]
---
---
This example demonstrates how to use **FailSafeConfig** to set runtime safety limits on an agent.
The agent is an expense tracker with five tools (add, list, total, filter by category, remove).
A FailSafeConfig is attached to the agent to cap the number of loop steps, LLM calls, and tool
invocations per run. If any limit is reached, the runner raises an **AgentLoopLimitError** which
the conversation loop catches and reports gracefully instead of crashing.
---
"""

from datetime import date  # <- We use date.today() to automatically timestamp each expense when it is added.

from pydantic import BaseModel, Field  # <- Pydantic is used to define structured argument models for tools. This lets you specify exactly what inputs each tool expects, with types, descriptions, and validation built in.

from afk.agents import Agent, FailSafeConfig, AgentLoopLimitError  # <- Agent is the main class for creating agents. FailSafeConfig defines runtime safety limits (max steps, max LLM calls, max tool calls, etc.). AgentLoopLimitError is raised when any of those limits are exceeded during a run.
from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.tools import tool  # <- The @tool decorator turns a plain Python function into an AFK Tool. You provide an args_model (a Pydantic BaseModel) so the framework knows how to validate and pass arguments from the LLM.

# ---------------------------------------------------------------------------
# Shared state -- this list persists across tool calls within a session
# ---------------------------------------------------------------------------
expenses: list[dict] = []  # <- Each expense is {"id": int, "description": str, "amount": float, "category": str, "date": str}
_next_id = 1  # <- Auto-incrementing ID counter for new expenses

VALID_CATEGORIES = ["food", "transport", "entertainment", "bills", "other"]  # <- Allowed expense categories. If the LLM passes something outside this list, the tool maps it to "other".


# ---------------------------------------------------------------------------
# Tool 1: Add an expense
# ---------------------------------------------------------------------------
class AddExpenseArgs(BaseModel):  # <- Every tool needs an args model. This one takes a description, amount, and optional category for the expense.
    description: str = Field(description="Short description of the expense (e.g. 'lunch', 'uber ride')")
    amount: float = Field(description="The expense amount in dollars (e.g. 12.50)")
    category: str = Field(
        default="other",
        description="Expense category: food, transport, entertainment, bills, or other",
    )


@tool(args_model=AddExpenseArgs, name="add_expense", description="Add a new expense entry with a description, amount, and category")  # <- The @tool decorator registers this function as a tool the agent can call. We give it a name, description (shown to the LLM), and the args model that defines its parameters.
def add_expense(args: AddExpenseArgs) -> str:  # <- The function receives a validated instance of AddExpenseArgs. Returning a string sends the result back to the agent as the tool output.
    global _next_id  # <- We use a global counter to assign unique IDs to each expense. This is a simple approach for an example -- in production you might use a database.
    category = args.category.lower().strip()
    if category not in VALID_CATEGORIES:
        category = "other"  # <- Gracefully handle unknown categories by falling back to "other" instead of raising an error.
    expense = {
        "id": _next_id,
        "description": args.description,
        "amount": round(args.amount, 2),  # <- Round to two decimal places to keep amounts clean.
        "category": category,
        "date": date.today().isoformat(),  # <- Automatically record today's date in ISO format (YYYY-MM-DD).
    }
    expenses.append(expense)
    _next_id += 1
    return f"Added expense #{expense['id']}: \"{expense['description']}\" -- ${expense['amount']:.2f} [{expense['category']}] on {expense['date']}"


# ---------------------------------------------------------------------------
# Tool 2: List all expenses
# ---------------------------------------------------------------------------
class EmptyArgs(BaseModel):  # <- Some tools don't need any arguments. We still need an args model, so we use an empty one. This tells the LLM "just call this tool, no parameters needed."
    pass


@tool(args_model=EmptyArgs, name="list_expenses", description="List all recorded expenses with their ID, description, amount, category, and date")
def list_expenses(args: EmptyArgs) -> str:
    if not expenses:
        return "No expenses recorded yet. Try adding one!"
    lines = []
    for e in expenses:
        lines.append(
            f"  #{e['id']}  ${e['amount']:.2f}  [{e['category']}]  {e['description']}  ({e['date']})"
        )  # <- Format each expense as a readable line with currency formatting ($XX.XX).
    return "Your expenses:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: Get total of all expenses
# ---------------------------------------------------------------------------
@tool(args_model=EmptyArgs, name="get_total", description="Calculate and return the total of all recorded expenses")
def get_total(args: EmptyArgs) -> str:
    if not expenses:
        return "No expenses recorded yet. Your total is $0.00."
    total = sum(e["amount"] for e in expenses)  # <- Sum all expense amounts.
    return f"Your total expenses: ${total:.2f} across {len(expenses)} item(s)."


# ---------------------------------------------------------------------------
# Tool 4: Get expenses by category
# ---------------------------------------------------------------------------
class CategoryArgs(BaseModel):  # <- This args model takes a single category string so the user can filter expenses by type.
    category: str = Field(description="The category to filter by: food, transport, entertainment, bills, or other")


@tool(args_model=CategoryArgs, name="get_by_category", description="List all expenses that match a specific category and show their subtotal")
def get_by_category(args: CategoryArgs) -> str:
    category = args.category.lower().strip()
    if category not in VALID_CATEGORIES:
        return f"Unknown category \"{args.category}\". Valid categories are: {', '.join(VALID_CATEGORIES)}"  # <- Give the user a helpful error message listing valid options.
    matched = [e for e in expenses if e["category"] == category]
    if not matched:
        return f"No expenses found in the \"{category}\" category."
    lines = []
    for e in matched:
        lines.append(f"  #{e['id']}  ${e['amount']:.2f}  {e['description']}  ({e['date']})")
    subtotal = sum(e["amount"] for e in matched)
    return f"Expenses in \"{category}\" ({len(matched)} item(s), subtotal ${subtotal:.2f}):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: Remove an expense
# ---------------------------------------------------------------------------
class RemoveExpenseArgs(BaseModel):  # <- This args model takes an integer ID. The Field description helps the LLM understand what value to pass.
    expense_id: int = Field(description="The ID of the expense to remove")


@tool(args_model=RemoveExpenseArgs, name="remove_expense", description="Remove an expense entry by its ID")
def remove_expense(args: RemoveExpenseArgs) -> str:
    for i, e in enumerate(expenses):
        if e["id"] == args.expense_id:
            removed = expenses.pop(i)  # <- Removing from the shared list. The expense is gone for all future tool calls.
            return f"Removed expense #{removed['id']}: \"{removed['description']}\" -- ${removed['amount']:.2f}"
    return f"No expense found with ID {args.expense_id}."


# ---------------------------------------------------------------------------
# Fail-safe configuration
# ---------------------------------------------------------------------------
fail_safe = FailSafeConfig(
    max_steps=10,  # <- Maximum number of agent loop iterations per run. If the agent loops more than 10 times without finishing, the runner raises AgentLoopLimitError. This prevents infinite reasoning loops.
    max_llm_calls=15,  # <- Cap on LLM API calls per run. Each time the agent asks the model for a response counts as one call. This protects against runaway token costs.
    max_tool_calls=20,  # <- Maximum tool invocations per run. If the agent calls tools more than 20 times in a single run, the runner raises AgentLoopLimitError. This prevents pathological tool-call loops.
)


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------
tracker = Agent(
    name="expense-tracker",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a helpful expense tracking assistant. You help users record, review, and manage their expenses.

    You have five tools available:
    - add_expense: Add a new expense with a description, amount, and category (food, transport, entertainment, bills, other)
    - list_expenses: Show all recorded expenses
    - get_total: Calculate the total of all expenses
    - get_by_category: Filter expenses by category and show the subtotal
    - remove_expense: Remove an expense by its ID

    When the user asks to add an expense, use the add_expense tool. If no category is mentioned, default to "other".
    When the user asks to see, show, or list their expenses, use the list_expenses tool.
    When the user asks for a total or sum, use the get_total tool.
    When the user asks about a specific category, use the get_by_category tool.
    When the user asks to remove or delete an expense, use the remove_expense tool.

    Always format amounts as currency (e.g. $12.50). Be concise and helpful.
    If the user's request is ambiguous, ask for clarification.

    """,  # <- Instructions that guide the agent's behavior. This is where you can specify the agent's goals, constraints, and any other information that will help it perform its task.
    tools=[add_expense, list_expenses, get_total, get_by_category, remove_expense],  # <- Attach all five tools to the agent. The LLM will automatically pick the right one based on what the user asks.
    fail_safe=fail_safe,  # <- Attach the fail-safe configuration to the agent. The runner will enforce these limits during execution. If any limit is exceeded, it raises AgentLoopLimitError instead of letting the agent run forever.
)
runner = Runner()  # <- Create a Runner instance to execute the agent. The runner handles the request-response loop, tool dispatch, and state management.

if __name__ == "__main__":
    print("[expense-tracker] > Hi! I'm your expense tracker. I can add, list, total, filter, and remove expenses.")
    print("[expense-tracker] > Try saying things like 'Add lunch $12.50 food' or 'Show my total'")
    print("[expense-tracker] > Type 'quit' to exit.\n")

    while True:  # <- Conversation loop -- because this is a stateful app, we keep taking input so the user can manage their expenses across multiple turns.
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agent.

        if user_input.lower() in ("quit", "exit", "q"):
            print("[expense-tracker] > Goodbye! Keep tracking those expenses!")
            break

        try:
            response = runner.run_sync(
                tracker, user_message=user_input
            )  # <- Run the agent synchronously using the Runner. We pass the user's input as a message to the agent. The agent may call one or more tools before responding. The runner enforces the fail-safe limits during this call.

            print(
                f"[expense-tracker] > {response.final_text}\n"
            )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.

        except AgentLoopLimitError as e:
            print(
                f"[expense-tracker] > Safety limit reached: {e}\n"
            )  # <- If the agent exceeds any fail-safe limit (max_steps, max_llm_calls, or max_tool_calls), the runner raises AgentLoopLimitError. We catch it here and print a friendly message instead of crashing. This is the key pattern: wrap runner.run_sync() in a try/except to handle fail-safe violations gracefully.



"""
---
Tl;dr: This example creates an expense tracking agent with five tools (add, list, total, filter by category, remove) and a FailSafeConfig that limits the agent to 10 loop steps, 15 LLM calls, and 20 tool invocations per run. If any limit is exceeded, the runner raises AgentLoopLimitError which is caught in the conversation loop and reported as a friendly message. This pattern protects against infinite loops, runaway API costs, and pathological tool-call chains -- essential for any production agent deployment.
---
---
What's next?
- Try lowering the fail-safe limits (e.g. max_steps=3) and asking a complex question to see AgentLoopLimitError in action.
- Experiment with the max_total_cost_usd field in FailSafeConfig to set a dollar budget ceiling for your agent runs.
- Add more fail-safe fields like max_wall_time_s to set a wall-clock timeout, or fallback_model_chain to specify cheaper fallback models.
- Check out the other examples in the library to see how to use subagents, delegation, and policy engines for more advanced agent architectures!
---
"""
