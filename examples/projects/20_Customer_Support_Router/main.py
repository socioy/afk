"""
---
name: Customer Support Router
description: A customer support system that uses a SubagentRouter to dynamically route queries to specialist agents.
tags: [agent, runner, subagents, subagent-router, delegation]
---
---
This example demonstrates how to use a SubagentRouter callback to dynamically route user
messages to the most appropriate specialist subagent. Instead of letting the LLM decide
which subagent to delegate to (which can be slow and unreliable), you provide a Python
function that examines the message and returns the target subagent name(s). This gives
you deterministic, fast routing logic while still leveraging LLM intelligence within each
specialist for the actual response.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner executes agents and manages their state.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.


# ===========================================================================
# Simulated customer data
# ===========================================================================

CUSTOMER_ACCOUNTS: dict[str, dict] = {  # <- Simulated customer database. In production this would be a real database — but static data keeps the focus on routing.
    "alice": {"name": "Alice Johnson", "email": "alice@example.com", "plan": "premium", "balance": 49.99, "status": "active"},
    "bob": {"name": "Bob Smith", "email": "bob@example.com", "plan": "basic", "balance": 0.00, "status": "active"},
    "charlie": {"name": "Charlie Davis", "email": "charlie@example.com", "plan": "premium", "balance": 129.50, "status": "suspended"},
}

SERVICE_STATUS: dict[str, str] = {  # <- Simulated service health dashboard.
    "api": "operational",
    "web_app": "operational",
    "database": "degraded",
    "cdn": "operational",
    "email_service": "down",
}

KNOWN_ISSUES: list[dict] = [  # <- Known issues database for tech support.
    {"id": "BUG-101", "title": "Login fails on Safari 17", "status": "investigating", "workaround": "Use Chrome or Firefox"},
    {"id": "BUG-102", "title": "Slow dashboard loading", "status": "identified", "workaround": "Clear browser cache and hard refresh"},
    {"id": "BUG-103", "title": "Email notifications delayed", "status": "in_progress", "workaround": "Check spam folder; notifications may arrive late"},
]


# ===========================================================================
# Billing specialist tools
# ===========================================================================

class CustomerLookupArgs(BaseModel):
    username: str = Field(description="The customer's username to look up")


@tool(args_model=CustomerLookupArgs, name="check_balance", description="Check a customer's current account balance and plan details")
def check_balance(args: CustomerLookupArgs) -> str:  # <- Billing-specific tool for account queries.
    account = CUSTOMER_ACCOUNTS.get(args.username.lower())
    if account is None:
        return f"Customer '{args.username}' not found. Known customers: {', '.join(CUSTOMER_ACCOUNTS.keys())}"
    return (
        f"Account: {account['name']}\n"
        f"Plan: {account['plan']}\n"
        f"Balance due: ${account['balance']:.2f}\n"
        f"Status: {account['status']}"
    )


@tool(args_model=CustomerLookupArgs, name="check_plan", description="Check what plan a customer is on and recommend upgrades")
def check_plan(args: CustomerLookupArgs) -> str:
    account = CUSTOMER_ACCOUNTS.get(args.username.lower())
    if account is None:
        return f"Customer '{args.username}' not found."
    current = account["plan"]
    recommendation = "You're already on our best plan!" if current == "premium" else "Consider upgrading to Premium for priority support and advanced features."
    return f"Current plan: {current}\n{recommendation}"


# ===========================================================================
# Technical support tools
# ===========================================================================

class EmptyArgs(BaseModel):
    pass


@tool(args_model=EmptyArgs, name="check_service_status", description="Check the current status of all services")
def check_service_status(args: EmptyArgs) -> str:  # <- Tech-specific tool for system health.
    lines = []
    for service, status in SERVICE_STATUS.items():
        icon = "ok" if status == "operational" else ("warn" if status == "degraded" else "DOWN")
        lines.append(f"  [{icon}] {service}: {status}")
    return "Service Status:\n" + "\n".join(lines)


@tool(args_model=EmptyArgs, name="list_known_issues", description="List all known issues and their workarounds")
def list_known_issues(args: EmptyArgs) -> str:  # <- Tech-specific tool for known bugs.
    if not KNOWN_ISSUES:
        return "No known issues at this time."
    lines = []
    for issue in KNOWN_ISSUES:
        lines.append(
            f"  [{issue['id']}] {issue['title']}\n"
            f"    Status: {issue['status']} | Workaround: {issue['workaround']}"
        )
    return "Known Issues:\n" + "\n".join(lines)


# ===========================================================================
# Account management tools
# ===========================================================================

class UpdateEmailArgs(BaseModel):
    username: str = Field(description="The customer's username")
    new_email: str = Field(description="The new email address")


@tool(args_model=UpdateEmailArgs, name="update_email", description="Update a customer's email address on their account")
def update_email(args: UpdateEmailArgs) -> str:  # <- Account-specific tool for profile changes.
    account = CUSTOMER_ACCOUNTS.get(args.username.lower())
    if account is None:
        return f"Customer '{args.username}' not found."
    old_email = account["email"]
    account["email"] = args.new_email  # <- Mutate the simulated database.
    return f"Email updated for {account['name']}: {old_email} -> {args.new_email}"


@tool(args_model=CustomerLookupArgs, name="get_account_info", description="Get full account information for a customer")
def get_account_info(args: CustomerLookupArgs) -> str:
    account = CUSTOMER_ACCOUNTS.get(args.username.lower())
    if account is None:
        return f"Customer '{args.username}' not found."
    lines = [f"  {key}: {value}" for key, value in account.items()]
    return f"Account Information for {account['name']}:\n" + "\n".join(lines)


# ===========================================================================
# Specialist subagents
# ===========================================================================

billing_agent = Agent(  # <- Specialist agent for billing/payment questions. Has only billing-related tools.
    name="billing-support",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a billing support specialist. You handle questions about:
    - Account balances and payments
    - Plan details and upgrades
    - Billing disputes and refunds

    Be helpful and clear about pricing. If you can't resolve an issue, suggest the customer
    contact billing@company.com directly.
    """,
    tools=[check_balance, check_plan],
)

technical_agent = Agent(  # <- Specialist agent for technical issues. Has system status and known issues tools.
    name="technical-support",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a technical support specialist. You handle questions about:
    - Service outages and degraded performance
    - Known bugs and their workarounds
    - Technical troubleshooting steps

    Always check service status first when a user reports an issue. If there's a known issue,
    share the workaround. Be empathetic about technical difficulties.
    """,
    tools=[check_service_status, list_known_issues],
)

account_agent = Agent(  # <- Specialist agent for account management. Has profile update tools.
    name="account-support",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are an account management specialist. You handle questions about:
    - Account information and profile updates
    - Email address changes
    - Account status (active, suspended, etc.)

    Always verify the customer's identity (ask for username) before making changes.
    Confirm all changes with the customer.
    """,
    tools=[get_account_info, update_email],
)

general_agent = Agent(  # <- Fallback agent for queries that don't match any specialist. No tools — just friendly conversation.
    name="general-support",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a friendly general support agent. You handle questions that don't fit
    into billing, technical, or account categories. Provide helpful answers and,
    if the user needs specialist help, suggest they ask about billing, technical issues,
    or account management specifically so we can route them to the right team.
    """,
)


# ===========================================================================
# SubagentRouter (the key concept)
# ===========================================================================

def route_support(context: dict) -> list[str]:  # <- This is the SubagentRouter callback. It receives the runtime context (which includes the user's message) and returns a list of subagent names to route to. The runner will only consider these subagents for delegation, ignoring the others. This is MUCH faster than letting the LLM evaluate all subagents.
    message = context.get("user_message", "").lower()  # <- Extract the user's message from context for keyword matching.

    # --- Billing keywords ---
    billing_keywords = ["bill", "payment", "charge", "invoice", "balance", "plan", "upgrade", "subscription", "price", "refund"]
    if any(kw in message for kw in billing_keywords):  # <- Simple keyword matching for routing. In production, you might use a lightweight classifier or embedding similarity.
        return ["billing-support"]

    # --- Technical keywords ---
    tech_keywords = ["error", "bug", "crash", "slow", "down", "outage", "broken", "not working", "issue", "status", "fix"]
    if any(kw in message for kw in tech_keywords):
        return ["technical-support"]

    # --- Account keywords ---
    account_keywords = ["account", "email", "profile", "password", "username", "update", "change", "settings", "suspend"]
    if any(kw in message for kw in account_keywords):
        return ["account-support"]

    return ["general-support"]  # <- Default fallback: route to the general agent if no keywords match.


# ===========================================================================
# Coordinator agent with subagent router
# ===========================================================================

support_coordinator = Agent(
    name="support-coordinator",  # <- The coordinator agent manages the dispatch.
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a customer support coordinator. Your job is to help customers by routing
    their questions to the right specialist team.

    You have four specialist teams available:
    - Billing Support: handles payments, plans, and billing questions
    - Technical Support: handles service issues, bugs, and outages
    - Account Support: handles profile changes and account management
    - General Support: handles everything else

    Listen to the customer's issue and delegate to the appropriate specialist.
    Introduce which team is handling their request so the customer knows who they're
    talking to.
    """,
    subagents=[billing_agent, technical_agent, account_agent, general_agent],  # <- All specialist agents are registered as subagents. The coordinator can delegate to any of them.
    subagent_router=route_support,  # <- The router callback determines which subagents are eligible for each request. This overrides the LLM's free choice — the runner will only consider the subagents returned by this function.
)

runner = Runner()


if __name__ == "__main__":
    print("Customer Support System (type 'quit' to exit)")
    print("=" * 50)
    print("Ask about billing, technical issues, account management, or anything else!\n")

    while True:  # <- Conversation loop for the support interaction.
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Thank you for contacting support. Goodbye!")
            break

        response = runner.run_sync(
            support_coordinator,
            user_message=user_input,
            context={"user_message": user_input},  # <- Pass the user message as part of the context so the subagent_router function can access it. The router reads context["user_message"] to decide where to route.
        )

        print(f"[support] > {response.final_text}\n")



"""
---
Tl;dr: This example creates a customer support system with a coordinator agent and four specialist subagents
(billing, technical, account, general). A SubagentRouter callback function examines the user's message
keywords and returns the name(s) of the appropriate specialist(s) to route to. This gives you deterministic,
fast routing instead of relying on the LLM to pick the right subagent — the LLM's intelligence is used
within each specialist for crafting the actual response, not for choosing who handles the query.
---
---
What's next?
- Try adding routing confidence — if multiple keyword sets match, route to multiple specialists and let the coordinator combine their insights.
- Experiment with an ML-based router (e.g. embedding similarity) instead of keyword matching for more robust routing.
- Add a "routing_log" that records which specialist handled each query for analytics.
- Combine the router with a PolicyEngine to add approval requirements for sensitive operations (like account changes).
- Explore returning multiple subagent names from the router to enable parallel specialist consultation.
- Check out the DelegationPlan examples for DAG-based multi-agent orchestration with dependencies!
---
"""
