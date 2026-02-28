"""
---
name: Customer Support Router — Agents
description: Specialist subagent definitions and the SubagentRouter callback.
tags: [agents, subagents, subagent-router]
---
---
Agent definitions are separated from the entry point (main.py) for clarity. Each specialist
agent has its own instruction set and tools. The coordinator agent ties them together with
a SubagentRouter for deterministic routing and InstructionRole callbacks for dynamic behavior.
---
"""

from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.

from tools import (  # <- Import tools from the tools module.
    check_balance, check_plan,
    check_service_status, list_known_issues,
    get_account_info, update_email,
)
from roles import (  # <- Import InstructionRole callbacks from the roles module.
    customer_tier_role,
    system_health_role,
    business_hours_role,
)


# ===========================================================================
# Specialist subagents
# ===========================================================================

billing_agent = Agent(
    name="billing-support",  # <- Specialist for billing/payment questions.
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

technical_agent = Agent(
    name="technical-support",  # <- Specialist for technical issues.
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

account_agent = Agent(
    name="account-support",  # <- Specialist for account management.
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

general_agent = Agent(
    name="general-support",  # <- Fallback agent for unroutable queries.
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a friendly general support agent. You handle questions that don't fit
    into billing, technical, or account categories. Provide helpful answers and,
    if the user needs specialist help, suggest they ask about billing, technical issues,
    or account management specifically so we can route them to the right team.
    """,
)


# ===========================================================================
# SubagentRouter — deterministic routing logic
# ===========================================================================

def route_support(context: dict) -> list[str]:  # <- SubagentRouter callback: receives context dict, returns list of subagent names to route to. The runner will only consider these subagents for delegation.
    message = context.get("user_message", "").lower()

    billing_keywords = ["bill", "payment", "charge", "invoice", "balance", "plan", "upgrade", "subscription", "price", "refund"]
    if any(kw in message for kw in billing_keywords):
        return ["billing-support"]

    tech_keywords = ["error", "bug", "crash", "slow", "down", "outage", "broken", "not working", "issue", "status", "fix"]
    if any(kw in message for kw in tech_keywords):
        return ["technical-support"]

    account_keywords = ["account", "email", "profile", "password", "username", "update", "change", "settings", "suspend"]
    if any(kw in message for kw in account_keywords):
        return ["account-support"]

    return ["general-support"]


# ===========================================================================
# Coordinator agent with InstructionRole callbacks
# ===========================================================================

support_coordinator = Agent(
    name="support-coordinator",  # <- The top-level coordinator that dispatches to specialists.
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
    subagents=[billing_agent, technical_agent, account_agent, general_agent],
    subagent_router=route_support,  # <- SubagentRouter for deterministic routing based on keywords.
    instruction_roles=[  # <- InstructionRole callbacks. These are called at runtime and their return values are APPENDED after the base instructions. Multiple roles stack — each adds its own dynamic context. This is the key new feature demonstrated here.
        customer_tier_role,  # <- Adds VIP handling instructions for premium customers.
        system_health_role,  # <- Adds proactive alerts when services are degraded or down.
        business_hours_role,  # <- Adds time-of-day context (business hours, after-hours, overnight).
    ],
    context_defaults={  # <- Default context values. These can be overridden at runtime via runner.run(context={...}).
        "customer_username": "",
        "user_message": "",
    },
)
