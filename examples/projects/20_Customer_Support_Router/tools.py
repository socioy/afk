"""
---
name: Customer Support Router — Tools
description: Tool definitions and simulated data for the customer support system.
tags: [tools]
---
---
All tool definitions and simulated data for the customer support system live here.
Separating tools from agent definitions keeps each module focused and testable.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.


# ===========================================================================
# Simulated customer data
# ===========================================================================

CUSTOMER_ACCOUNTS: dict[str, dict] = {
    "alice": {"name": "Alice Johnson", "email": "alice@example.com", "plan": "premium", "balance": 49.99, "status": "active"},
    "bob": {"name": "Bob Smith", "email": "bob@example.com", "plan": "basic", "balance": 0.00, "status": "active"},
    "charlie": {"name": "Charlie Davis", "email": "charlie@example.com", "plan": "premium", "balance": 129.50, "status": "suspended"},
}

SERVICE_STATUS: dict[str, str] = {
    "api": "operational",
    "web_app": "operational",
    "database": "degraded",
    "cdn": "operational",
    "email_service": "down",
}

KNOWN_ISSUES: list[dict] = [
    {"id": "BUG-101", "title": "Login fails on Safari 17", "status": "investigating", "workaround": "Use Chrome or Firefox"},
    {"id": "BUG-102", "title": "Slow dashboard loading", "status": "identified", "workaround": "Clear browser cache and hard refresh"},
    {"id": "BUG-103", "title": "Email notifications delayed", "status": "in_progress", "workaround": "Check spam folder; notifications may arrive late"},
]


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class CustomerLookupArgs(BaseModel):
    username: str = Field(description="The customer's username to look up")


class UpdateEmailArgs(BaseModel):
    username: str = Field(description="The customer's username")
    new_email: str = Field(description="The new email address")


class EmptyArgs(BaseModel):
    pass


# ===========================================================================
# Billing tools
# ===========================================================================

@tool(args_model=CustomerLookupArgs, name="check_balance", description="Check a customer's current account balance and plan details")
def check_balance(args: CustomerLookupArgs) -> str:
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

@tool(args_model=EmptyArgs, name="check_service_status", description="Check the current status of all services")
def check_service_status(args: EmptyArgs) -> str:
    lines = []
    for service, status in SERVICE_STATUS.items():
        icon = "ok" if status == "operational" else ("warn" if status == "degraded" else "DOWN")
        lines.append(f"  [{icon}] {service}: {status}")
    return "Service Status:\n" + "\n".join(lines)


@tool(args_model=EmptyArgs, name="list_known_issues", description="List all known issues and their workarounds")
def list_known_issues(args: EmptyArgs) -> str:
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

@tool(args_model=UpdateEmailArgs, name="update_email", description="Update a customer's email address on their account")
def update_email(args: UpdateEmailArgs) -> str:
    account = CUSTOMER_ACCOUNTS.get(args.username.lower())
    if account is None:
        return f"Customer '{args.username}' not found."
    old_email = account["email"]
    account["email"] = args.new_email
    return f"Email updated for {account['name']}: {old_email} -> {args.new_email}"


@tool(args_model=CustomerLookupArgs, name="get_account_info", description="Get full account information for a customer")
def get_account_info(args: CustomerLookupArgs) -> str:
    account = CUSTOMER_ACCOUNTS.get(args.username.lower())
    if account is None:
        return f"Customer '{args.username}' not found."
    lines = [f"  {key}: {value}" for key, value in account.items()]
    return f"Account Information for {account['name']}:\n" + "\n".join(lines)
