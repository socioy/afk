"""
---
name: Customer Support Router — InstructionRoles
description: InstructionRole callbacks that dynamically augment agent instructions based on runtime context and state.
tags: [instruction-role, dynamic-instructions]
---
---
InstructionRole is a protocol that lets you append dynamic instructions to an agent at runtime.
Unlike a static instruction string or a callable InstructionProvider (which replaces the entire
instruction), InstructionRole callbacks ADD extra instruction text on top of the base instructions.
Multiple InstructionRoles can be stacked — each one is called in order and their outputs are
concatenated after the base instructions.

Each InstructionRole receives:
  - context: dict[str, JSONValue] — the runtime context from runner.run(context={...})
  - state: AgentState — the current runtime state (e.g., "running")

And returns:
  - str | list[str] | None — additional instruction text (or None to skip)

This is perfect for cross-cutting concerns like: "always mention business hours", "add VIP
handling for premium customers", "proactively warn about known outages".
---
"""

from tools import CUSTOMER_ACCOUNTS, SERVICE_STATUS  # <- Import simulated data to check customer tier and system health.


# ===========================================================================
# InstructionRole 1: Customer tier awareness
# ===========================================================================
# This role checks if the current customer is a premium user and adds
# VIP handling instructions. The context must contain "customer_username"
# for this to work. If not present, it returns None (no extra instructions).

def customer_tier_role(context: dict, state: str) -> str | None:  # <- InstructionRole callback: receives context dict and agent state, returns optional instruction text. The state parameter is the AgentState string (e.g., "running", "paused") — useful for phase-aware logic.
    username = context.get("customer_username", "")
    if not username:
        return None  # <- Return None to add no extra instructions. The base instructions remain unchanged.

    account = CUSTOMER_ACCOUNTS.get(username.lower())
    if account is None:
        return None

    if account["plan"] == "premium":
        return (  # <- This text is appended AFTER the agent's base instructions. The agent sees its original instructions plus this extra paragraph.
            "\n\n--- VIP Customer Notice ---\n"
            f"The current customer ({account['name']}) is on the PREMIUM plan.\n"
            "- Provide priority response with extra care and attention.\n"
            "- Offer to escalate to a senior specialist if needed.\n"
            "- Mention their loyalty and thank them for being a valued premium member.\n"
            "- Do NOT suggest plan upgrades — they are already on the best plan."
        )

    return (
        f"\n\nNote: Customer {account['name']} is on the {account['plan']} plan. "
        f"If appropriate, mention the benefits of upgrading to Premium."
    )


# ===========================================================================
# InstructionRole 2: System health awareness
# ===========================================================================
# This role checks current service status and adds proactive awareness
# instructions when systems are degraded or down. This way, the agent can
# mention known issues before the customer even asks about them.

def system_health_role(context: dict, state: str) -> str | list[str] | None:  # <- Can also return a list of strings. Each string becomes a separate instruction paragraph.
    degraded = []
    down = []
    for service, status in SERVICE_STATUS.items():
        if status == "degraded":
            degraded.append(service)
        elif status == "down":
            down.append(service)

    if not degraded and not down:
        return None  # <- All systems healthy, no extra instructions needed.

    instructions = []  # <- Build a list of instruction strings. Each is appended separately to the agent's instructions.

    if down:
        instructions.append(
            f"\n\n--- System Alert (DOWN) ---\n"
            f"The following services are currently DOWN: {', '.join(down)}.\n"
            f"Proactively inform the customer if their issue might be related. "
            f"Apologize for the inconvenience and assure them the team is working on it."
        )

    if degraded:
        instructions.append(
            f"\n\n--- System Alert (DEGRADED) ---\n"
            f"The following services are experiencing degraded performance: {', '.join(degraded)}.\n"
            f"If the customer reports slowness, acknowledge this known issue."
        )

    return instructions  # <- Return list[str] — each string is appended as a separate instruction block.


# ===========================================================================
# InstructionRole 3: Business hours awareness
# ===========================================================================
# This role adds time-of-day context so the agent can set appropriate
# expectations about response times and availability.

def business_hours_role(context: dict, state: str) -> str | None:
    import datetime
    hour = datetime.datetime.now().hour  # <- Check current time to determine business hours.

    if 9 <= hour < 17:  # <- 9 AM to 5 PM local time.
        return (
            "\n\nCurrent status: Business hours (9 AM - 5 PM). "
            "Full support team is available. Escalations will be handled promptly."
        )
    elif 17 <= hour < 21:
        return (
            "\n\nCurrent status: After-hours (evening shift). "
            "Reduced team available. For urgent issues, offer to create a priority ticket "
            "that the full team will review first thing tomorrow morning."
        )
    else:
        return (
            "\n\nCurrent status: Overnight. Automated support only. "
            "Let the customer know that a human agent will follow up during business hours. "
            "Create a ticket for any issues that need human attention."
        )
