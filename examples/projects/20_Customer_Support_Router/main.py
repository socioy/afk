"""
---
name: Customer Support Router
description: A customer support system combining SubagentRouter for deterministic routing with InstructionRole callbacks for dynamic instruction augmentation.
tags: [agent, runner, subagents, subagent-router, instruction-role, delegation, context]
---
---
This example demonstrates two powerful AFK features working together:

1. **SubagentRouter**: A Python callback that examines the runtime context and returns subagent
   name(s) for deterministic routing. Instead of the LLM choosing which subagent to delegate to
   (slow, unreliable), your code makes the routing decision instantly.

2. **InstructionRole**: Protocol callbacks that dynamically AUGMENT the agent's base instructions
   at runtime. Unlike `instructions=` (which sets the whole instruction), InstructionRoles APPEND
   extra instruction text on top. Multiple roles can stack:
   - customer_tier_role: adds VIP handling for premium customers
   - system_health_role: adds proactive alerts when services are degraded/down
   - business_hours_role: adds time-of-day context for response expectations

Each InstructionRole receives `(context: dict, state: AgentState)` and returns `str | list[str] | None`.
Returning None means "no extra instructions for this context". This is ideal for cross-cutting
concerns that shouldn't clutter the base instruction string.
---
"""

from afk.core import Runner  # <- Runner executes agents and manages their state.

from agents import support_coordinator  # <- Import the coordinator agent from agents.py. It has subagents, router, and InstructionRoles already configured.


runner = Runner()


# ===========================================================================
# Interactive session
# ===========================================================================

if __name__ == "__main__":
    print("Customer Support System (type 'quit' to exit)")
    print("=" * 55)
    print()
    print("Features active:")
    print("  - SubagentRouter: deterministic keyword-based routing")
    print("  - InstructionRole: customer tier awareness (VIP for premium)")
    print("  - InstructionRole: system health proactive alerts")
    print("  - InstructionRole: business hours context")
    print()

    # --- Optional: identify the customer for VIP handling ---
    customer_username = input("Customer username (alice/bob/charlie, or Enter to skip): ").strip()
    if customer_username:
        print(f"  Identified customer: {customer_username}")
    print()
    print("Ask about billing, technical issues, account management, or anything else!\n")

    while True:
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Thank you for contacting support. Goodbye!")
            break

        response = runner.run_sync(
            support_coordinator,
            user_message=user_input,
            context={  # <- Runtime context flows to: (1) SubagentRouter for routing, (2) each InstructionRole for dynamic instruction augmentation, and (3) any InstructionProvider or Jinja2 template.
                "user_message": user_input,  # <- Used by the SubagentRouter to match keywords.
                "customer_username": customer_username,  # <- Used by the customer_tier_role InstructionRole to add VIP handling.
            },
        )

        print(f"[support] > {response.final_text}\n")



"""
---
Tl;dr: This example combines SubagentRouter (deterministic keyword-based routing to specialist
subagents) with InstructionRole callbacks (dynamic instruction augmentation at runtime). Three
InstructionRoles are stacked on the coordinator agent: customer_tier_role checks the customer's
plan and adds VIP handling instructions for premium users; system_health_role checks the service
status dashboard and proactively alerts about degraded/down services; business_hours_role adds
time-of-day context for appropriate response expectations. Each role receives (context, state)
and returns str | list[str] | None. Returning None means "skip" — the base instructions remain
as-is. The SubagentRouter examines context["user_message"] keywords to route to billing, tech,
account, or general support without LLM involvement.
---
---
What's next?
- Examine roles.py to see the three InstructionRole implementations with detailed inline documentation.
- Try adding a fourth InstructionRole (e.g., "language_detection_role" that switches tone by language).
- Use async InstructionRoles (async def) for roles that need database or API lookups.
- Combine InstructionRole with instruction_file templates — both work on the same agent simultaneously.
- Check out the Content Moderator example to see PolicyRole callbacks for runtime policy decisions!
---
"""
