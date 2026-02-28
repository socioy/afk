"""
---
name: Content Moderator
description: A content moderation agent combining static PolicyEngine rules with dynamic PolicyRole callbacks for two-layer policy enforcement.
tags: [agent, runner, tools, policy-engine, policy-rule, policy-role, moderation]
---
---
This example demonstrates two layers of policy enforcement in AFK:

1. **PolicyEngine + PolicyRule** (static, declarative): Rules defined at construction time with
   conditions (tool_name, context_equals). Evaluated deterministically by priority. Best for
   well-known, fixed guardrails like "deny publish for rejected content."

2. **PolicyRole** (dynamic, callback): Protocol callbacks invoked at runtime for every policy
   event. They receive the full PolicyEvent (including tool_args) and return a PolicyDecision.
   Best for logic that depends on runtime state: content scanning, rate-limiting, argument
   rewriting. Three PolicyRoles are demonstrated:
   - content_scan_role: scans actual tool_args for sensitive keywords (catches cases static rules miss)
   - author_rate_limit_role: limits publishing per author based on session state
   - category_rewrite_role: auto-corrects invalid categories using the "rewrite" action

Both layers work together: the Runner evaluates PolicyEngine rules first, then calls each
PolicyRole. If ANY layer denies the action, the tool call is blocked.
---
"""

from afk.core import Runner  # <- Runner orchestrates agent execution and applies both PolicyEngine and PolicyRole checks.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.

from tools import (  # <- Import tools from the tools module.
    analyze_content, publish_post, flag_content, reject_content,
    PUBLISHED_CONTENT, FLAGGED_CONTENT, REJECTED_CONTENT,
)
from policy import (  # <- Import policy components from the policy module.
    policy_engine,  # <- Static PolicyEngine with declarative rules.
    content_scan_role,  # <- Dynamic PolicyRole: scans tool_args for sensitive keywords.
    author_rate_limit_role,  # <- Dynamic PolicyRole: rate-limits per author.
    category_rewrite_role,  # <- Dynamic PolicyRole: auto-corrects invalid categories.
)


# ===========================================================================
# Agent setup — PolicyEngine + PolicyRoles
# ===========================================================================

moderator = Agent(
    name="content-moderator",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a content moderation agent. Your job is to review user-submitted posts and decide
    whether they should be published, flagged for review, or rejected.

    Workflow for each post:
    1. ALWAYS analyze the content first using the analyze_content tool.
    2. Based on the analysis:
       - If the content is CLEAN: publish it using publish_post.
       - If the content is BORDERLINE: flag it for human review using flag_content.
       - If the content VIOLATES policies: reject it using reject_content.
    3. Explain your decision clearly to the user.

    Important: The system has both static PolicyEngine rules and dynamic PolicyRole checks
    that may DENY your publish_post calls. If a publish attempt is denied, explain why and
    suggest the user modify their content.

    Be fair, consistent, and transparent about moderation decisions.
    """,
    tools=[analyze_content, publish_post, flag_content, reject_content],
    policy_engine=policy_engine,  # <- Static policy layer: evaluates rules by priority using context_equals conditions. Checked first.
    policy_roles=[  # <- Dynamic policy layer: callbacks invoked for every policy event. Each receives the full PolicyEvent and returns a PolicyDecision. Checked after PolicyEngine rules. If ANY role denies, the action is blocked.
        content_scan_role,  # <- Scans actual tool_args content for sensitive keywords. Catches cases where the LLM publishes without analyzing first.
        author_rate_limit_role,  # <- Rate-limits per author (max 5 posts per session). Depends on runtime state.
        category_rewrite_role,  # <- Auto-corrects invalid categories via the "rewrite" action. Changes tool_args before execution.
    ],
)

runner = Runner()


# ===========================================================================
# Sample posts for demonstration
# ===========================================================================

SAMPLE_POSTS: list[dict] = [
    {"author": "alice", "content": "Just had the most amazing sunset hike today! The trail was breathtaking.", "expected": "PUBLISH (clean)"},
    {"author": "bob", "content": "This GUARANTEED miracle cure will fix everything! Visit http://totallynotascam.com NOW!", "expected": "FLAG (spam keywords)"},
    {"author": "charlie", "content": "I think this is a controversial political topic we should discuss.", "expected": "FLAG (political/controversial)"},
    {"author": "diana", "content": "Stop the violence and hate! We need to end harassment in online spaces.", "expected": "REJECT (sensitive keywords in tool_args, caught by PolicyRole)"},
]


# ===========================================================================
# Interactive session
# ===========================================================================

if __name__ == "__main__":
    print("Content Moderator Agent (type 'quit' to exit)")
    print("=" * 55)
    print()
    print("Policy layers active:")
    print("  - PolicyEngine: static rules (deny flagged/rejected, defer opinion)")
    print("  - PolicyRole: content_scan_role (scans tool_args for keywords)")
    print("  - PolicyRole: author_rate_limit_role (max 5 posts per author)")
    print("  - PolicyRole: category_rewrite_role (auto-corrects invalid categories)")
    print()
    print("Sample posts to try:")
    for i, post in enumerate(SAMPLE_POSTS, 1):
        print(f"  {i}. [{post['author']}] {post['content'][:60]}...")
        print(f"     Expected: {post['expected']}")
    print(f"\nOr type your own post to moderate.\n")

    while True:
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print(f"\nModeration summary:")
            print(f"  Published: {len(PUBLISHED_CONTENT)}")
            print(f"  Flagged:   {len(FLAGGED_CONTENT)}")
            print(f"  Rejected:  {len(REJECTED_CONTENT)}")
            print("Goodbye!")
            break

        response = runner.run_sync(moderator, user_message=user_input)
        print(f"[moderator] > {response.final_text}\n")



"""
---
Tl;dr: This example demonstrates two-layer policy enforcement in AFK. The static layer uses
PolicyEngine with PolicyRules evaluated by priority (deny-publish-rejected at 300, deny-publish-flagged
at 200, defer-opinion at 150, default-allow at 50). The dynamic layer uses three PolicyRole callbacks:
content_scan_role scans actual tool_args for sensitive keywords (catches what static rules miss),
author_rate_limit_role prevents any author from publishing more than 5 posts (runtime state logic
that cannot be a static rule), and category_rewrite_role uses the "rewrite" action to auto-correct
invalid categories by modifying tool_args before execution. Both layers are evaluated by the Runner —
PolicyEngine first, then each PolicyRole. If ANY layer denies, the tool call is blocked.
---
---
What's next?
- Examine policy.py for the three PolicyRole implementations (content_scan, rate_limit, category_rewrite).
- Try the "rewrite" action: submit a post with an invalid category and watch it get auto-corrected.
- Add an async PolicyRole (async def) that checks an external API before allowing publish.
- Combine PolicyRole with InstructionRole for full dynamic behavior (see Customer Support Router example).
- Use policy_roles on subagents too — each agent can have its own policy layer.
- Check out the Document Approval example for AgentRunHandle lifecycle controls!
---
"""
