"""
---
name: Code Reviewer
description: A coordinator agent that delegates code review tasks to specialist subagents for style, bugs, and security analysis.
tags: [agent, runner, subagents, delegation]
---
---
This example introduces **subagents** -- agents that can delegate work to child agents. A parent "coordinator" agent receives code for review and delegates to three specialist subagents: a style reviewer, a bug detector, and a security auditor. The coordinator then synthesizes their findings into a single prioritized report.
---
"""

from afk.core import Runner  # <- Runner executes agents and manages their state. Same as Example 01.
from afk.agents import Agent  # <- Agent is the main building block. Here we create *multiple* agents that work together.

# ---------------------------------------------------------------------------
# Step 1: Define specialist subagents
# ---------------------------------------------------------------------------
# Each subagent has a focused role and its own instructions.
# They don't know about each other -- the coordinator decides when to call them.

style_reviewer = Agent(
    name="style-reviewer",  # <- A descriptive name so the coordinator can identify this subagent.
    model="ollama_chat/gpt-oss:20b",  # <- Same model for all agents in this example; you can mix models.
    instructions="""You review code for style and readability. Check for:
    - Naming conventions (variables, functions, classes)
    - Code formatting and consistency
    - Comments and documentation
    - Code organization and structure
    Provide specific, actionable feedback.""",  # <- Narrow, focused instructions make subagents more effective than one giant prompt.
)

bug_detector = Agent(
    name="bug-detector",  # <- Another specialist -- this one hunts for bugs.
    model="ollama_chat/gpt-oss:20b",
    instructions="""You review code for potential bugs and logic errors. Look for:
    - Off-by-one errors
    - Null/None reference issues
    - Unhandled edge cases
    - Type mismatches
    - Resource leaks
    Explain each issue clearly with the line/section affected.""",  # <- Asking the agent to reference specific lines keeps feedback concrete.
)

security_auditor = Agent(
    name="security-auditor",  # <- The third specialist focuses on security.
    model="ollama_chat/gpt-oss:20b",
    instructions="""You audit code for security vulnerabilities. Check for:
    - Injection vulnerabilities (SQL, command, XSS)
    - Hardcoded secrets or credentials
    - Insecure data handling
    - Missing input validation
    - Authentication/authorization gaps
    Rate each finding by severity (low/medium/high/critical).""",  # <- Severity ratings help the coordinator prioritize the final report.
)

# ---------------------------------------------------------------------------
# Step 2: Create the coordinator agent with subagents
# ---------------------------------------------------------------------------
# The coordinator is the *parent* agent. It receives the user's code and
# decides which subagents to delegate to. The Runner handles the routing
# automatically -- the coordinator just needs to know the subagents exist.

coordinator = Agent(
    name="code-reviewer",  # <- The top-level agent the user interacts with.
    model="ollama_chat/gpt-oss:20b",
    instructions="""You are a senior code reviewer. When a user submits code for review,
    delegate to your specialist subagents (style-reviewer, bug-detector, security-auditor)
    to get comprehensive feedback. Synthesize their findings into a clear, prioritized report.

    Your report should:
    1. Start with a brief summary of the code's purpose
    2. List critical and high-severity issues first
    3. Group findings by category (security, bugs, style)
    4. End with positive observations and overall assessment""",  # <- The coordinator's instructions focus on *orchestration*, not analysis.
    subagents=[style_reviewer, bug_detector, security_auditor],  # <- subagents enables delegation! The coordinator can route work to any of these child agents.
)

runner = Runner()  # <- One Runner instance is enough -- it can execute any agent (parent or child).

# ---------------------------------------------------------------------------
# Step 3: Run the review
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Paste the code you want reviewed (press Enter twice to submit):")
    print()

    lines = []  # <- Collect multi-line input so users can paste entire code snippets.
    while True:
        line = input()
        if line == "":
            if lines and lines[-1] == "":
                break  # <- Two consecutive empty lines signal end of input.
            lines.append(line)
        else:
            lines.append(line)

    user_code = "\n".join(lines).strip()  # <- Join all lines into a single string for the agent.

    if not user_code:
        print("[code-reviewer] > No code provided. Please paste a code snippet to review.")
    else:
        review_request = f"Please review the following code:\n\n```\n{user_code}\n```"  # <- Wrap in a code fence so the agent clearly sees the code boundary.

        response = runner.run_sync(
            coordinator, user_message=review_request
        )  # <- The Runner executes the coordinator, which may internally delegate to subagents. All of this happens inside run_sync.

        print(
            f"\n[code-reviewer] >\n{response.final_text}"
        )  # <- The final_text contains the coordinator's synthesized report after all subagent delegations are complete.


"""
---
Tl;dr: This example creates a code review system using subagents. Three specialist agents (style-reviewer, bug-detector, security-auditor) each focus on a specific aspect of code quality. A coordinator agent (code-reviewer) delegates to these specialists via the subagents parameter and synthesizes their findings into a prioritized report. The user pastes code, and the coordinator orchestrates the full review automatically.
---
---
What's next?
- Try adding more specialist subagents, such as a performance-reviewer or a test-coverage-checker, to expand the review scope.
- Experiment with giving each subagent a different model to see how model choice affects the quality of specialist feedback.
- Look at how the coordinator's instructions affect the final report -- try changing the synthesis format or priority ordering.
- Explore giving subagents their own tools (e.g., a linter or static analysis tool) to ground their reviews in concrete findings.
- Check out the other examples in the library to learn about hooks, guardrails, and memory for building even more capable agent systems!
---
"""
