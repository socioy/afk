
# Content Moderator

A content moderation agent combining static PolicyEngine rules with dynamic PolicyRole callbacks for two-layer policy enforcement.

## Project Structure

```
25_Content_Moderator/
  main.py       # Entry point — agent with both PolicyEngine and PolicyRoles
  tools.py      # Tool definitions and simulated data
  policy.py     # Static PolicyEngine rules + 3 dynamic PolicyRole callbacks
```

## Key Concepts

- **PolicyEngine + PolicyRule**: Static, declarative rules evaluated by priority. Use `context_equals` to match.
- **PolicyRole**: `(event: PolicyEvent) -> PolicyDecision` callbacks for dynamic runtime decisions
- **PolicyDecision actions**: `allow`, `deny`, `defer`, `rewrite` (modifies tool_args before execution)
- **Stacking**: PolicyEngine runs first, then each PolicyRole. ANY deny blocks the action.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/25_Content_Moderator

Expected interaction
User: Moderate this post by alice: "Just had the most amazing sunset hike today!"
Agent: [Analyzes] Clean. [Publishes] Post published successfully!
User: Moderate: "This GUARANTEED miracle cure will fix everything!"
Agent: [Analyzes] Flaggable keywords. [Flags] Content flagged for review.
