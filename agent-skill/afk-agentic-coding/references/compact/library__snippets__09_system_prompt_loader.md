# 09_system_prompt_loader

Compact reference for 09_system_prompt_loader.

Source: `docs/library/snippets/09_system_prompt_loader.mdx`

````python 09_system_prompt_loader.py
"""
Example 09: System prompt loader, template context, and prompt reuse.

Run:
    python 09_system_prompt_loader.py
"""

from __future__ import annotations

from pathlib import Path

from afk.agents import Agent

def bootstrap_prompts(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "CHAT_AGENT.md").write_text(
        (
            "You are {{ agent_name }}. "
            "Always answer in {{ ctx.locale }} for user {{ user_id }}."
        ),
        encoding="utf-8",
    )
    (base_dir / "BILLING_REVIEWER.md").write_text(
        (
            "You are a billing reviewer for account {{ context.account_id }}. "
            "Respond with concise operational steps."
        ),
        encoding="utf-8",
    )
    (base_dir / "shared_incident_prompt.md").write_text(
        (
            "You are an incident assistant. "
            "Use tenant={{ context.tenant }} and locale={{ locale }}."
        ),
        encoding="utf-8",
    )

````

> Code block truncated to 40 lines. Source: `docs/library/snippets/09_system_prompt_loader.mdx`
