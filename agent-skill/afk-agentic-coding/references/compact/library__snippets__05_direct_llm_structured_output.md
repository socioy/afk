# 05_direct_llm_structured_output

Compact reference for 05_direct_llm_structured_output.

Source: `docs/library/snippets/05_direct_llm_structured_output.mdx`

````python 05_direct_llm_structured_output.py
"""
Example 05: Direct LLM usage with structured output.

Run:
    python 05_direct_llm_structured_output.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from afk.llms import LLMRequest, Message, create_llm

class Plan(BaseModel):
    title: str = Field(min_length=1)
    steps: list[str] = Field(min_length=2, max_length=8)

async def main() -> None:
    adapter = os.getenv("AFK_LLM_ADAPTER", "openai")
    model_name = os.getenv("AFK_LLM_MODEL", "gpt-4.1-mini")
    llm = create_llm(adapter)

    req = LLMRequest(
        model=model_name or llm.config.default_model,
        idempotency_key="demo-plan-001",
        messages=[
            Message(
                role="user",
                content="Create a small onboarding plan for a new backend engineer.",
            )
        ],
    )

    response = await llm.chat(req, response_model=Plan)

    print("adapter:", adapter)
````

> Code block truncated to 40 lines. Source: `docs/library/snippets/05_direct_llm_structured_output.mdx`
