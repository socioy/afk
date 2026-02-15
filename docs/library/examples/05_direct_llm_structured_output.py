"""
Example 05: Direct LLM usage with structured output.

Run:
    uv run python docs/library/examples/05_direct_llm_structured_output.py
"""

from __future__ import annotations

import asyncio
import os

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
    print("request_id:", response.request_id)
    print("model:", response.model)
    print("structured_response:", response.structured_response)
    print("raw_text:", response.text)


if __name__ == "__main__":
    asyncio.run(main())
