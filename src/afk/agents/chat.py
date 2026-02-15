"""
Chat-oriented agent convenience wrapper.
"""

from __future__ import annotations

from .base import Agent
from .types import AgentResult, JSONValue


class ChatAgent(Agent):
    """
    Convenience agent requiring a user message for each call.
    """

    async def call(
        self,
        user_message: str,
        *,
        context: dict[str, JSONValue] | None = None,
        thread_id: str | None = None,
    ) -> AgentResult:
        """
        Run chat flow with a required user message.

        Args:
            user_message: End-user input for this turn.
            context: Optional JSON-safe context payload.
            thread_id: Optional thread identifier for memory continuity.

        Returns:
            Terminal `AgentResult` for the run.
        """
        return await super().call(
            user_message=user_message,
            context=context,
            thread_id=thread_id,
        )
