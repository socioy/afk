"""Input models for tool hook and middleware flows."""

from typing import Any

from pydantic import BaseModel, Field


class DraftReplyArgs(BaseModel):
    channel: str = Field(min_length=1)
    message: str = Field(min_length=1)


class PostPayload(BaseModel):
    output: Any
    tool_name: str | None = None
