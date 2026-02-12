from __future__ import annotations

from attrs import frozen
"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module defines the common types used in LLM interactions.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Literal, Sequence

Role = Literal["user", "assistant", "system", "tool"]

@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: Union[str, Dict[str, Any]]
    name: Optional[str] = None

@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """
    Data-only representation of a model-returned tool call.
    The agent module decides if/when/how to execute this.
    """
    id: Optional[str] = None
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    structued_response: Optional[Dict[str, Any]] = None # for json outputs with fixed schema defined in the prompt.
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Usage
    raw: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None

@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    embeddings: List[List[float]]
    raw: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None

@dataclass(frozen=True, slots=True)
class LLMRequest:
    """
    Canonical request type used by middleware, the client and the agents.
    """
    model: str
    messages: List[Message] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None # e.g. "auto", "none", or a specific tool dict
    max_tokens: Optional[int] = None
    timeout_s: Optional[float] = None 
    metadata: Dict[str, Any] = field(default_factory=dict) 
    extra: Dict[str, Any] = field(default_factory=dict) # provider-specific extra params