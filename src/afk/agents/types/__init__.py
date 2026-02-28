"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Provider-neutral types for AFK agent runtime contracts.

This package re-exports all types for backward compatibility. Types are
organized in sub-modules:

- ``common``: Literal type aliases (AgentState, FailurePolicy, etc.)
- ``result``: AgentResult, UsageAggregate, execution records
- ``policy``: AgentRunEvent, PolicyEvent, PolicyDecision, FailSafeConfig
- ``interaction``: Approval/input requests/decisions, AgentRunHandle
- ``config``: SkillRef, SkillToolPolicy, RouterInput/Decision, type aliases
- ``protocols``: InstructionRole, PolicyRole, SubagentRouter
"""

from __future__ import annotations

# Re-export the JSONValue alias so existing `from .types import JSONValue` works
from ...llms.types import JSONValue

# --- Common literal types ---
from .common import (
    AgentEventType,
    AgentState,
    DecisionKind,
    FailurePolicy,
    InteractionMode,
    PolicyAction,
    SubagentParallelismMode,
)

# --- Config & routing types ---
from .config import (
    ContextInheritance,
    InstructionProvider,
    MCPServerLike,
    RouterDecision,
    RouterInput,
    SkillRef,
    SkillResolutionResult,
    SkillToolPolicy,
    ToolLike,
)

# --- Interaction types ---
from .interaction import (
    AgentRunHandle,
    ApprovalDecision,
    ApprovalRequest,
    DeferredDecision,
    UserInputDecision,
    UserInputRequest,
)

# --- Policy & event types ---
from .policy import (
    AgentRunEvent,
    FailSafeConfig,
    PolicyDecision,
    PolicyEvent,
)

# --- Protocol interfaces ---
from .protocols import (
    InstructionRole,
    PolicyRole,
    SubagentRouter,
)

# --- Result & execution record types ---
from .result import (
    AgentResult,
    CommandExecutionRecord,
    SkillReadRecord,
    SubagentExecutionRecord,
    ToolExecutionRecord,
    UsageAggregate,
    json_value_from_tool_result,
    tool_record_from_result,
)

__all__ = [
    # Common
    "AgentState",
    "SubagentParallelismMode",
    "FailurePolicy",
    "InteractionMode",
    "DecisionKind",
    "PolicyAction",
    "AgentEventType",
    # Result
    "UsageAggregate",
    "ToolExecutionRecord",
    "SubagentExecutionRecord",
    "SkillReadRecord",
    "CommandExecutionRecord",
    "AgentResult",
    "json_value_from_tool_result",
    "tool_record_from_result",
    # Policy
    "AgentRunEvent",
    "PolicyEvent",
    "PolicyDecision",
    "FailSafeConfig",
    # Interaction
    "ApprovalRequest",
    "UserInputRequest",
    "DeferredDecision",
    "ApprovalDecision",
    "UserInputDecision",
    "AgentRunHandle",
    # Config
    "SkillRef",
    "SkillResolutionResult",
    "SkillToolPolicy",
    "RouterInput",
    "RouterDecision",
    "InstructionProvider",
    "MCPServerLike",
    "ToolLike",
    "ContextInheritance",
    # Protocols
    "InstructionRole",
    "PolicyRole",
    "SubagentRouter",
    # Re-exports
    "JSONValue",
]
