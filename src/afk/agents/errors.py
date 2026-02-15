"""
Agent-layer error taxonomy.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for all agent-runtime failures."""
    pass


class AgentConfigurationError(AgentError):
    """
    Raised when agent configuration is invalid.

    Typical cases:
    - invalid constructor values
    - unsupported mode combinations
    - missing required runtime dependencies
    """
    pass


class AgentExecutionError(AgentError):
    """Raised for runtime execution failures not tied to configuration."""
    pass


class AgentRetryableError(AgentExecutionError):
    """Raised for runtime failures that may be retried safely."""
    pass


class AgentLoopLimitError(AgentExecutionError):
    """Raised when loop/step guard limits are exceeded."""
    pass


class AgentBudgetExceededError(AgentExecutionError):
    """Raised when wall-time/cost/token/tool budgets are exceeded."""
    pass


class AgentCancelledError(AgentExecutionError):
    """Raised when a run is cancelled by caller or control plane."""
    pass


class AgentInterruptedError(AgentExecutionError):
    """Raised when a run is interrupted mid-execution."""
    pass


class AgentPausedError(AgentExecutionError):
    """Raised when an operation is attempted while run is paused."""
    pass


class SubagentRoutingError(AgentExecutionError):
    """Raised when subagent routing is invalid, unsafe, or inconsistent."""
    pass


class SubagentExecutionError(AgentExecutionError):
    """Raised when delegated subagent execution fails."""
    pass


class SkillResolutionError(AgentConfigurationError):
    """Raised when resolving requested skills from `skills_dir` fails."""

    pass


class SkillAccessError(AgentExecutionError):
    """Raised when skill file access violates path safety constraints."""
    pass


class SkillCommandDeniedError(AgentExecutionError):
    """Raised when skill command execution is denied by policy or allowlist."""
    pass


class AgentCheckpointCorruptionError(AgentExecutionError):
    """Raised when checkpoint payload cannot be validated or loaded."""
    pass


class AgentCircuitOpenError(AgentExecutionError):
    """Raised when an open circuit breaker blocks dependency execution."""
    pass
