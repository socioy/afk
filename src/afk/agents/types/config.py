"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Skill, routing, and configuration types.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from afk.llms.types import JSONValue

if TYPE_CHECKING:
    from afk.mcp.store import MCPServerRef
    from afk.tools.core.base import Tool


@dataclass(frozen=True, slots=True)
class SkillRef:
    """
    Resolved skill metadata for a specific skill name.

    Attributes:
        name: Name of the skill, extracted from the 'SKILL.md' file.
        description: Description of the skill, extracted from `SKILL.md`.
        root_dir: Absolute resolved skills root directory.
        skill_md_path: Absolute path to the skill's `SKILL.md`.
        checksum: Optional SHA checksum for skill content integrity tracking.
    """

    name: str
    description: str
    root_dir: str
    skill_md_path: str
    checksum: str | None = None


@dataclass(frozen=True, slots=True)
class SkillResolutionResult:
    """
    Result of resolving requested skills from the skills directory.

    Attributes:
        resolved_skills: Successfully resolved skill references.
        missing_skills: Skill names that could not be resolved.
    """

    resolved_skills: list[SkillRef]
    missing_skills: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SkillToolPolicy:
    """
    Security and execution limits applied to skill command tools.

    Attributes:
        command_allowlist: Command prefixes allowed for `run_skill_command`.
        deny_shell_operators: When `True`, block shell chaining/operators.
        max_stdout_chars: Maximum stdout characters retained in results.
        max_stderr_chars: Maximum stderr characters retained in results.
        command_timeout_s: Maximum command execution time in seconds.
    """

    command_allowlist: list[str] = field(default_factory=list)
    deny_shell_operators: bool = True
    max_stdout_chars: int = 20_000
    max_stderr_chars: int = 20_000
    command_timeout_s: float = 30.0


@dataclass(frozen=True, slots=True)
class RouterInput:
    """
    Payload passed into subagent router callbacks.

    Attributes:
        run_id: Current run identifier.
        thread_id: Current thread identifier.
        step: Current step index.
        context: JSON-safe runtime context snapshot.
        messages: Current message transcript payload.
    """

    run_id: str
    thread_id: str
    step: int
    context: dict[str, JSONValue]
    messages: list[dict[str, JSONValue]]


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """
    Subagent routing decision returned by router callbacks.

    Attributes:
        targets: Subagent names selected for execution.
        parallel: Whether selected targets should execute in parallel.
        metadata: Additional router metadata for audit/debug.
    """

    targets: list[str] = field(default_factory=list)
    parallel: bool = False
    metadata: dict[str, JSONValue] = field(default_factory=dict)


InstructionProvider: TypeAlias = Callable[[dict[str, JSONValue]], str]
ToolLike: TypeAlias = "Tool[Any, Any] | Callable[[], Tool[Any, Any]]"
MCPServerLike: TypeAlias = "str | dict[str, Any] | MCPServerRef"
ContextInheritance: TypeAlias = list[str]
