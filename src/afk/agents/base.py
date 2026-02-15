"""
Base agent abstractions with DX-first constructor.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..llms import LLM
from ..tools import Tool, ToolContext, ToolRegistry
from .errors import AgentConfigurationError
from .types import (
    AgentResult,
    FailSafeConfig,
    InstructionProvider,
    InstructionRole,
    JSONValue,
    PolicyRole,
    SkillToolPolicy,
    SubagentParallelismMode,
    SubagentRouter,
    ToolLike,
)

if TYPE_CHECKING:
    from .policy import PolicyEngine
    from ..core.runner import Runner
    from .resolution import ModelResolver


class BaseAgent:
    """
    Canonical agent configuration object consumed by the runner.

    The class is intentionally declarative: it stores model, tools, policies,
    skills, and orchestration settings. Execution happens in `Runner`.
    """

    def __init__(
        self,
        *,
        model: str | LLM,
        name: str | None = None,
        tools: list[ToolLike] | None = None,
        subagents: list["BaseAgent"] | None = None,
        instructions: str | InstructionProvider | None = None,
        context_defaults: dict[str, JSONValue] | None = None,
        inherit_context_keys: list[str] | None = None,
        model_resolver: "ModelResolver | None" = None,
        skills: list[str] | None = None,
        skills_dir: str | Path = ".agents/skills",
        instruction_roles: list[InstructionRole] | None = None,
        policy_roles: list[PolicyRole] | None = None,
        policy_engine: "PolicyEngine | None" = None,
        subagent_router: SubagentRouter | None = None,
        max_steps: int = 20,
        tool_parallelism: int | None = None,
        subagent_parallelism_mode: SubagentParallelismMode = "configurable",
        fail_safe: FailSafeConfig | None = None,
        skill_tool_policy: SkillToolPolicy | None = None,
        enable_skill_tools: bool = True,
        runner: "Runner | None" = None,
    ) -> None:
        """
        Initialize an agent definition.

        Args:
            model: Either an instantiated `LLM` adapter or a model string that
                will be resolved by the runtime (for example `gpt-4o`,
                `ollama_chat/gpt-oss:20b`, `claude-sonnet-4`).
            name: Optional logical name used in traces and subagent routing.
            tools: List of tools (or callables returning tools) that the agent
                can invoke through LLM tool-calls.
            subagents: Child agents that can be routed/executed by this agent.
            instructions: Static instruction string or callable instruction
                provider resolved per run.
            context_defaults: Default JSON-safe context merged into each run
                before caller-provided context.
            inherit_context_keys: Context keys this agent accepts from a parent
                when used as subagent.
            model_resolver: Optional override resolver for model strings.
            skills: Skill names to resolve under `skills_dir`.
            skills_dir: Root directory for skills (`<skill>/SKILL.md`).
            instruction_roles: Callbacks that append dynamic instruction text.
            policy_roles: Callbacks that can allow/deny/defer runtime actions.
            policy_engine: Deterministic rule engine applied before policy roles.
            subagent_router: Router callback deciding subagent targets.
            max_steps: Maximum reasoning/tool loop steps for this agent.
            tool_parallelism: Max concurrent tool calls. When `None`, runtime
                uses `fail_safe.max_parallel_tools`.
            subagent_parallelism_mode: Subagent execution mode. `configurable`
                follows router decision; `single` and `parallel` force behavior.
            fail_safe: Runtime limits and failure policies for this agent.
            skill_tool_policy: Security/limits policy for built-in skill tools.
            enable_skill_tools: Whether to auto-register built-in skill tools.
            runner: Optional runner override; defaults to `Runner()` at call time.

        Raises:
            AgentConfigurationError: If `max_steps < 1` or
                `subagent_parallelism_mode` is invalid.
        """
        self.model = model
        self.name = name or self.__class__.__name__
        self.tools = list(tools or [])
        self.subagents = list(subagents or [])
        self.instructions = instructions
        self.context_defaults = dict(context_defaults or {})
        self.inherit_context_keys = list(inherit_context_keys or [])
        self.model_resolver = model_resolver
        self.skills = list(skills or [])
        self.skills_dir = Path(skills_dir)
        self.instruction_roles = list(instruction_roles or [])
        self.policy_roles = list(policy_roles or [])
        self.policy_engine = policy_engine
        self.subagent_router = subagent_router
        self.max_steps = max_steps
        self.tool_parallelism = tool_parallelism
        self.subagent_parallelism_mode = subagent_parallelism_mode
        self.fail_safe = fail_safe or FailSafeConfig(max_steps=max_steps)
        self.skill_tool_policy = skill_tool_policy or SkillToolPolicy()
        self.enable_skill_tools = enable_skill_tools
        self.runner = runner

        if max_steps < 1:
            raise AgentConfigurationError("max_steps must be >= 1")
        if self.subagent_parallelism_mode not in {"single", "parallel", "configurable"}:
            raise AgentConfigurationError(
                "subagent_parallelism_mode must be one of: single, parallel, configurable"
            )

    async def resolve_instructions(
        self,
        context: dict[str, JSONValue],
    ) -> str | None:
        """
        Resolve base instructions and append runtime instruction-role outputs.

        Args:
            context: JSON-safe run context available to instruction providers.

        Returns:
            Joined instruction text, or `None` when no instruction content
            is produced.
        """
        chunks: list[str] = []
        if isinstance(self.instructions, str):
            text = self.instructions.strip()
            if text:
                chunks.append(text)
        elif callable(self.instructions):
            maybe = self.instructions(context)
            if inspect.isawaitable(maybe):
                maybe = await maybe
            if isinstance(maybe, str) and maybe.strip():
                chunks.append(maybe.strip())

        for role in self.instruction_roles:
            out = role(context, "running")
            if inspect.isawaitable(out):
                out = await out
            if isinstance(out, str):
                if out.strip():
                    chunks.append(out.strip())
            elif isinstance(out, list):
                for row in out:
                    if isinstance(row, str) and row.strip():
                        chunks.append(row.strip())

        if not chunks:
            return None
        return "\n\n".join(chunks)

    async def call(
        self,
        user_message: str | None = None,
        *,
        context: dict[str, JSONValue] | None = None,
        thread_id: str | None = None,
    ) -> AgentResult:
        """
        Execute this agent through a runner and return terminal result.

        Args:
            user_message: Optional user message to seed the run.
            context: Optional JSON-safe context payload for this run.
            thread_id: Optional thread identifier for memory continuity.

        Returns:
            Terminal `AgentResult` for the run.
        """
        from ..core.runner import Runner

        runner = self.runner or Runner()
        return await runner.run(
            self,
            user_message=user_message,
            context=context,
            thread_id=thread_id,
        )

    def build_tool_registry(
        self,
        *,
        extra_tools: list[Tool[Any, Any]] | None = None,
        policy: Callable[[str, dict[str, Any], ToolContext], None] | None = None,
        middlewares: list[Any] | None = None,
    ) -> ToolRegistry:
        """
        Create a per-agent isolated tool registry.

        Args:
            extra_tools: Additional tools to register for this registry only.
            policy: Optional tool-execution policy callback enforced by registry.
            middlewares: Optional tool middleware chain.

        Returns:
            Isolated `ToolRegistry` configured for this agent.
        """
        parallelism = self.tool_parallelism or self.fail_safe.max_parallel_tools
        registry = ToolRegistry(
            max_concurrency=parallelism,
            policy=policy,
            middlewares=middlewares,
        )
        for item in [*self.tools, *(extra_tools or [])]:
            tool = self._normalize_tool(item)
            registry.register(tool)
        return registry

    def _normalize_tool(self, candidate: ToolLike) -> Tool[Any, Any]:
        """
        Normalize a declared tool entry into a concrete `Tool`.

        Args:
            candidate: Either a `Tool` instance or callable returning `Tool`.

        Returns:
            Concrete `Tool` instance.

        Raises:
            AgentConfigurationError: If the candidate is not a valid tool shape.
        """
        if isinstance(candidate, Tool):
            return candidate
        if callable(candidate):
            value = candidate()
            if isinstance(value, Tool):
                return value
        raise AgentConfigurationError(
            f"Invalid tool '{candidate}'. Expected Tool or callable returning Tool."
        )


class Agent(BaseAgent):
    """Concrete base agent used by developers."""

    pass
