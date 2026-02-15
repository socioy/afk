"""
Provider-neutral types for AFK agent runtime contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    TypeAlias,
)

from ..llms.types import JSONValue, LLMResponse, Usage
from ..tools import ToolResult

if TYPE_CHECKING:
    from ..tools import Tool


AgentState = Literal[
    "pending",
    "running",
    "paused",
    "cancelling",
    "cancelled",
    "degraded",
    "failed",
    "completed",
]

SubagentParallelismMode = Literal["single", "parallel", "configurable"]
FailurePolicy = Literal[
    "retry_then_fail",
    "retry_then_degrade",
    "fail_fast",
    "continue_with_error",
    "retry_then_continue",
    "continue",
    "fail_run",
    "skip_action",
]

InteractionMode = Literal["headless", "interactive", "external"]
DecisionKind = Literal["allow", "deny", "defer"]
PolicyAction = Literal[
    "allow",
    "deny",
    "defer",
    "request_approval",
    "request_user_input",
]
AgentEventType = Literal[
    "run_started",
    "step_started",
    "policy_decision",
    "llm_called",
    "llm_completed",
    "tool_batch_started",
    "tool_completed",
    "subagent_started",
    "subagent_completed",
    "run_paused",
    "run_resumed",
    "run_cancelled",
    "run_interrupted",
    "run_failed",
    "run_completed",
    "warning",
]


@dataclass(frozen=True, slots=True)
class SkillRef:
    """
    Resolved skill metadata for a specific skill name.

    Attributes:
        name: Logical skill name requested by developer.
        root_dir: Absolute resolved skills root directory.
        skill_md_path: Absolute path to the skill's `SKILL.md`.
        checksum: Optional SHA checksum for skill content integrity tracking.
    """

    name: str
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
class UsageAggregate:
    """
    Aggregated token usage across LLM calls in a run.

    Attributes:
        input_tokens: Sum of prompt/input tokens.
        output_tokens: Sum of completion/output tokens.
        total_tokens: Sum of total token counts.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add_usage(self, usage: Usage) -> "UsageAggregate":
        """
        Return a new aggregate with additional usage applied.

        Args:
            usage: Usage payload from a single LLM response.

        Returns:
            New `UsageAggregate` with usage added.
        """
        return UsageAggregate(
            input_tokens=self.input_tokens + (usage.input_tokens or 0),
            output_tokens=self.output_tokens + (usage.output_tokens or 0),
            total_tokens=self.total_tokens + (usage.total_tokens or 0),
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionRecord:
    """
    Normalized record for one tool execution.

    Attributes:
        tool_name: Executed tool name.
        tool_call_id: Provider/LLM tool-call identifier when available.
        success: Whether tool execution succeeded.
        output: JSON-safe tool output payload.
        error: Error message when execution failed.
        latency_ms: Execution latency in milliseconds.
    """

    tool_name: str
    tool_call_id: str | None
    success: bool
    output: JSONValue | None = None
    error: str | None = None
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class SubagentExecutionRecord:
    """
    Normalized record for one subagent execution.

    Attributes:
        subagent_name: Executed subagent name.
        success: Whether subagent run succeeded.
        output_text: Final text returned by subagent.
        error: Error message when subagent failed.
        latency_ms: Subagent execution latency in milliseconds.
    """

    subagent_name: str
    success: bool
    output_text: str | None = None
    error: str | None = None
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class SkillReadRecord:
    """
    Record of reading skill metadata or files.

    Attributes:
        skill_name: Skill name being read.
        path: Absolute path accessed by the runtime/tool.
        checksum: Optional content checksum for auditability.
    """

    skill_name: str
    path: str
    checksum: str | None = None


@dataclass(frozen=True, slots=True)
class CommandExecutionRecord:
    """
    Record for skill/runtime command execution.

    Attributes:
        command: Executed command as argv list.
        exit_code: Process exit code.
        stdout: Captured stdout content (possibly truncated).
        stderr: Captured stderr content (possibly truncated).
        denied: Whether execution was denied before process launch.
    """

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    denied: bool = False


@dataclass(frozen=True, slots=True)
class AgentRunEvent:
    """
    Event emitted during agent execution lifecycle.

    Attributes:
        type: Event category (run/tool/subagent/policy/etc).
        run_id: Unique run identifier.
        thread_id: Thread identifier for memory continuity.
        state: Runtime state at event emission.
        step: Optional loop step index.
        message: Optional human-readable message.
        data: Structured event payload.
        schema_version: Event schema version string.
    """

    type: AgentEventType
    run_id: str
    thread_id: str
    state: AgentState
    step: int | None = None
    message: str | None = None
    data: dict[str, JSONValue] = field(default_factory=dict)
    schema_version: str = "v1"


@dataclass(frozen=True, slots=True)
class PolicyEvent:
    """
    Runtime policy hook payload.

    Attributes:
        event_type: Policy hook type (for example `tool_before_execute`).
        run_id: Current run identifier.
        thread_id: Current thread identifier.
        step: Current step index.
        context: JSON-safe run context snapshot.
        tool_name: Target tool name when event is tool-related.
        tool_args: JSON-safe tool arguments when relevant.
        subagent_name: Target subagent name when relevant.
        metadata: Additional runtime metadata for policy matching.
    """

    event_type: str
    run_id: str
    thread_id: str
    step: int
    context: dict[str, JSONValue]
    tool_name: str | None = None
    tool_args: dict[str, JSONValue] | None = None
    subagent_name: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """
    Action selected by policy engine or policy roles.

    Attributes:
        action: Policy action (`allow`, `deny`, `defer`, etc).
        reason: Optional human-readable explanation.
        updated_tool_args: Optional rewritten tool args for execution.
        request_payload: Payload for approval/input defer flows.
        policy_id: Identifier of matched policy rule.
        matched_rules: Ordered list of matched rule ids.
    """

    action: PolicyAction = "allow"
    reason: str | None = None
    updated_tool_args: dict[str, JSONValue] | None = None
    request_payload: dict[str, JSONValue] = field(default_factory=dict)
    policy_id: str | None = None
    matched_rules: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """
    Request payload for human approval interaction.

    Attributes:
        run_id: Run identifier.
        thread_id: Thread identifier.
        step: Current execution step.
        reason: Reason shown to the approver.
        payload: Additional JSON-safe context for approval UI.
    """

    run_id: str
    thread_id: str
    step: int
    reason: str
    payload: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UserInputRequest:
    """
    Request payload for human user-input interaction.

    Attributes:
        run_id: Run identifier.
        thread_id: Thread identifier.
        step: Current execution step.
        prompt: Prompt text for human response.
        payload: Additional JSON-safe context for the input request.
    """

    run_id: str
    thread_id: str
    step: int
    prompt: str
    payload: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DeferredDecision:
    """
    Deferred interaction token returned by interaction providers.

    Attributes:
        token: Opaque token used to resolve deferred decision later.
        message: Optional provider message for logs/UI.
    """

    token: str
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """
    Resolved decision for an approval request.

    Attributes:
        kind: Decision outcome (`allow`, `deny`, or `defer`).
        reason: Optional explanation of the decision.
    """

    kind: DecisionKind
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class UserInputDecision:
    """
    Resolved decision for a user-input request.

    Attributes:
        kind: Decision outcome (`allow`, `deny`, or `defer`).
        value: User-provided text value when available.
        reason: Optional explanation or fallback reason.
    """

    kind: DecisionKind
    value: str | None = None
    reason: str | None = None


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


@dataclass(frozen=True, slots=True)
class FailSafeConfig:
    """
    Runtime limits and failure-policy settings for an agent run.

    Attributes:
        llm_failure_policy: Strategy when LLM calls fail.
        tool_failure_policy: Strategy when tool calls fail.
        subagent_failure_policy: Strategy when subagent calls fail.
        approval_denial_policy: Strategy when approval is denied/timeouts.
        max_steps: Maximum run loop iterations.
        max_wall_time_s: Maximum wall-clock runtime.
        max_llm_calls: Maximum number of LLM invocations.
        max_tool_calls: Maximum number of tool invocations.
        max_parallel_tools: Max concurrent tools per batch.
        max_subagent_depth: Maximum subagent recursion depth.
        max_subagent_fanout_per_step: Maximum selected subagents per step.
        max_total_cost_usd: Optional cost ceiling for run termination.
        fallback_model_chain: Ordered fallback model list for LLM retries.
        breaker_failure_threshold: Breaker open threshold.
        breaker_cooldown_s: Breaker cooldown window in seconds.
    """

    llm_failure_policy: FailurePolicy = "retry_then_fail"
    tool_failure_policy: FailurePolicy = "continue_with_error"
    subagent_failure_policy: FailurePolicy = "continue"
    approval_denial_policy: FailurePolicy = "skip_action"
    max_steps: int = 20
    max_wall_time_s: float = 300.0
    max_llm_calls: int = 50
    max_tool_calls: int = 200
    max_parallel_tools: int = 16
    max_subagent_depth: int = 3
    max_subagent_fanout_per_step: int = 4
    max_total_cost_usd: float | None = None
    fallback_model_chain: list[str] = field(default_factory=list)
    breaker_failure_threshold: int = 5
    breaker_cooldown_s: float = 30.0


@dataclass(frozen=True, slots=True)
class AgentResult:
    """
    Terminal result payload returned by runner/agent calls.

    Attributes:
        run_id: Run identifier.
        thread_id: Thread identifier.
        state: Terminal agent state.
        final_text: Final assistant text.
        requested_model: User-requested model identifier.
        normalized_model: Effective model identifier used in runtime.
        provider_adapter: Adapter/provider id used for execution.
        final_structured: Final structured output payload when available.
        llm_response: Final raw normalized LLM response.
        tool_executions: Ordered tool execution records.
        subagent_executions: Ordered subagent execution records.
        skills_used: Skill names enabled for this run.
        skill_reads: Skill file read audit records.
        skill_command_executions: Skill command execution records.
        usage_aggregate: Total token usage across LLM calls.
        total_cost_usd: Optional aggregated cost in USD.
        session_token: Optional provider session token for resume.
        checkpoint_token: Optional provider checkpoint token for resume.
        state_snapshot: Terminal runtime snapshot payload.
    """

    run_id: str
    thread_id: str
    state: AgentState
    final_text: str
    requested_model: str | None = None
    normalized_model: str | None = None
    provider_adapter: str | None = None
    final_structured: dict[str, JSONValue] | None = None
    llm_response: LLMResponse | None = None
    tool_executions: list[ToolExecutionRecord] = field(default_factory=list)
    subagent_executions: list[SubagentExecutionRecord] = field(default_factory=list)
    skills_used: list[str] = field(default_factory=list)
    skill_reads: list[SkillReadRecord] = field(default_factory=list)
    skill_command_executions: list[CommandExecutionRecord] = field(default_factory=list)
    usage_aggregate: UsageAggregate = field(default_factory=UsageAggregate)
    total_cost_usd: float | None = None
    session_token: str | None = None
    checkpoint_token: str | None = None
    state_snapshot: dict[str, JSONValue] = field(default_factory=dict)


InstructionProvider: TypeAlias = Callable[[dict[str, JSONValue]], str]
ToolLike: TypeAlias = "Tool[Any, Any] | Callable[[], Tool[Any, Any]]"
ContextInheritance: TypeAlias = list[str]


class InstructionRole(Protocol):
    """
    Hook protocol for dynamic instruction augmentation.

    Implementations receive run context and current state and can return:
    one string, a list of strings, or `None`.
    """

    def __call__(
        self,
        context: dict[str, JSONValue],
        state: AgentState,
    ) -> str | list[str] | None | Awaitable[str | list[str] | None]:
        """
        Return additional instruction text for current state.

        Args:
            context: JSON-safe run context.
            state: Current runtime state.

        Returns:
            Optional instruction text chunks.
        """
        ...


class PolicyRole(Protocol):
    """
    Hook protocol for runtime policy decisions.

    Implementations can deny/defer/rewrite runtime actions before execution.
    """

    def __call__(
        self,
        event: PolicyEvent,
    ) -> PolicyDecision | Awaitable[PolicyDecision]:
        """
        Return a policy decision for the given runtime event.

        Args:
            event: Runtime event payload under policy evaluation.

        Returns:
            Policy decision for the event.
        """
        ...


class SubagentRouter(Protocol):
    """
    Hook protocol for selecting subagents during execution.

    Router implementations decide target subagents and whether they run in
    parallel for the current step.
    """

    def __call__(
        self,
        data: RouterInput,
    ) -> RouterDecision | Awaitable[RouterDecision]:
        """
        Return a routing decision for subagent execution.

        Args:
            data: Router input payload with context and transcript.

        Returns:
            Router decision containing targets and parallelism flag.
        """
        ...


class AgentRunHandle(Protocol):
    """Protocol for asynchronous run lifecycle controls."""

    @property
    def events(self) -> AsyncIterator[AgentRunEvent]:
        """
        Event stream for the run lifecycle.

        Returns:
            Async iterator of `AgentRunEvent`.
        """
        ...

    async def pause(self) -> None:
        """Pause cooperative execution at safe boundaries."""
        ...

    async def resume(self) -> None:
        """Resume execution after pause."""
        ...

    async def cancel(self) -> None:
        """Cancel run and terminate with no result."""
        ...

    async def interrupt(self) -> None:
        """Interrupt in-flight operations where supported."""
        ...

    async def await_result(self) -> AgentResult | None:
        """
        Await terminal result.

        Returns:
            `AgentResult` on completion, or `None` when cancelled.
        """
        ...


def json_value_from_tool_result(value: Any) -> JSONValue:
    """
    Best-effort conversion of tool outputs to JSON-safe payloads.

    Args:
        value: Arbitrary tool output object.

    Returns:
        JSON-safe equivalent value. Unsupported objects are stringified with
        `repr(...)`.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [json_value_from_tool_result(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            out[str(key)] = json_value_from_tool_result(item)
        return out
    return repr(value)


def tool_record_from_result(
    tool_name: str,
    tool_call_id: str | None,
    result: ToolResult[Any],
    *,
    latency_ms: float | None = None,
) -> ToolExecutionRecord:
    """
    Convert a `ToolResult` into a normalized execution record.

    Args:
        tool_name: Name of executed tool.
        tool_call_id: Tool-call identifier from LLM, when present.
        result: Raw tool execution result object.
        latency_ms: Optional measured latency in milliseconds.

    Returns:
        Normalized `ToolExecutionRecord` for event/result reporting.
    """
    return ToolExecutionRecord(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        success=result.success,
        output=json_value_from_tool_result(result.output),
        error=result.error_message,
        latency_ms=latency_ms,
    )
