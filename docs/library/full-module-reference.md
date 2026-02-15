# AFK Full Module Reference

This file is generated from the code under `afk` and includes every module, top-level symbol, and class method signature.

Use this as the exhaustive API map for internal and public code paths.

## How To Use This File

- Use this for exact symbol lookup and code navigation.
- Prefer [API Reference](./api-reference.md) for human-guided explanations.
- Prefer [Developer Onboarding Guide](./developer-guide.md) for first-time setup and workflows.
- Treat this file as generated reference; optimize readability in companion docs, not here.

## Package: `agents`

### Module `afk/agents/__init__.py`

Source: [afk/agents/__init__.py](../../src/afk/agents/__init__.py)

Module summary: AFK agent public API.

Symbols: none (re-export or namespace module).

### Module `afk/agents/base.py`

Source: [afk/agents/base.py](../../src/afk/agents/base.py)

Module summary: Base agent abstractions with DX-first constructor.

Top-level symbols:

- `class BaseAgent` at `afk/agents/base.py:33` - Canonical agent configuration object consumed by the runner.
  Methods:
  - `sync __init__(self, *, model: str | LLM, name: str | None=None, tools: list[ToolLike] | None=None, subagents: list['BaseAgent'] | None=None, instructions: str | InstructionProvider | None=None, context_defaults: dict[str, JSONValue] | None=None, inherit_context_keys: list[str] | None=None, model_resolver: 'ModelResolver | None'=None, skills: list[str] | None=None, skills_dir: str | Path='.agents/skills', instruction_roles: list[InstructionRole] | None=None, policy_roles: list[PolicyRole] | None=None, policy_engine: 'PolicyEngine | None'=None, subagent_router: SubagentRouter | None=None, max_steps: int=20, tool_parallelism: int | None=None, subagent_parallelism_mode: SubagentParallelismMode='configurable', fail_safe: FailSafeConfig | None=None, skill_tool_policy: SkillToolPolicy | None=None, enable_skill_tools: bool=True, runner: 'Runner | None'=None)` at `afk/agents/base.py:41` - Initialize an agent definition.
  - `async resolve_instructions(self, context: dict[str, JSONValue])` at `afk/agents/base.py:133` - Resolve base instructions and append runtime instruction-role outputs.
  - `async call(self, user_message: str | None=None, *, context: dict[str, JSONValue] | None=None, thread_id: str | None=None)` at `afk/agents/base.py:175` - Execute this agent through a runner and return terminal result.
  - `sync build_tool_registry(self, *, extra_tools: list[Tool[Any, Any]] | None=None, policy: Callable[[str, dict[str, Any], ToolContext], None] | None=None, middlewares: list[Any] | None=None)` at `afk/agents/base.py:203` - Create a per-agent isolated tool registry.
  - `sync _normalize_tool(self, candidate: ToolLike)` at `afk/agents/base.py:232` - Normalize a declared tool entry into a concrete `Tool`.
- `class Agent` at `afk/agents/base.py:256` - Concrete base agent used by developers.

### Module `afk/agents/chat.py`

Source: [afk/agents/chat.py](../../src/afk/agents/chat.py)

Module summary: Chat-oriented agent convenience wrapper.

Top-level symbols:

- `class ChatAgent` at `afk/agents/chat.py:11` - Convenience agent requiring a user message for each call.
  Methods:
  - `async call(self, user_message: str, *, context: dict[str, JSONValue] | None=None, thread_id: str | None=None)` at `afk/agents/chat.py:16` - Run chat flow with a required user message.

### Module `afk/agents/errors.py`

Source: [afk/agents/errors.py](../../src/afk/agents/errors.py)

Module summary: Agent-layer error taxonomy.

Top-level symbols:

- `class AgentError` at `afk/agents/errors.py:8` - Base exception for all agent-runtime failures.
- `class AgentConfigurationError` at `afk/agents/errors.py:13` - Raised when agent configuration is invalid.
- `class AgentExecutionError` at `afk/agents/errors.py:25` - Raised for runtime execution failures not tied to configuration.
- `class AgentRetryableError` at `afk/agents/errors.py:30` - Raised for runtime failures that may be retried safely.
- `class AgentLoopLimitError` at `afk/agents/errors.py:35` - Raised when loop/step guard limits are exceeded.
- `class AgentBudgetExceededError` at `afk/agents/errors.py:40` - Raised when wall-time/cost/token/tool budgets are exceeded.
- `class AgentCancelledError` at `afk/agents/errors.py:45` - Raised when a run is cancelled by caller or control plane.
- `class AgentInterruptedError` at `afk/agents/errors.py:50` - Raised when a run is interrupted mid-execution.
- `class AgentPausedError` at `afk/agents/errors.py:55` - Raised when an operation is attempted while run is paused.
- `class SubagentRoutingError` at `afk/agents/errors.py:60` - Raised when subagent routing is invalid, unsafe, or inconsistent.
- `class SubagentExecutionError` at `afk/agents/errors.py:65` - Raised when delegated subagent execution fails.
- `class SkillResolutionError` at `afk/agents/errors.py:70` - Raised when resolving requested skills from `skills_dir` fails.
- `class SkillAccessError` at `afk/agents/errors.py:76` - Raised when skill file access violates path safety constraints.
- `class SkillCommandDeniedError` at `afk/agents/errors.py:81` - Raised when skill command execution is denied by policy or allowlist.
- `class AgentCheckpointCorruptionError` at `afk/agents/errors.py:86` - Raised when checkpoint payload cannot be validated or loaded.
- `class AgentCircuitOpenError` at `afk/agents/errors.py:91` - Raised when an open circuit breaker blocks dependency execution.

### Module `afk/agents/policy.py`

Source: [afk/agents/policy.py](../../src/afk/agents/policy.py)

Module summary: Rule-based policy engine for agent/runtime controls.

Top-level symbols:

- `class PolicyRuleCondition` at `afk/agents/policy.py:18` - Match conditions used by policy rules.
  Methods:
  - `sync matches(self, event: PolicyEvent)` at `afk/agents/policy.py:29` - Return `True` when this condition matches the given event.
- `class PolicyRule` at `afk/agents/policy.py:61` - Single policy rule with deterministic priority ordering.
  Methods:
  - `sync applies_to(self, event: PolicyEvent)` at `afk/agents/policy.py:74` - Return `True` when this rule applies to the given event.
- `class PolicyEvaluation` at `afk/agents/policy.py:88` - Evaluation output containing final decision and matched rules.
- `class PolicyEngine` at `afk/agents/policy.py:95` - Deterministic rule evaluator.
  Methods:
  - `sync __init__(self, rules: Iterable[PolicyRule] | None=None)` at `afk/agents/policy.py:105`
  - `sync rules(self)` at `afk/agents/policy.py:112` - Return configured rules in evaluation order.
  - `sync evaluate(self, event: PolicyEvent)` at `afk/agents/policy.py:116` - Evaluate rules and return the selected policy decision.
- `def infer_policy_subject(event_type: str)` at `afk/agents/policy.py:147` - Infer policy subject channel from event type name.
- `def normalize_policy_payload(payload: dict[str, Any] | None)` at `afk/agents/policy.py:161` - Normalize arbitrary payload values into JSON-safe policy payload.

### Module `afk/agents/resolution.py`

Source: [afk/agents/resolution.py](../../src/afk/agents/resolution.py)

Module summary: LLM resolution helpers for agent runtime.

Top-level symbols:

- `class ResolvedModel` at `afk/agents/resolution.py:18` - Resolved model metadata plus instantiated LLM adapter.
- `def resolve_model_to_llm(model: str | LLM, *, resolver: ModelResolver | None=None)` at `afk/agents/resolution.py:27` - Resolve model input into a concrete LLM adapter + normalized model name.
- `def _split_prefix(model_name: str)` at `afk/agents/resolution.py:112` - Split `provider/model` strings into provider prefix and model name.

### Module `afk/agents/runtime.py`

Source: [afk/agents/runtime.py](../../src/afk/agents/runtime.py)

Module summary: Runtime helpers used by the core runner.

Top-level symbols:

- `def resolve_skills(*, skill_names: list[str], skills_dir: Path, cwd: Path)` at `afk/agents/runtime.py:24` - Resolve requested skill names to concrete SKILL.md paths.
- `def build_skill_manifest_prompt(skills: list[SkillRef])` at `afk/agents/runtime.py:65` - Build compact skill manifest text to place in system instructions.
- `def to_message_payload(role: str, text: str)` at `afk/agents/runtime.py:84` - Build a JSON-safe message payload used by router inputs.
- `def _is_under(path: Path, root: Path)` at `afk/agents/runtime.py:89` - Return `True` when `path` is within `root`.
- `def _sha256(blob: bytes)` at `afk/agents/runtime.py:98` - Return SHA-256 hex digest for bytes.
- `def json_hash(payload: dict[str, Any])` at `afk/agents/runtime.py:103` - Build a stable hash for JSON-like payloads.
- `def checkpoint_state_key(run_id: str, step: int, phase: str)` at `afk/agents/runtime.py:109` - Return namespaced key for step-level checkpoint state.
- `def checkpoint_latest_key(run_id: str)` at `afk/agents/runtime.py:114` - Return namespaced key for latest checkpoint pointer.
- `def effect_state_key(run_id: str, step: int, tool_call_id: str)` at `afk/agents/runtime.py:119` - Return namespaced key for idempotent tool-effect state.
- `def validate_state_transition(current: AgentState, target: AgentState)` at `afk/agents/runtime.py:136` - Validate and return a legal state transition target.
- `class EffectJournal` at `afk/agents/runtime.py:149` - Idempotency journal for tool-side effects.
  Methods:
  - `sync get(self, run_id: str, step: int, tool_call_id: str)` at `afk/agents/runtime.py:156` - Fetch a previously journaled tool-effect record.
  - `sync put(self, run_id: str, step: int, tool_call_id: str, input_hash: str, output_hash: str, *, output: JSONValue | None, success: bool)` at `afk/agents/runtime.py:160` - Store idempotency metadata for a side-effecting tool call.
- `class CircuitBreaker` at `afk/agents/runtime.py:181` - Minimal breaker for runtime dependencies.
  Methods:
  - `sync record_success(self, key: str)` at `afk/agents/runtime.py:189` - Reset failure history for a dependency after success.
  - `sync record_failure(self, key: str)` at `afk/agents/runtime.py:193` - Record a dependency failure within cooldown window.
  - `sync ensure_closed(self, key: str)` at `afk/agents/runtime.py:201` - Raise when failure threshold is exceeded for dependency key.
- `def state_snapshot(*, state: AgentState, step: int, llm_calls: int, tool_calls: int, started_at_s: float, requested_model: str | None=None, normalized_model: str | None=None, provider_adapter: str | None=None, total_cost_usd: float | None=None, replayed_effect_count: int=0)` at `afk/agents/runtime.py:211` - Return normalized runtime snapshot payload for checkpoint persistence.

### Module `afk/agents/security.py`

Source: [afk/agents/security.py](../../src/afk/agents/security.py)

Module summary: Prompt/tool-output security helpers.

Top-level symbols:

- `def trusted_system_channel_header` at `afk/agents/security.py:29` - Return marker for trusted system-originated content.
- `def untrusted_tool_channel_header(tool_name: str)` at `afk/agents/security.py:34` - Return marker for untrusted tool-output content.
- `def sanitize_text(text: str, *, max_chars: int)` at `afk/agents/security.py:39` - Redact suspicious prompt-injection markers and enforce length limits.
- `def sanitize_json_value(value: Any, *, max_chars: int)` at `afk/agents/security.py:50` - Recursively sanitize JSON-like values from untrusted tool output.
- `def render_untrusted_tool_message(*, tool_name: str, payload: dict[str, Any], max_chars: int)` at `afk/agents/security.py:65` - Render sanitized untrusted tool output for model-visible transcript.

### Module `afk/agents/types.py`

Source: [afk/agents/types.py](../../src/afk/agents/types.py)

Module summary: Provider-neutral types for AFK agent runtime contracts.

Top-level symbols:

- `class SkillRef` at `afk/agents/types.py:79` - Resolved skill metadata for a specific skill name.
- `class SkillResolutionResult` at `afk/agents/types.py:97` - Result of resolving requested skills from the skills directory.
- `class SkillToolPolicy` at `afk/agents/types.py:111` - Security and execution limits applied to skill command tools.
- `class UsageAggregate` at `afk/agents/types.py:131` - Aggregated token usage across LLM calls in a run.
  Methods:
  - `sync add_usage(self, usage: Usage)` at `afk/agents/types.py:145` - Return a new aggregate with additional usage applied.
- `class ToolExecutionRecord` at `afk/agents/types.py:163` - Normalized record for one tool execution.
- `class SubagentExecutionRecord` at `afk/agents/types.py:185` - Normalized record for one subagent execution.
- `class SkillReadRecord` at `afk/agents/types.py:205` - Record of reading skill metadata or files.
- `class CommandExecutionRecord` at `afk/agents/types.py:221` - Record for skill/runtime command execution.
- `class AgentRunEvent` at `afk/agents/types.py:241` - Event emitted during agent execution lifecycle.
- `class PolicyEvent` at `afk/agents/types.py:267` - Runtime policy hook payload.
- `class PolicyDecision` at `afk/agents/types.py:295` - Action selected by policy engine or policy roles.
- `class ApprovalRequest` at `afk/agents/types.py:317` - Request payload for human approval interaction.
- `class UserInputRequest` at `afk/agents/types.py:337` - Request payload for human user-input interaction.
- `class DeferredDecision` at `afk/agents/types.py:357` - Deferred interaction token returned by interaction providers.
- `class ApprovalDecision` at `afk/agents/types.py:371` - Resolved decision for an approval request.
- `class UserInputDecision` at `afk/agents/types.py:385` - Resolved decision for a user-input request.
- `class RouterInput` at `afk/agents/types.py:401` - Payload passed into subagent router callbacks.
- `class RouterDecision` at `afk/agents/types.py:421` - Subagent routing decision returned by router callbacks.
- `class FailSafeConfig` at `afk/agents/types.py:437` - Runtime limits and failure-policy settings for an agent run.
- `class AgentResult` at `afk/agents/types.py:477` - Terminal result payload returned by runner/agent calls.
- `class InstructionRole` at `afk/agents/types.py:529` - Hook protocol for dynamic instruction augmentation.
  Methods:
  - `sync __call__(self, context: dict[str, JSONValue], state: AgentState)` at `afk/agents/types.py:537` - Return additional instruction text for current state.
- `class PolicyRole` at `afk/agents/types.py:555` - Hook protocol for runtime policy decisions.
  Methods:
  - `sync __call__(self, event: PolicyEvent)` at `afk/agents/types.py:562` - Return a policy decision for the given runtime event.
- `class SubagentRouter` at `afk/agents/types.py:578` - Hook protocol for selecting subagents during execution.
  Methods:
  - `sync __call__(self, data: RouterInput)` at `afk/agents/types.py:586` - Return a routing decision for subagent execution.
- `class AgentRunHandle` at `afk/agents/types.py:602` - Protocol for asynchronous run lifecycle controls.
  Methods:
  - `sync events(self)` at `afk/agents/types.py:606` - Event stream for the run lifecycle.
  - `async pause(self)` at `afk/agents/types.py:615` - Pause cooperative execution at safe boundaries.
  - `async resume(self)` at `afk/agents/types.py:619` - Resume execution after pause.
  - `async cancel(self)` at `afk/agents/types.py:623` - Cancel run and terminate with no result.
  - `async interrupt(self)` at `afk/agents/types.py:627` - Interrupt in-flight operations where supported.
  - `async await_result(self)` at `afk/agents/types.py:631` - Await terminal result.
- `def json_value_from_tool_result(value: Any)` at `afk/agents/types.py:641` - Best-effort conversion of tool outputs to JSON-safe payloads.
- `def tool_record_from_result(tool_name: str, tool_call_id: str | None, result: ToolResult[Any], *, latency_ms: float | None=None)` at `afk/agents/types.py:664` - Convert a `ToolResult` into a normalized execution record.

### Module `afk/agents/versioning.py`

Source: [afk/agents/versioning.py](../../src/afk/agents/versioning.py)

Module summary: Schema versioning helpers for agent events/checkpoints.

Top-level symbols:

- `class VersionCheckResult` at `afk/agents/versioning.py:18` - Compatibility result for a schema version check.
- `class MigrationResult` at `afk/agents/versioning.py:36` - Migration result payload for legacy event/checkpoint records.
- `def check_event_schema_version(version: str | None)` at `afk/agents/versioning.py:53` - Validate event schema version compatibility.
- `def check_checkpoint_schema_version(version: str | None)` at `afk/agents/versioning.py:78` - Validate checkpoint schema version compatibility.
- `def migrate_event_record(record: dict[str, Any])` at `afk/agents/versioning.py:103` - Migrate legacy event records into current event schema.
- `def migrate_checkpoint_record(record: dict[str, Any])` at `afk/agents/versioning.py:151` - Migrate legacy checkpoint records into current checkpoint schema.

## Package: `core`

### Module `afk/core/__init__.py`

Source: [afk/core/__init__.py](../../src/afk/core/__init__.py)

Module summary: Core runtime exports.

Symbols: none (re-export or namespace module).

### Module `afk/core/interaction.py`

Source: [afk/core/interaction.py](../../src/afk/core/interaction.py)

Module summary: Pluggable interaction providers for human-in-the-loop workflows.

Top-level symbols:

- `class InteractionProvider` at `afk/core/interaction.py:22` - Runtime-portable interaction contract.
  Methods:
  - `async request_approval(self, request: ApprovalRequest)` at `afk/core/interaction.py:33` - Request approval for a gated action.
  - `async request_user_input(self, request: UserInputRequest)` at `afk/core/interaction.py:49` - Request user input for the active run.
  - `async await_deferred(self, token: str, *, timeout_s: float)` at `afk/core/interaction.py:65` - Wait for a deferred interaction decision.
  - `async notify(self, event: AgentRunEvent)` at `afk/core/interaction.py:84` - Receive lifecycle notifications emitted by the runner.
- `class HeadlessInteractionProvider` at `afk/core/interaction.py:95` - Non-blocking default interaction provider for autonomous/runtime-server use.
  Methods:
  - `async request_approval(self, request: ApprovalRequest)` at `afk/core/interaction.py:107` - Return an immediate fallback approval decision.
  - `async request_user_input(self, request: UserInputRequest)` at `afk/core/interaction.py:123` - Return an immediate fallback user-input decision.
  - `async await_deferred(self, token: str, *, timeout_s: float)` at `afk/core/interaction.py:139` - Resolve deferred tokens in headless mode.
  - `async notify(self, event: AgentRunEvent)` at `afk/core/interaction.py:162` - Receive lifecycle notifications.
- `class InMemoryInteractiveProvider` at `afk/core/interaction.py:174` - In-memory provider useful for tests/local development.
  Methods:
  - `async request_approval(self, request: ApprovalRequest)` at `afk/core/interaction.py:187` - Create a deferred approval token.
  - `async request_user_input(self, request: UserInputRequest)` at `afk/core/interaction.py:202` - Create a deferred user-input token.
  - `async await_deferred(self, token: str, *, timeout_s: float)` at `afk/core/interaction.py:217` - Resolve a deferred decision from memory or time out.
  - `async notify(self, event: AgentRunEvent)` at `afk/core/interaction.py:238` - Store emitted events for assertions/debugging.
  - `sync set_deferred_result(self, token: str, decision: ApprovalDecision | UserInputDecision)` at `afk/core/interaction.py:247` - Set a deferred decision to be returned by `await_deferred`.
  - `sync notifications(self)` at `afk/core/interaction.py:261` - Return captured notification events.

### Module `afk/core/runner.py`

Source: [afk/core/runner.py](../../src/afk/core/runner.py)

Module summary: Canonical AFK runner assembled from focused mixins.

Top-level symbols:

- `class Runner` at `afk/core/runner.py:15` - Canonical runtime runner for AFK agents.

### Module `afk/core/runner_api.py`

Source: [afk/core/runner_api.py](../../src/afk/core/runner_api.py)

Module summary: Public runner API and lifecycle entrypoints.

Top-level symbols:

- `class RunnerAPIMixin` at `afk/core/runner_api.py:27` - Public API surface for running, resuming, and compacting agent threads.
  Methods:
  - `sync __init__(self, *, memory_store: MemoryStore | None=None, interaction_provider: InteractionProvider | None=None, policy_engine: PolicyEngine | None=None, telemetry: TelemetrySink | None=None, config: RunnerConfig | None=None)` at `afk/core/runner_api.py:36` - Initialize a runner API surface with optional runtime dependencies.
  - `async compact_thread(self, *, thread_id: str, event_policy: RetentionPolicy | None=None, state_policy: StateRetentionPolicy | None=None)` at `afk/core/runner_api.py:82` - Compact retained memory records for a thread.
  - `async run(self, agent: BaseAgent, *, user_message: str | None=None, context: dict[str, Any] | None=None, thread_id: str | None=None)` at `afk/core/runner_api.py:113` - Execute an agent run and wait for terminal result.
  - `async resume(self, agent: BaseAgent, *, run_id: str, thread_id: str, context: dict[str, Any] | None=None)` at `afk/core/runner_api.py:147` - Resume a previously checkpointed run and wait for completion.
  - `async resume_handle(self, agent: BaseAgent, *, run_id: str, thread_id: str, context: dict[str, Any] | None=None)` at `afk/core/runner_api.py:181` - Resume a run and return a live handle for lifecycle control.
  - `async run_handle(self, agent: BaseAgent, *, user_message: str | None=None, context: dict[str, Any] | None=None, thread_id: str | None=None, _depth: int=0, _lineage: tuple[int, ...]=(), _resume_run_id: str | None=None, _resume_snapshot: dict[str, Any] | None=None)` at `afk/core/runner_api.py:247` - Start execution and return an async run handle.

### Module `afk/core/runner_execution.py`

Source: [afk/core/runner_execution.py](../../src/afk/core/runner_execution.py)

Module summary: Core execution loop for the AFK runner.

Top-level symbols:

- `class RunnerExecutionMixin` at `afk/core/runner_execution.py:63` - Implements the main agent loop, tool orchestration, and recovery flow.
  Methods:
  - `async _execute(self, handle: _RunHandle, agent: BaseAgent, *, user_message: str | None, context: dict[str, Any] | None, thread_id: str | None, depth: int, lineage: tuple[int, ...], resume_run_id: str | None=None, resume_snapshot: dict[str, Any] | None=None)` at `afk/core/runner_execution.py:66` - Execute one agent run end-to-end and resolve the run handle.

### Module `afk/core/runner_interaction.py`

Source: [afk/core/runner_interaction.py](../../src/afk/core/runner_interaction.py)

Module summary: Subagent and interaction/policy orchestration mixin.

Top-level symbols:

- `class RunnerInteractionMixin` at `afk/core/runner_interaction.py:40` - Implements policy evaluation, subagent dispatch, and HITL requests.
  Methods:
  - `async _run_subagents(self, *, agent: BaseAgent, targets: list[str], parallel: bool, context: dict[str, Any], thread_id: str, depth: int, lineage: tuple[int, ...], run_id: str, step: int, handle: _RunHandle, memory: MemoryStore, user_id: str | None)` at `afk/core/runner_interaction.py:43` - Execute selected subagents and return records plus bridge text.
  - `async _call_router(self, agent: BaseAgent, *, run_id: str, thread_id: str, step: int, context: dict[str, Any], messages: list[Message])` at `afk/core/runner_interaction.py:203` - Invoke configured subagent router and normalize response.
  - `async _evaluate_policy(self, *, agent: BaseAgent, event: PolicyEvent, handle: _RunHandle | None=None, memory: MemoryStore | None=None, user_id: str | None=None, state: AgentState='running')` at `afk/core/runner_interaction.py:258` - Evaluate policy engine and policy roles for runtime event.
  - `async _emit_policy_audit(self, *, handle: _RunHandle | None, memory: MemoryStore | None, event: PolicyEvent, decision: PolicyDecision, user_id: str | None, state: AgentState)` at `afk/core/runner_interaction.py:347` - Emit normalized policy-decision audit event.
  - `async _request_approval(self, *, handle: _RunHandle, memory: MemoryStore, run_id: str, thread_id: str, step: int, reason: str, payload: dict[str, Any], user_id: str | None)` at `afk/core/runner_interaction.py:390` - Request approval and handle deferred pause/resume flow.
  - `async _request_user_input(self, *, handle: _RunHandle, memory: MemoryStore, run_id: str, thread_id: str, step: int, prompt: str, payload: dict[str, Any], user_id: str | None)` at `afk/core/runner_interaction.py:512` - Request user input and handle deferred pause/resume flow.
  - `sync _is_defer_user_input(self, decision: PolicyDecision)` at `afk/core/runner_interaction.py:634` - Check whether a defer decision represents user-input interaction.
  - `sync _resolve_user_input_prompt(self, *, tool_name: str, decision: PolicyDecision)` at `afk/core/runner_interaction.py:649` - Resolve prompt text used for user-input interaction requests.

### Module `afk/core/runner_internals.py`

Source: [afk/core/runner_internals.py](../../src/afk/core/runner_internals.py)

Module summary: Runner helpers for persistence, serialization, budgeting, and replay logic.

Top-level symbols:

- `class RunnerInternalsMixin` at `afk/core/runner_internals.py:51` - Shared internal helpers used by the execution and API mixins.
  Methods:
  - `sync _enforce_budget(self, *, fail_safe: FailSafeConfig, step: int, llm_calls: int, tool_calls: int, started_at_s: float, total_cost_usd: float)` at `afk/core/runner_internals.py:54` - Enforce runtime budget and loop guards.
  - `async _emit(self, handle: _RunHandle, memory: MemoryStore, event: AgentRunEvent, *, user_id: str | None)` at `afk/core/runner_internals.py:102` - Emit lifecycle event to handle, interaction provider, memory, telemetry.
  - `async _append_event_with_retry(self, memory: MemoryStore, event: MemoryEvent)` at `afk/core/runner_internals.py:166` - Append memory event with best-effort retry.
  - `async _ensure_memory_store(self)` at `afk/core/runner_internals.py:191` - Ensure memory store is initialized and ready.
  - `sync _transition_state(self, current: AgentState, target: AgentState)` at `afk/core/runner_internals.py:210` - Validate and return next state transition.
  - `async _persist_checkpoint(self, *, memory: MemoryStore, thread_id: str, run_id: str, step: int, phase: str, payload: dict[str, Any])` at `afk/core/runner_internals.py:223` - Persist checkpoint for step/phase and update latest pointer.
  - `sync _build_llm_candidates(self, *, primary, fallback_chain: list[str], resolver)` at `afk/core/runner_internals.py:263` - Build ordered unique list of primary + fallback LLM candidates.
  - `sync _apply_llm_failure_policy(self, policy: str)` at `afk/core/runner_internals.py:294` - Normalize LLM failure policy to runtime action.
  - `sync _apply_tool_failure_policy(self, policy: str)` at `afk/core/runner_internals.py:310` - Normalize tool failure policy to runtime action.
  - `sync _apply_subagent_failure_policy(self, policy: str)` at `afk/core/runner_internals.py:326` - Normalize subagent failure policy to runtime action.
  - `sync _apply_approval_denial_policy(self, policy: str)` at `afk/core/runner_internals.py:342` - Normalize approval denial policy to runtime action.
  - `sync _resolve_subagent_parallel(self, *, agent_parallelism_mode: str, router_parallel: bool)` at `afk/core/runner_internals.py:358` - Resolve effective subagent parallelism mode.
  - `sync _build_subagent_context(self, *, context: dict[str, Any], inherit_keys: list[str])` at `afk/core/runner_internals.py:381` - Build subagent context from inherited key allowlist.
  - `sync _accumulate_cost(self, current_cost: float, response)` at `afk/core/runner_internals.py:405` - Add response cost (if available) to current aggregate cost.
  - `async _resolve_effect_replay_result(self, *, memory: MemoryStore, thread_id: str, run_id: str, step: int, tool_call_id: str | None, tool_name: str, call_args: dict[str, Any])` at `afk/core/runner_internals.py:429` - Resolve replayable tool result from effect journal/checkpoint state.
  - `async _persist_effect_result(self, *, memory: MemoryStore, thread_id: str, run_id: str, step: int, tool_call_id: str, input_hash: str, output_hash: str, output: Any, success: bool)` at `afk/core/runner_internals.py:499` - Persist tool effect journal row for idempotent replay.
  - `async _chat_with_interrupt_support(self, *, handle: _RunHandle, llm: Any, req: LLMRequest)` at `afk/core/runner_internals.py:548` - Execute chat call with optional provider interrupt wiring.
  - `async _persist_runtime_snapshot(self, *, memory: MemoryStore, thread_id: str, run_id: str, step: int, state: AgentState, context: dict[str, Any], messages: list[Message], llm_calls: int, tool_calls: int, started_at_s: float, usage: UsageAggregate, total_cost_usd: float, session_token: str | None, checkpoint_token: str | None, requested_model: str | None, normalized_model: str | None, provider_adapter: str | None, tool_execs: list[ToolExecutionRecord], sub_execs: list[SubagentExecutionRecord], skill_reads: list[SkillReadRecord], skill_cmd_execs: list[CommandExecutionRecord], final_text: str, final_structured: dict[str, Any] | None, pending_llm_response: LLMResponse | None, final_response: LLMResponse | None, replayed_effect_count: int)` at `afk/core/runner_internals.py:586` - Persist checkpoint payload containing full runtime snapshot.
  - `sync _restore_runtime_snapshot(self, snapshot: dict[str, Any])` at `afk/core/runner_internals.py:690` - Extract runtime payload from normalized checkpoint snapshot.
  - `sync _new_id(self, prefix: str)` at `afk/core/runner_internals.py:708` - Generate unique id via runner module indirection.
  - `sync _create_memory_store_from_env(self)` at `afk/core/runner_internals.py:722` - Create memory store from environment configuration.
  - `async _load_latest_runtime_snapshot(self, *, memory: MemoryStore, thread_id: str, run_id: str, latest: dict[str, Any])` at `afk/core/runner_internals.py:733` - Load most recent runtime_state checkpoint for resume.
  - `sync _normalize_checkpoint_record(self, record: dict[str, Any])` at `afk/core/runner_internals.py:776` - Migrate and validate checkpoint record.
  - `sync _serialize_messages(self, messages: list[Message])` at `afk/core/runner_internals.py:801` - Serialize message objects into checkpoint-friendly dictionaries.
  - `sync _deserialize_messages(self, value: Any)` at `afk/core/runner_internals.py:835` - Deserialize messages from checkpoint payload.
  - `sync _serialize_llm_response(self, value: LLMResponse | None)` at `afk/core/runner_internals.py:866` - Serialize optional LLM response for checkpoint persistence.
  - `sync _deserialize_llm_response(self, value: Any)` at `afk/core/runner_internals.py:903` - Deserialize optional LLM response from checkpoint payload.
  - `sync _serialize_tool_records(self, rows: list[ToolExecutionRecord])` at `afk/core/runner_internals.py:974` - Serialize tool execution records.
  - `sync _deserialize_tool_records(self, value: Any)` at `afk/core/runner_internals.py:996` - Deserialize tool execution records.
  - `sync _serialize_subagent_records(self, rows: list[SubagentExecutionRecord])` at `afk/core/runner_internals.py:1031` - Serialize subagent execution records.
  - `sync _deserialize_subagent_records(self, value: Any)` at `afk/core/runner_internals.py:1055` - Deserialize subagent execution records.
  - `sync _serialize_skill_reads(self, rows: list[SkillReadRecord])` at `afk/core/runner_internals.py:1089` - Serialize skill read records.
  - `sync _deserialize_skill_reads(self, value: Any)` at `afk/core/runner_internals.py:1108` - Deserialize skill read records.
  - `sync _serialize_command_records(self, rows: list[CommandExecutionRecord])` at `afk/core/runner_internals.py:1139` - Serialize command execution records.
  - `sync _deserialize_command_records(self, value: Any)` at `afk/core/runner_internals.py:1163` - Deserialize command execution records.
  - `sync _build_terminal_result(self, *, run_id: str, thread_id: str, state: AgentState, final_text: str, requested_model: str | None, normalized_model: str | None, provider_adapter: str | None, final_structured: dict[str, Any] | None, llm_response: LLMResponse | None, tool_execs: list[ToolExecutionRecord], sub_execs: list[SubagentExecutionRecord], skills: list[Any], skill_reads: list[SkillReadRecord], skill_cmd_execs: list[CommandExecutionRecord], usage: UsageAggregate, total_cost_usd: float, session_token: str | None, checkpoint_token: str | None, step: int, llm_calls: int, tool_calls: int, started_at_s: float, replayed_effect_count: int)` at `afk/core/runner_internals.py:1193` - Build terminal `AgentResult` from accumulated runtime state.
  - `sync _serialize_agent_result(self, result: AgentResult)` at `afk/core/runner_internals.py:1284` - Serialize `AgentResult` for checkpoint storage.
  - `sync _deserialize_agent_result(self, value: dict[str, Any])` at `afk/core/runner_internals.py:1324` - Deserialize `AgentResult` from checkpoint payload.
  - `sync _maybe_str(self, value: Any)` at `afk/core/runner_internals.py:1392` - Return string value when input is a string.
  - `sync _telemetry_start_span(self, name: str, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/runner_internals.py:1404` - Safely start telemetry span.
  - `sync _telemetry_end_span(self, span: TelemetrySpan | None, *, status: str, error: str | None=None, attributes: dict[str, JSONValue] | None=None)` at `afk/core/runner_internals.py:1425` - Safely end telemetry span.
  - `sync _telemetry_counter(self, name: str, *, value: int=1, attributes: dict[str, JSONValue] | None=None)` at `afk/core/runner_internals.py:1452` - Safely record telemetry counter.
  - `sync _telemetry_histogram(self, name: str, *, value: float, attributes: dict[str, JSONValue] | None=None)` at `afk/core/runner_internals.py:1472` - Safely record telemetry histogram value.

### Module `afk/core/runner_types.py`

Source: [afk/core/runner_types.py](../../src/afk/core/runner_types.py)

Module summary: Shared runtime types for AFK runner internals.

Top-level symbols:

- `class RunnerConfig` at `afk/core/runner_types.py:22` - Runtime configuration for runner behavior and safety defaults.
- `class _RunHandle` at `afk/core/runner_types.py:66` - Concrete async run handle used by the runner implementation.
  Methods:
  - `sync __init__(self)` at `afk/core/runner_types.py:74` - Initialize queue, result future, and lifecycle flags.
  - `sync attach_task(self, task: asyncio.Task[None])` at `afk/core/runner_types.py:87` - Attach the underlying execution task.
  - `sync set_interrupt_callback(self, callback)` at `afk/core/runner_types.py:96` - Register provider-specific interrupt callback.
  - `sync events(self)` at `afk/core/runner_types.py:106` - Return run event stream.
  - `async _iter_events(self)` at `afk/core/runner_types.py:121` - Internal async generator that yields queued events until run end marker.
  - `async emit(self, event: AgentRunEvent)` at `afk/core/runner_types.py:129` - Push event into handle stream.
  - `async pause(self)` at `afk/core/runner_types.py:138` - Pause cooperative execution at safe boundaries.
  - `async resume(self)` at `afk/core/runner_types.py:143` - Resume a paused run.
  - `async cancel(self)` at `afk/core/runner_types.py:148` - Request cancellation and resolve handle with `None`.
  - `async interrupt(self)` at `afk/core/runner_types.py:161` - Request interruption and invoke interrupt callback if available.
  - `async await_result(self)` at `afk/core/runner_types.py:178` - Await terminal run result.
  - `async wait_if_paused(self)` at `afk/core/runner_types.py:187` - Block until resume is requested.
  - `sync is_cancel_requested(self)` at `afk/core/runner_types.py:191` - Return `True` when cancellation has been requested.
  - `sync is_interrupt_requested(self)` at `afk/core/runner_types.py:195` - Return `True` when interruption has been requested.
  - `async set_result(self, result: AgentResult | None)` at `afk/core/runner_types.py:199` - Set terminal result and close event stream.
  - `async set_exception(self, exc: Exception)` at `afk/core/runner_types.py:210` - Set terminal exception and close event stream.

### Module `afk/core/telemetry.py`

Source: [afk/core/telemetry.py](../../src/afk/core/telemetry.py)

Module summary: Telemetry sinks for runner observability.

Top-level symbols:

- `class TelemetryEvent` at `afk/core/telemetry.py:18` - Point-in-time telemetry event.
- `class TelemetrySpan` at `afk/core/telemetry.py:34` - Started telemetry span.
- `class TelemetrySink` at `afk/core/telemetry.py:51` - Protocol implemented by telemetry backends.
  Methods:
  - `sync record_event(self, event: TelemetryEvent)` at `afk/core/telemetry.py:54` - Record a single event.
  - `sync start_span(self, name: str, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:63` - Start a span when backend supports spans.
  - `sync end_span(self, span: TelemetrySpan | None, *, status: str, error: str | None=None, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:76` - End a span with status and optional metadata.
  - `sync increment_counter(self, name: str, value: int=1, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:95` - Increment a named counter.
  - `sync record_histogram(self, name: str, value: float, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:112` - Record a histogram measurement.
- `class NullTelemetrySink` at `afk/core/telemetry.py:131` - No-op telemetry sink used as safe default.
  Methods:
  - `sync record_event(self, event: TelemetryEvent)` at `afk/core/telemetry.py:134` - Ignore event payload.
  - `sync start_span(self, name: str, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:144` - Return `None` because spans are disabled.
  - `sync end_span(self, span: TelemetrySpan | None, *, status: str, error: str | None=None, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:156` - Ignore span completion call.
  - `sync increment_counter(self, name: str, value: int=1, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:179` - Ignore counter call.
  - `sync record_histogram(self, name: str, value: float, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:199` - Ignore histogram call.
- `class InMemoryTelemetrySink` at `afk/core/telemetry.py:221` - Test/debug telemetry sink that stores emitted measurements.
  Methods:
  - `sync record_event(self, event: TelemetryEvent)` at `afk/core/telemetry.py:230` - Store emitted event.
  - `sync start_span(self, name: str, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:239` - Start and store in-memory span.
  - `sync end_span(self, span: TelemetrySpan | None, *, status: str, error: str | None=None, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:254` - Close span and persist completed record.
  - `sync increment_counter(self, name: str, value: int=1, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:288` - Record counter data point.
  - `sync record_histogram(self, name: str, value: float, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:312` - Record histogram data point.
  - `sync events(self)` at `afk/core/telemetry.py:336` - Return captured events.
  - `sync spans(self)` at `afk/core/telemetry.py:345` - Return closed span records.
  - `sync counters(self)` at `afk/core/telemetry.py:354` - Return captured counter records.
  - `sync histograms(self)` at `afk/core/telemetry.py:363` - Return captured histogram records.
- `class OpenTelemetrySink` at `afk/core/telemetry.py:374` - OpenTelemetry sink using the global tracer/meter providers.
  Methods:
  - `sync _ensure_clients(self)` at `afk/core/telemetry.py:390` - Lazily initialize OpenTelemetry tracer and meter clients.
  - `sync _attr(self, value: dict[str, JSONValue] | None)` at `afk/core/telemetry.py:404` - Convert JSON attribute map into OpenTelemetry-compatible attributes.
  - `sync record_event(self, event: TelemetryEvent)` at `afk/core/telemetry.py:411` - Record event by incrementing OpenTelemetry counter.
  - `sync start_span(self, name: str, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:424` - Start OpenTelemetry span.
  - `sync end_span(self, span: TelemetrySpan | None, *, status: str, error: str | None=None, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:450` - End OpenTelemetry span with status metadata.
  - `sync increment_counter(self, name: str, value: int=1, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:487` - Increment or create OpenTelemetry counter.
  - `sync record_histogram(self, name: str, value: float, *, attributes: dict[str, JSONValue] | None=None)` at `afk/core/telemetry.py:512` - Record OpenTelemetry histogram measurement.
- `def now_ms` at `afk/core/telemetry.py:538` - Return current Unix epoch time in milliseconds.
- `def _to_attr(value: JSONValue)` at `afk/core/telemetry.py:548` - Convert JSON-safe values into OpenTelemetry attribute-compatible values.

## Package: `evals`

### Module `afk/evals/__init__.py`

Source: [afk/evals/__init__.py](../../src/afk/evals/__init__.py)

Module summary: Evaluation harness exports.

Symbols: none (re-export or namespace module).

### Module `afk/evals/harness.py`

Source: [afk/evals/harness.py](../../src/afk/evals/harness.py)

Top-level symbols:

- `class EvalScenario` at `afk/evals/harness.py:18`
- `class EvalResult` at `afk/evals/harness.py:27`
- `async def run_scenario(runner: Runner, scenario: EvalScenario)` at `afk/evals/harness.py:36`
- `def write_golden_trace(path: str | Path, events: list[AgentRunEvent])` at `afk/evals/harness.py:61`
- `def compare_event_types(expected: list[str], observed: list[str])` at `afk/evals/harness.py:75`
- `def run_scenarios(*, runner_factory: Callable[[], Runner], scenarios: list[EvalScenario])` at `afk/evals/harness.py:81`

## Package: `llms`

### Module `afk/llms/__init__.py`

Source: [afk/llms/__init__.py](../../src/afk/llms/__init__.py)

Symbols: none (re-export or namespace module).

### Module `afk/llms/clients/__init__.py`

Source: [afk/llms/clients/__init__.py](../../src/afk/llms/clients/__init__.py)

Module summary: LLM client package.

Symbols: none (re-export or namespace module).

### Module `afk/llms/clients/adapters/__init__.py`

Source: [afk/llms/clients/adapters/__init__.py](../../src/afk/llms/clients/adapters/__init__.py)

Module summary: Provider adapter implementations.

Symbols: none (re-export or namespace module).

### Module `afk/llms/clients/adapters/anthropic_agent.py`

Source: [afk/llms/clients/adapters/anthropic_agent.py](../../src/afk/llms/clients/adapters/anthropic_agent.py)

Top-level symbols:

- `class AnthropicAgentClient` at `afk/llms/clients/adapters/anthropic_agent.py:36` - Concrete adapter that integrates with `claude-agent-sdk`.
  Methods:
  - `sync __init__(self, **kwargs: Any)` at `afk/llms/clients/adapters/anthropic_agent.py:49`
  - `sync provider_id(self)` at `afk/llms/clients/adapters/anthropic_agent.py:54`
  - `sync capabilities(self)` at `afk/llms/clients/adapters/anthropic_agent.py:58`
  - `async _chat_core(self, req: LLMRequest, *, response_model: type[BaseModel] | None=None)` at `afk/llms/clients/adapters/anthropic_agent.py:61` - Execute one non-streaming SDK query and normalize final response.
  - `async _chat_stream_core(self, req: LLMRequest, *, response_model: type[BaseModel] | None=None)` at `afk/llms/clients/adapters/anthropic_agent.py:128` - Execute streaming SDK query and emit normalized stream events.
  - `async _embed_core(self, req: EmbeddingRequest)` at `afk/llms/clients/adapters/anthropic_agent.py:267` - Embeddings are intentionally unsupported for this adapter.
  - `async _interrupt_request(self, req: LLMRequest)` at `afk/llms/clients/adapters/anthropic_agent.py:275` - Interrupt active SDK streaming request for the given request id.
  - `sync _load_sdk_api(self)` at `afk/llms/clients/adapters/anthropic_agent.py:288` - Import and return required SDK symbols with a clear config error.
  - `sync _build_sdk_options(self, req: LLMRequest, *, options_type: Any, system_prompt: str | None, response_model: type[BaseModel] | None, include_partial_messages: bool)` at `afk/llms/clients/adapters/anthropic_agent.py:304` - Map normalized request fields into `ClaudeAgentOptions`.
  - `sync _allowed_tools(self, req: LLMRequest)` at `afk/llms/clients/adapters/anthropic_agent.py:402` - Derive `allowed_tools` from normalized tools/tool_choice fields.
  - `sync _build_prompt(self, messages: Iterable[Message])` at `afk/llms/clients/adapters/anthropic_agent.py:432` - Build SDK prompt/system_prompt from normalized messages.
  - `sync _content_to_text(self, content: Any)` at `afk/llms/clients/adapters/anthropic_agent.py:461` - Convert normalized message content into plain text transcript form.
  - `sync _iter_content_blocks(self, message: Any)` at `afk/llms/clients/adapters/anthropic_agent.py:506` - Return assistant content blocks when present.
  - `sync _extract_text_blocks(self, blocks: Iterable[Any])` at `afk/llms/clients/adapters/anthropic_agent.py:513` - Extract text chunks from SDK/content dict block variants.
  - `sync _extract_tool_blocks(self, blocks: Iterable[Any])` at `afk/llms/clients/adapters/anthropic_agent.py:529` - Extract normalized tool calls from SDK/content block variants.
  - `sync _stream_events_from_raw(self, event: dict[str, Any])` at `afk/llms/clients/adapters/anthropic_agent.py:558` - Map raw SDK partial stream events into normalized stream events.
  - `sync _extract_result_structured(self, message: Any)` at `afk/llms/clients/adapters/anthropic_agent.py:600` - Extract structured result payload from SDK `ResultMessage`.
  - `sync _extract_result_session_token(self, message: Any)` at `afk/llms/clients/adapters/anthropic_agent.py:607` - Extract session continuation token from SDK `ResultMessage`.
  - `sync _extract_result_checkpoint_token(self, message: Any)` at `afk/llms/clients/adapters/anthropic_agent.py:611` - Extract checkpoint token from SDK `ResultMessage` when available.
  - `sync _usage_from_obj(self, usage_obj: Any)` at `afk/llms/clients/adapters/anthropic_agent.py:615` - Normalize usage counters from SDK result usage object/dict.
  - `sync _dedupe_tool_calls(self, calls: list[ToolCall])` at `afk/llms/clients/adapters/anthropic_agent.py:641` - Deduplicate tool calls emitted across mixed SDK events/messages.
  - `sync _serialize_sdk_message(self, message: Any)` at `afk/llms/clients/adapters/anthropic_agent.py:656` - Serialize SDK message objects into raw debug-safe payloads.

### Module `afk/llms/clients/adapters/litellm.py`

Source: [afk/llms/clients/adapters/litellm.py](../../src/afk/llms/clients/adapters/litellm.py)

Top-level symbols:

- `class LiteLLMClient` at `afk/llms/clients/adapters/litellm.py:17` - Concrete adapter using `litellm` Responses API wrappers.
  Methods:
  - `sync provider_id(self)` at `afk/llms/clients/adapters/litellm.py:21`
  - `async _responses_create(self, payload: dict[str, Any])` at `afk/llms/clients/adapters/litellm.py:24` - Dispatch chat/stream payload to `litellm.aresponses`.
  - `async _embedding_create(self, payload: dict[str, Any])` at `afk/llms/clients/adapters/litellm.py:35` - Dispatch embedding payload to `litellm.aembedding`.
  - `sync _message_to_responses_input_items(self, message: Message)` at `afk/llms/clients/adapters/litellm.py:46` - Convert one normalized message into one LiteLLM/OpenAI-style input item.
  - `sync _structured_output_payload(self, response_model: type[BaseModel])` at `afk/llms/clients/adapters/litellm.py:139` - LiteLLM accepts a model type directly for structured text format.
  - `sync _with_transport_defaults(self, payload: dict[str, Any])` at `afk/llms/clients/adapters/litellm.py:146` - Apply config-level transport defaults without overriding explicit extras.

### Module `afk/llms/clients/adapters/openai.py`

Source: [afk/llms/clients/adapters/openai.py](../../src/afk/llms/clients/adapters/openai.py)

Top-level symbols:

- `class OpenAIClient` at `afk/llms/clients/adapters/openai.py:17` - Concrete adapter using `openai.AsyncOpenAI` Responses API.
  Methods:
  - `sync provider_id(self)` at `afk/llms/clients/adapters/openai.py:30`
  - `sync _provider_supported_thinking_efforts(self)` at `afk/llms/clients/adapters/openai.py:33` - OpenAI Responses API supports official effort labels.
  - `sync _provider_default_thinking_effort(self)` at `afk/llms/clients/adapters/openai.py:37` - Default effort when `thinking=True` and no explicit effort is set.
  - `async _responses_create(self, payload: dict[str, Any])` at `afk/llms/clients/adapters/openai.py:41` - Dispatch chat/stream payload to OpenAI Responses API.
  - `async _embedding_create(self, payload: dict[str, Any])` at `afk/llms/clients/adapters/openai.py:47` - Dispatch embedding payload to OpenAI embeddings API.
  - `sync _message_to_responses_input_items(self, message: Message)` at `afk/llms/clients/adapters/openai.py:53` - Convert one normalized message into OpenAI Responses input items.
  - `sync _structured_output_payload(self, response_model: type[BaseModel])` at `afk/llms/clients/adapters/openai.py:173` - OpenAI structured output payload for strict JSON schema mode.
  - `sync _build_client(self)` at `afk/llms/clients/adapters/openai.py:189` - Construct AsyncOpenAI client from shared config.
  - `sync _with_transport_headers(self, payload: dict[str, Any])` at `afk/llms/clients/adapters/openai.py:206` - Map AFK transport keys into OpenAI request options/headers.

### Module `afk/llms/clients/base/__init__.py`

Source: [afk/llms/clients/base/__init__.py](../../src/afk/llms/clients/base/__init__.py)

Module summary: Adapter base classes.

Symbols: none (re-export or namespace module).

### Module `afk/llms/clients/base/responses.py`

Source: [afk/llms/clients/base/responses.py](../../src/afk/llms/clients/base/responses.py)

Top-level symbols:

- `class ResponsesClientBase` at `afk/llms/clients/base/responses.py:43` - Provider-agnostic base for Responses-compatible clients.
  Methods:
  - `sync capabilities(self)` at `afk/llms/clients/base/responses.py:56` - Responses adapters expose chat/stream/tool/structured/embed.
  - `async _chat_core(self, req: LLMRequest, *, response_model: type[BaseModel] | None=None)` at `afk/llms/clients/base/responses.py:60` - Execute non-streaming call using provider transport hook.
  - `async _chat_stream_core(self, req: LLMRequest, *, response_model: type[BaseModel] | None=None)` at `afk/llms/clients/base/responses.py:75` - Execute streaming call and normalize provider stream events.
  - `async _embed_core(self, req: EmbeddingRequest)` at `afk/llms/clients/base/responses.py:161` - Execute embedding call using provider transport hook.
  - `sync _build_responses_payload(self, req: LLMRequest, *, response_model: type[BaseModel] | None, stream: bool)` at `afk/llms/clients/base/responses.py:192` - Map normalized `LLMRequest` into a Responses API payload.
  - `sync _messages_to_responses_input(self, messages: list[Message])` at `afk/llms/clients/base/responses.py:252` - Convert normalized messages to Responses API input items.
  - `sync _tool_to_responses_tool(self, tool: dict[str, Any])` at `afk/llms/clients/base/responses.py:259` - Convert normalized function tool definition to Responses tool schema.
  - `sync _tool_choice_to_responses_tool_choice(self, tool_choice: Any)` at `afk/llms/clients/base/responses.py:280` - Convert normalized tool-choice into Responses `tool_choice` shape.
  - `sync _normalize_responses_response(self, raw: Any)` at `afk/llms/clients/base/responses.py:304` - Normalize raw Responses payload into `LLMResponse`.
  - `sync _extract_text_from_responses_output(self, output_items: list[Any])` at `afk/llms/clients/base/responses.py:365` - Extract assistant text from Responses output message items.
  - `sync _extract_structured_from_responses_output(self, output_items: list[Any])` at `afk/llms/clients/base/responses.py:391` - Extract provider-native parsed JSON payload when available.
  - `sync _extract_tool_calls_from_responses_output(self, output_items: list[Any])` at `afk/llms/clients/base/responses.py:417` - Extract normalized function tool calls from Responses output items.
  - `sync _update_stream_tool_buffer_from_item(self, *, output_index: int, item: dict[str, Any], tool_buffers: dict[int, dict[str, Any]])` at `afk/llms/clients/base/responses.py:446` - Update stream tool-call buffer from `response.output_item.*` event.
  - `sync _stream_provider_label(self)` at `afk/llms/clients/base/responses.py:476` - Raw-provider label attached to synthesized stream fallback payloads.
  - `async _responses_create(self, payload: dict[str, Any])` at `afk/llms/clients/base/responses.py:481` - Provider transport hook for Responses API calls.
  - `async _embedding_create(self, payload: dict[str, Any])` at `afk/llms/clients/base/responses.py:485` - Provider transport hook for embedding API calls.
  - `sync _message_to_responses_input_items(self, message: Message)` at `afk/llms/clients/base/responses.py:489` - Provider-specific mapping from one normalized message to input items.
  - `sync _structured_output_payload(self, response_model: type[BaseModel])` at `afk/llms/clients/base/responses.py:493` - Provider-specific structured-output payload fragment.

### Module `afk/llms/clients/shared/__init__.py`

Source: [afk/llms/clients/shared/__init__.py](../../src/afk/llms/clients/shared/__init__.py)

Module summary: Shared client helper utilities.

Symbols: none (re-export or namespace module).

### Module `afk/llms/clients/shared/normalization.py`

Source: [afk/llms/clients/shared/normalization.py](../../src/afk/llms/clients/shared/normalization.py)

Top-level symbols:

- `def to_plain_dict(value: Any)` at `afk/llms/clients/shared/normalization.py:14` - Best-effort conversion of SDK/provider objects into plain dictionaries.
- `def to_jsonable(value: Any)` at `afk/llms/clients/shared/normalization.py:54` - Recursively coerce values into JSON-serializable primitives/containers.
- `def extract_text_from_content(content: Any)` at `afk/llms/clients/shared/normalization.py:78` - Extract plain text from common OpenAI/LiteLLM content shapes.
- `def extract_usage(raw_dict: dict[str, Any])` at `afk/llms/clients/shared/normalization.py:112` - Normalize usage token counters from provider payloads.
- `def extract_tool_calls(raw_tool_calls: Any)` at `afk/llms/clients/shared/normalization.py:134` - Extract normalized tool calls from chat completion payloads.
- `def finalize_stream_tool_calls(tool_buffers: dict[int, dict[str, Any]])` at `afk/llms/clients/shared/normalization.py:165` - Build final normalized tool calls from accumulated stream deltas.
- `def get_attr(obj: Any, name: str)` at `afk/llms/clients/shared/normalization.py:182` - Safe getattr helper.
- `def get_attr_str(obj: Any, name: str)` at `afk/llms/clients/shared/normalization.py:187` - Safe getattr helper returning strings only.

### Module `afk/llms/config.py`

Source: [afk/llms/config.py](../../src/afk/llms/config.py)

Top-level symbols:

- `class LLMConfig` at `afk/llms/config.py:14`
  Methods:
  - `sync from_env()` at `afk/llms/config.py:34`

### Module `afk/llms/errors.py`

Source: [afk/llms/errors.py](../../src/afk/llms/errors.py)

Top-level symbols:

- `class LLMError` at `afk/llms/errors.py:12` - Base exception for all AFK LLM-related errors.
- `class LLMTimeoutError` at `afk/llms/errors.py:18`
- `class LLMRetryableError` at `afk/llms/errors.py:22` - Transient failures: reate limits, timeouts, provider issues, etc.
- `class LLMInvalidResponseError` at `afk/llms/errors.py:31` - The LLM returned a response that we couldn't parse or validate.
- `class LLMConfigurationError` at `afk/llms/errors.py:40`
- `class LLMCapabilityError` at `afk/llms/errors.py:44` - Raised when the selected provider adapter does not support a requested
- `class LLMCancelledError` at `afk/llms/errors.py:53` - Raised when an in-flight streaming request is cancelled by caller.
- `class LLMInterruptedError` at `afk/llms/errors.py:59` - Raised when an in-flight request is interrupted by provider/user action.
- `class LLMSessionError` at `afk/llms/errors.py:65` - Raised for invalid session lifecycle operations.
- `class LLMSessionPausedError` at `afk/llms/errors.py:71` - Raised when a session call is attempted while the session is paused.

### Module `afk/llms/factory.py`

Source: [afk/llms/factory.py](../../src/afk/llms/factory.py)

Top-level symbols:

- `def register_llm_adapter(name: str, factory: AdapterFactory, *, overwrite: bool=False)` at `afk/llms/factory.py:24` - Register a custom adapter factory by name.
- `def available_llm_adapters` at `afk/llms/factory.py:41` - Return built-in and runtime-registered adapter names.
- `def create_llm(adapter: str, *, config: LLMConfig | None=None, middlewares: MiddlewareStack | None=None, thinking_effort_aliases: Mapping[str, str] | None=None, supported_thinking_efforts: set[str] | None=None, default_thinking_effort: str | None=None, observers: list[LLMObserver] | None=None)` at `afk/llms/factory.py:46` - Create an LLM client instance for a specific adapter key.
- `def create_llm_from_env(*, config: LLMConfig | None=None, middlewares: MiddlewareStack | None=None, thinking_effort_aliases: Mapping[str, str] | None=None, supported_thinking_efforts: set[str] | None=None, default_thinking_effort: str | None=None, observers: list[LLMObserver] | None=None)` at `afk/llms/factory.py:90` - Create an LLM client using `AFK_LLM_ADAPTER` (defaults to `litellm`).
- `def _builtin_factory(adapter: str, *, thinking_effort_aliases: Mapping[str, str] | None, supported_thinking_efforts: set[str] | None, default_thinking_effort: str | None, observers: list[LLMObserver] | None)` at `afk/llms/factory.py:112` - Resolve built-in adapter factories lazily to avoid hard imports.

### Module `afk/llms/llm.py`

Source: [afk/llms/llm.py](../../src/afk/llms/llm.py)

Top-level symbols:

- `class _QueuedStreamHandle` at `afk/llms/llm.py:59` - Default stream handle with local cancel semantics and optional interrupt.
  Methods:
  - `sync __init__(self, *, source: AsyncIterator[LLMStreamEvent], request_id: str, provider_id: str, model: str | None, emit_event: Callable[..., Awaitable[None]], interrupt_callback: Callable[[], Awaitable[None]] | None)` at `afk/llms/llm.py:62`
  - `sync _ensure_started(self)` at `afk/llms/llm.py:88`
  - `async _pump(self)` at `afk/llms/llm.py:92`
  - `async _iter_events(self)` at `afk/llms/llm.py:137`
  - `sync events(self)` at `afk/llms/llm.py:148`
  - `async cancel(self)` at `afk/llms/llm.py:156`
  - `async interrupt(self)` at `afk/llms/llm.py:170`
  - `async await_result(self)` at `afk/llms/llm.py:193`
- `class _TokenSessionHandle` at `afk/llms/llm.py:203` - Token-only session continuity handle used by adapters with session support.
  Methods:
  - `sync __init__(self, *, llm: LLM, session_token: str | None=None, checkpoint_token: str | None=None)` at `afk/llms/llm.py:206`
  - `sync _ensure_active(self)` at `afk/llms/llm.py:221`
  - `async _capture_stream_result(self, handle: LLMStreamHandle)` at `afk/llms/llm.py:227`
  - `async chat(self, req: LLMRequest, *, response_model: type[BaseModel] | None=None)` at `afk/llms/llm.py:240`
  - `async stream(self, req: LLMRequest, *, response_model: type[BaseModel] | None=None)` at `afk/llms/llm.py:259`
  - `async pause(self)` at `afk/llms/llm.py:276`
  - `async resume(self, session_token: str | None=None)` at `afk/llms/llm.py:281`
  - `async interrupt(self)` at `afk/llms/llm.py:290`
  - `async close(self)` at `afk/llms/llm.py:297`
  - `async snapshot(self)` at `afk/llms/llm.py:309`
- `class LLM` at `afk/llms/llm.py:318` - Base class for provider-agnostic LLM interactions.
  Methods:
  - `sync __init__(self, *, config: LLMConfig | None=None, middlewares: MiddlewareStack | None=None, thinking_effort_aliases: Mapping[str, str] | None=None, supported_thinking_efforts: set[str] | None=None, default_thinking_effort: str | None=None, observers: list[LLMObserver] | None=None)` at `afk/llms/llm.py:329` - Create a base LLM client.
  - `sync provider_id(self)` at `afk/llms/llm.py:363` - Stable provider id (e.g. 'litellm', 'anthropic_agent').
  - `sync capabilities(self)` at `afk/llms/llm.py:368` - Capability flags for the concrete adapter.
  - `sync from_env(cls, *, middlewares: MiddlewareStack | None=None, thinking_effort_aliases: Mapping[str, str] | None=None, supported_thinking_efforts: set[str] | None=None, default_thinking_effort: str | None=None, observers: list[LLMObserver] | None=None)` at `afk/llms/llm.py:372` - Build an LLM client from environment configuration.
  - `sync _provider_thinking_effort_aliases(self)` at `afk/llms/llm.py:407` - Provider-default aliases for request thinking effort labels.
  - `sync _provider_supported_thinking_efforts(self)` at `afk/llms/llm.py:416` - Provider-default allowed thinking effort labels.
  - `sync _provider_default_thinking_effort(self)` at `afk/llms/llm.py:424` - Provider-default effort used when `thinking=True` and no explicit effort
  - `sync thinking_effort_aliases(self)` at `afk/llms/llm.py:431` - Effective alias map after combining provider defaults and instance overrides.
  - `sync supported_thinking_efforts(self)` at `afk/llms/llm.py:437` - Effective allowed effort labels.
  - `sync default_thinking_effort(self)` at `afk/llms/llm.py:445` - Effective default effort when `thinking=True` and no effort is provided.
  - `sync normalize_thinking_effort(self, effort: str | None)` at `afk/llms/llm.py:451` - Normalize and validate a thinking effort label.
  - `sync resolve_thinking(self, req: LLMRequest)` at `afk/llms/llm.py:480` - Resolve request thinking controls into a normalized provider-agnostic config.
  - `async chat(self, req: LLMRequest, *, response_model: type[ModelT] | None=None)` at `afk/llms/llm.py:492` - Execute a non-streaming chat completion.
  - `sync chat_sync(self, req: LLMRequest, *, response_model: type[ModelT] | None=None)` at `afk/llms/llm.py:541` - Synchronous wrapper around `chat`.
  - `async chat_stream(self, req: LLMRequest, *, response_model: type[ModelT] | None=None)` at `afk/llms/llm.py:550` - Execute a streaming chat completion.
  - `async chat_stream_handle(self, req: LLMRequest, *, response_model: type[ModelT] | None=None)` at `afk/llms/llm.py:599` - Execute a streaming chat call and return a control handle.
  - `async embed(self, req: EmbeddingRequest)` at `afk/llms/llm.py:631` - Generate embeddings for a batch of input strings.
  - `sync embed_sync(self, req: EmbeddingRequest)` at `afk/llms/llm.py:662` - Synchronous wrapper around `embed`.
  - `sync start_session(self, *, session_token: str | None=None, checkpoint_token: str | None=None)` at `afk/llms/llm.py:666` - Start a provider session handle for continuity/control primitives.
  - `async _interrupt_request(self, req: LLMRequest)` at `afk/llms/llm.py:685` - Provider-specific interrupt hook used by `chat_stream_handle`.
  - `async _chat_core_with_safety(self, req: LLMRequest, *, response_model: type[ModelT] | None)` at `afk/llms/llm.py:696` - Run provider chat call under timeout/retry and structured checks.
  - `sync _chat_stream_with_safety(self, req: LLMRequest, *, response_model: type[ModelT] | None)` at `afk/llms/llm.py:730` - Run provider streaming call with retry setup and completion validation.
  - `async _embed_core_with_safety(self, req: EmbeddingRequest, *, request_id: str)` at `afk/llms/llm.py:813` - Run provider embedding call under timeout/retry policies.
  - `async _ensure_structured_response(self, req: LLMRequest, initial_response: LLMResponse, *, response_model: type[ModelT])` at `afk/llms/llm.py:837` - Ensure response matches a structured schema.
  - `sync _validate_structured_payload(self, response: LLMResponse, response_model: type[ModelT])` at `afk/llms/llm.py:881` - Validate structured payload or parse validated JSON from response text.
  - `sync _make_repair_request(self, req: LLMRequest, repair_prompt: str)` at `afk/llms/llm.py:914` - Create a follow-up repair request for schema correction retries.
  - `sync _ensure_capability(self, capability: str, enabled: bool)` at `afk/llms/llm.py:923` - Raise a capability error when an adapter lacks a required feature.
  - `sync _validate_chat_request(self, req: LLMRequest)` at `afk/llms/llm.py:931` - Validate chat request structure and enforce global input limits.
  - `sync _validate_embedding_request(self, req: EmbeddingRequest)` at `afk/llms/llm.py:1009` - Validate embedding request structure and enforce global input limits.
  - `sync _validate_message(self, message: Message, idx: int)` at `afk/llms/llm.py:1038` - Validate one normalized message and its content parts.
  - `sync _validate_tool_definition(self, tool: dict[str, Any], idx: int)` at `afk/llms/llm.py:1102` - Validate one provider-neutral tool definition.
  - `sync _validate_thinking_effort_aliases(self, aliases: Mapping[str, str] | None)` at `afk/llms/llm.py:1123` - Validate and normalize instance-level effort alias mapping.
  - `sync _validate_supported_thinking_efforts(self, supported: set[str] | None)` at `afk/llms/llm.py:1146` - Validate and normalize instance-level allowed effort labels.
  - `sync _validate_optional_thinking_effort_value(self, value: str | None, *, field_name: str)` at `afk/llms/llm.py:1165` - Validate a single optional thinking effort label.
  - `sync _validate_optional_nonempty(self, value: str | None, field_name: str)` at `afk/llms/llm.py:1183` - Validate optional token/id fields as non-empty strings when present.
  - `sync _message_char_count(self, message: Message)` at `afk/llms/llm.py:1190` - Best-effort character count used for input size limiting.
  - `sync _resolve_embedding_model(self, req: EmbeddingRequest)` at `afk/llms/llm.py:1211` - Resolve embedding model from request first, then config fallback.
  - `sync _ensure_request_id(self, req: LLMRequest)` at `afk/llms/llm.py:1229` - Ensure every request has a correlation id.
  - `sync _new_request_id(self)` at `afk/llms/llm.py:1236` - Generate a new opaque correlation id.
  - `sync _apply_response_context(self, req: LLMRequest, response: LLMResponse)` at `afk/llms/llm.py:1240` - Ensure response carries normalized request/session/checkpoint context.
  - `sync _can_retry_request(self, req: LLMRequest, safe_without_idempotency: bool)` at `afk/llms/llm.py:1251` - Decide retry eligibility for one request path.
  - `async _call_with_retries(self, fn: Callable[[], Awaitable[ReturnT]], *, request_id: str, model: str | None, max_retries: int | None=None)` at `afk/llms/llm.py:1262` - Execute a callable with retry-on-transient-error semantics.
  - `sync _classify_error(self, e: Exception)` at `afk/llms/llm.py:1331` - Map arbitrary exceptions into retryable vs non-retryable LLM errors.
  - `async _emit_lifecycle_event(self, *, event_type: str, request_id: str, model: str | None, attempt: int | None=None, latency_ms: float | None=None, usage: Usage | None=None, error: Exception | None=None)` at `afk/llms/llm.py:1429` - Emit one lifecycle event to observers, swallowing observer failures.
  - `sync _extract_usage_snapshot(self, result: Any)` at `afk/llms/llm.py:1464` - Extract usage counters from normalized return values when available.
  - `async _chat_core(self, req: LLMRequest, *, response_model: type[BaseModel] | None=None)` at `afk/llms/llm.py:1471` - Provider-specific chat implementation.
  - `async _chat_stream_core(self, req: LLMRequest, *, response_model: type[BaseModel] | None=None)` at `afk/llms/llm.py:1485` - Provider-specific streaming chat implementation.
  - `async _embed_core(self, req: EmbeddingRequest)` at `afk/llms/llm.py:1494` - Provider-specific embedding implementation.

### Module `afk/llms/middleware.py`

Source: [afk/llms/middleware.py](../../src/afk/llms/middleware.py)

Top-level symbols:

- `class LLMChatMiddleware` at `afk/llms/middleware.py:28`
  Methods:
  - `async __call__(self, call_next: LLMChatNext, req: LLMRequest)` at `afk/llms/middleware.py:29`
- `class LLMEmbedMiddleware` at `afk/llms/middleware.py:34`
  Methods:
  - `async __call__(self, call_next: LLMEmbedNext, req: EmbeddingRequest)` at `afk/llms/middleware.py:35`
- `class LLMStreamMiddleware` at `afk/llms/middleware.py:40`
  Methods:
  - `sync __call__(self, call_next: LLMChatStreamNext, req: LLMRequest)` at `afk/llms/middleware.py:41`
- `class MiddlewareStack` at `afk/llms/middleware.py:47`
  Methods:
  - `sync __init__(self, chat: list[LLMChatMiddleware] | None=None, embed: list[LLMEmbedMiddleware] | None=None, stream: list[LLMStreamMiddleware] | None=None)` at `afk/llms/middleware.py:52`

### Module `afk/llms/observability.py`

Source: [afk/llms/observability.py](../../src/afk/llms/observability.py)

Top-level symbols:

- `class LLMLifecycleEvent` at `afk/llms/observability.py:25` - One normalized lifecycle event emitted by the base LLM runtime.
- `class LLMObserver` at `afk/llms/observability.py:43` - Observer callback protocol used by the base LLM runtime.
  Methods:
  - `sync __call__(self, event: LLMLifecycleEvent)` at `afk/llms/observability.py:46`

### Module `afk/llms/structured.py`

Source: [afk/llms/structured.py](../../src/afk/llms/structured.py)

Top-level symbols:

- `def json_system_prompt(schema: type[T])` at `afk/llms/structured.py:21`
- `def parse_and_validate_json(text: str, schema: type[T])` at `afk/llms/structured.py:35`
- `def make_repair_prompt(invalid_response: str, schema: type[T])` at `afk/llms/structured.py:50`

### Module `afk/llms/types.py`

Source: [afk/llms/types.py](../../src/afk/llms/types.py)

Top-level symbols:

- `class TextContentPart` at `afk/llms/types.py:26`
- `class ImageURLRef` at `afk/llms/types.py:31`
- `class ImageURLContentPart` at `afk/llms/types.py:35`
- `class ToolUseContentPart` at `afk/llms/types.py:40`
- `class ToolResultContentPart` at `afk/llms/types.py:47`
- `class ToolFunctionSpec` at `afk/llms/types.py:63`
- `class ToolDefinition` at `afk/llms/types.py:69`
- `class ToolChoiceFunction` at `afk/llms/types.py:74`
- `class ToolChoiceNamed` at `afk/llms/types.py:78`
- `class Message` at `afk/llms/types.py:90`
- `class Usage` at `afk/llms/types.py:97`
- `class ToolCall` at `afk/llms/types.py:104` - Data-only representation of a model-returned tool call.
- `class LLMResponse` at `afk/llms/types.py:116`
- `class EmbeddingResponse` at `afk/llms/types.py:131`
- `class LLMRequest` at `afk/llms/types.py:138` - Canonical request type used by middleware, the client and agents.
- `class EmbeddingRequest` at `afk/llms/types.py:164`
- `class LLMCapabilities` at `afk/llms/types.py:173`
- `class ThinkingConfig` at `afk/llms/types.py:186` - Normalized thinking controls resolved by the base `LLM`.
- `class StreamMessageStartEvent` at `afk/llms/types.py:200`
- `class StreamTextDeltaEvent` at `afk/llms/types.py:206`
- `class StreamToolCallDeltaEvent` at `afk/llms/types.py:212`
- `class StreamMessageStopEvent` at `afk/llms/types.py:221`
- `class StreamErrorEvent` at `afk/llms/types.py:227`
- `class StreamCompletedEvent` at `afk/llms/types.py:233`
- `class LLMSessionSnapshot` at `afk/llms/types.py:249`
- `class LLMStreamHandle` at `afk/llms/types.py:256` - Control handle for a streaming chat request.
  Methods:
  - `sync events(self)` at `afk/llms/types.py:266`
  - `async cancel(self)` at `afk/llms/types.py:269`
  - `async interrupt(self)` at `afk/llms/types.py:272`
  - `async await_result(self)` at `afk/llms/types.py:275`
- `class LLMSessionHandle` at `afk/llms/types.py:279` - Provider/session continuity handle.
  Methods:
  - `async chat(self, req: LLMRequest, *, response_model: type['BaseModel'] | None=None)` at `afk/llms/types.py:287`
  - `async stream(self, req: LLMRequest, *, response_model: type['BaseModel'] | None=None)` at `afk/llms/types.py:295`
  - `async pause(self)` at `afk/llms/types.py:303`
  - `async resume(self, session_token: str | None=None)` at `afk/llms/types.py:306`
  - `async interrupt(self)` at `afk/llms/types.py:309`
  - `async close(self)` at `afk/llms/types.py:312`
  - `async snapshot(self)` at `afk/llms/types.py:315`

### Module `afk/llms/utils.py`

Source: [afk/llms/utils.py](../../src/afk/llms/utils.py)

Top-level symbols:

- `def clamp_str(s: str, max_chars: int)` at `afk/llms/utils.py:16`
- `def safe_json_loads(s: str)` at `afk/llms/utils.py:22`
- `def _strip_fenced_code_block(text: str)` at `afk/llms/utils.py:30` - If `text` starts with a fenced code block (``` or ~~~), return the content inside.
- `def extract_json_object(text: str)` at `afk/llms/utils.py:82` - Best-effort extraction of the first JSON object/array from a larger text blob.
- `def backoff_delay(attempt: int, base_s: float, jitter_s: float)` at `afk/llms/utils.py:182` - Exponential backoff with jitter.
- `def run_sync(coro)` at `afk/llms/utils.py:192` - Run an async coroutine from sync context.

## Package: `memory`

### Module `afk/memory/__init__.py`

Source: [afk/memory/__init__.py](../../src/afk/memory/__init__.py)

Top-level symbols:

- `def __getattr__(name: str)` at `afk/memory/__init__.py:30`

### Module `afk/memory/factory.py`

Source: [afk/memory/factory.py](../../src/afk/memory/factory.py)

Top-level symbols:

- `def _env_bool(name: str, default: bool=False)` at `afk/memory/factory.py:18` - Parse a boolean environment variable with common truthy values.
- `def create_memory_store_from_env` at `afk/memory/factory.py:26` - Create a memory store based on `AFK_MEMORY_BACKEND` and related environment settings.

### Module `afk/memory/lifecycle.py`

Source: [afk/memory/lifecycle.py](../../src/afk/memory/lifecycle.py)

Top-level symbols:

- `class RetentionPolicy` at `afk/memory/lifecycle.py:16`
- `class StateRetentionPolicy` at `afk/memory/lifecycle.py:23`
- `class MemoryCompactionResult` at `afk/memory/lifecycle.py:44`
- `def apply_event_retention(events: list[MemoryEvent], *, policy: RetentionPolicy)` at `afk/memory/lifecycle.py:54`
- `def apply_state_retention(state: dict[str, JsonValue], *, policy: StateRetentionPolicy)` at `afk/memory/lifecycle.py:73`
- `async def compact_thread_memory(memory: MemoryStore, *, thread_id: str, event_policy: RetentionPolicy | None=None, state_policy: StateRetentionPolicy | None=None)` at `afk/memory/lifecycle.py:171`
- `def _event_ids(events: list[MemoryEvent])` at `afk/memory/lifecycle.py:213`
- `def _safe_int(value: Any)` at `afk/memory/lifecycle.py:217`
- `def _extract_timestamp_ms(value: JsonValue)` at `afk/memory/lifecycle.py:227`
- `def _parse_checkpoint_latest_key(key: str)` at `afk/memory/lifecycle.py:234`
- `def _parse_checkpoint_state_key(key: str)` at `afk/memory/lifecycle.py:242`
- `def _parse_effect_key(key: str)` at `afk/memory/lifecycle.py:256`

### Module `afk/memory/models.py`

Source: [afk/memory/models.py](../../src/afk/memory/models.py)

Top-level symbols:

- `class MemoryEvent` at `afk/memory/models.py:25` - Represents an event in short-term memory for a specific conversation thread.
- `class LongTermMemory` at `afk/memory/models.py:38` - Represents a durable memory record for retrieval and personalization.
- `def now_ms` at `afk/memory/models.py:52`
- `def new_id(prefix: str='mem')` at `afk/memory/models.py:56`
- `def json_dumps(obj: JsonValue | dict[str, Any] | list[Any] | Any)` at `afk/memory/models.py:60`
- `def json_loads(s: str)` at `afk/memory/models.py:64`

### Module `afk/memory/store/__init__.py`

Source: [afk/memory/store/__init__.py](../../src/afk/memory/store/__init__.py)

Top-level symbols:

- `def __getattr__(name: str)` at `afk/memory/store/__init__.py:18`

### Module `afk/memory/store/base.py`

Source: [afk/memory/store/base.py](../../src/afk/memory/store/base.py)

Top-level symbols:

- `class MemoryCapabilities` at `afk/memory/store/base.py:19` - Describes optional backend features.
- `class MemoryStore` at `afk/memory/store/base.py:28` - Base contract for all memory backends.
  Methods:
  - `sync __init__(self)` at `afk/memory/store/base.py:40`
  - `async setup(self)` at `afk/memory/store/base.py:43` - Initialize backend resources.
  - `async close(self)` at `afk/memory/store/base.py:47` - Release backend resources.
  - `async __aenter__(self)` at `afk/memory/store/base.py:51`
  - `async __aexit__(self, exc_type, exc, tb)` at `afk/memory/store/base.py:55`
  - `sync _ensure_setup(self)` at `afk/memory/store/base.py:58`
  - `async append_event(self, event: MemoryEvent)` at `afk/memory/store/base.py:65` - Append one event for a thread.
  - `async get_recent_events(self, thread_id: str, limit: int=50)` at `afk/memory/store/base.py:69` - Return recent events for a thread in chronological order.
  - `async get_events_since(self, thread_id: str, since_ms: int, limit: int=500)` at `afk/memory/store/base.py:75` - Return events newer than `since_ms` in chronological order.
  - `async put_state(self, thread_id: str, key: str, value: JsonValue)` at `afk/memory/store/base.py:81` - Set a state value for a thread-scoped key.
  - `async get_state(self, thread_id: str, key: str)` at `afk/memory/store/base.py:85` - Return state value for a thread-scoped key.
  - `async list_state(self, thread_id: str, prefix: str | None=None)` at `afk/memory/store/base.py:89` - List thread-scoped state, optionally by key prefix.
  - `async upsert_long_term_memory(self, memory: LongTermMemory, *, embedding: Optional[Sequence[float]]=None)` at `afk/memory/store/base.py:95` - Insert or update one long-term memory record.
  - `async delete_long_term_memory(self, user_id: Optional[str], memory_id: str)` at `afk/memory/store/base.py:104` - Delete one long-term memory record.
  - `async list_long_term_memories(self, user_id: Optional[str], *, scope: str | None=None, limit: int=100)` at `afk/memory/store/base.py:110` - List long-term memories for a user and optional scope.
  - `async search_long_term_memory_text(self, user_id: Optional[str], query: str, *, scope: str | None=None, limit: int=20)` at `afk/memory/store/base.py:120` - Text search over long-term memories.
  - `async search_long_term_memory_vector(self, user_id: Optional[str], query_embedding: Sequence[float], *, scope: str | None=None, limit: int=20, min_score: float | None=None)` at `afk/memory/store/base.py:131` - Vector similarity search over long-term memories.
  - `async delete_state(self, thread_id: str, key: str)` at `afk/memory/store/base.py:142` - Delete one thread-scoped state key.
  - `async replace_thread_events(self, thread_id: str, events: list[MemoryEvent])` at `afk/memory/store/base.py:154` - Replace all events for a thread with `events` (chronological order).

### Module `afk/memory/store/in_memory.py`

Source: [afk/memory/store/in_memory.py](../../src/afk/memory/store/in_memory.py)

Top-level symbols:

- `class InMemoryMemoryStore` at `afk/memory/store/in_memory.py:21` - Fast, process-local memory backend with full text/vector retrieval support.
  Methods:
  - `sync __init__(self)` at `afk/memory/store/in_memory.py:28`
  - `async append_event(self, event: MemoryEvent)` at `afk/memory/store/in_memory.py:36`
  - `async get_recent_events(self, thread_id: str, limit: int=50)` at `afk/memory/store/in_memory.py:41`
  - `async get_events_since(self, thread_id: str, since_ms: int, limit: int=500)` at `afk/memory/store/in_memory.py:48`
  - `async put_state(self, thread_id: str, key: str, value: JsonValue)` at `afk/memory/store/in_memory.py:60`
  - `async delete_state(self, thread_id: str, key: str)` at `afk/memory/store/in_memory.py:65`
  - `async get_state(self, thread_id: str, key: str)` at `afk/memory/store/in_memory.py:70`
  - `async list_state(self, thread_id: str, prefix: str | None=None)` at `afk/memory/store/in_memory.py:75`
  - `async replace_thread_events(self, thread_id: str, events: list[MemoryEvent])` at `afk/memory/store/in_memory.py:92`
  - `async upsert_long_term_memory(self, memory: LongTermMemory, *, embedding: Optional[Sequence[float]]=None)` at `afk/memory/store/in_memory.py:101`
  - `async delete_long_term_memory(self, user_id: str | None, memory_id: str)` at `afk/memory/store/in_memory.py:115`
  - `async list_long_term_memories(self, user_id: str | None, *, scope: str | None=None, limit: int=100)` at `afk/memory/store/in_memory.py:128`
  - `async search_long_term_memory_text(self, user_id: str | None, query: str, *, scope: str | None=None, limit: int=20)` at `afk/memory/store/in_memory.py:147`
  - `async search_long_term_memory_vector(self, user_id: str | None, query_embedding: Sequence[float], *, scope: str | None=None, limit: int=20, min_score: float | None=None)` at `afk/memory/store/in_memory.py:186`

### Module `afk/memory/store/postgres.py`

Source: [afk/memory/store/postgres.py](../../src/afk/memory/store/postgres.py)

Top-level symbols:

- `class PostgresMemoryStore` at `afk/memory/store/postgres.py:27` - Production-grade memory store with JSONB and pgvector similarity search.
  Methods:
  - `sync __init__(self, *, dsn: str, vector_dim: int, pool_min: int=1, pool_max: int=10, ssl: bool=False)` at `afk/memory/store/postgres.py:34`
  - `async setup(self)` at `afk/memory/store/postgres.py:51`
  - `async close(self)` at `afk/memory/store/postgres.py:61`
  - `sync _pool_required(self)` at `afk/memory/store/postgres.py:67`
  - `async _create_schema(self)` at `afk/memory/store/postgres.py:74`
  - `async append_event(self, event: MemoryEvent)` at `afk/memory/store/postgres.py:136`
  - `async get_recent_events(self, thread_id: str, limit: int=50)` at `afk/memory/store/postgres.py:154`
  - `async get_events_since(self, thread_id: str, since_ms: int, limit: int=500)` at `afk/memory/store/postgres.py:167`
  - `async put_state(self, thread_id: str, key: str, value: JsonValue)` at `afk/memory/store/postgres.py:181`
  - `async delete_state(self, thread_id: str, key: str)` at `afk/memory/store/postgres.py:199`
  - `async get_state(self, thread_id: str, key: str)` at `afk/memory/store/postgres.py:209`
  - `async list_state(self, thread_id: str, prefix: str | None=None)` at `afk/memory/store/postgres.py:222`
  - `async replace_thread_events(self, thread_id: str, events: list[MemoryEvent])` at `afk/memory/store/postgres.py:243`
  - `async upsert_long_term_memory(self, memory: LongTermMemory, *, embedding: Optional[Sequence[float]]=None)` at `afk/memory/store/postgres.py:271`
  - `async delete_long_term_memory(self, user_id: str | None, memory_id: str)` at `afk/memory/store/postgres.py:334`
  - `async list_long_term_memories(self, user_id: str | None, *, scope: str | None=None, limit: int=100)` at `afk/memory/store/postgres.py:346`
  - `async search_long_term_memory_text(self, user_id: str | None, query: str, *, scope: str | None=None, limit: int=20)` at `afk/memory/store/postgres.py:381`
  - `async search_long_term_memory_vector(self, user_id: str | None, query_embedding: Sequence[float], *, scope: str | None=None, limit: int=20, min_score: float | None=None)` at `afk/memory/store/postgres.py:434`
  - `sync _record_to_event(record: asyncpg.Record)` at `afk/memory/store/postgres.py:488`
  - `sync _record_to_memory(record: asyncpg.Record)` at `afk/memory/store/postgres.py:500`

### Module `afk/memory/store/redis.py`

Source: [afk/memory/store/redis.py](../../src/afk/memory/store/redis.py)

Top-level symbols:

- `class RedisMemoryStore` at `afk/memory/store/redis.py:28` - Redis-backed memory store using hashes for state/memories and lists for events.
  Methods:
  - `sync __init__(self, *, url: str, events_max_per_thread: int=2000)` at `afk/memory/store/redis.py:35`
  - `async setup(self)` at `afk/memory/store/redis.py:41`
  - `async close(self)` at `afk/memory/store/redis.py:46`
  - `sync _redis(self)` at `afk/memory/store/redis.py:52`
  - `sync _events_key(thread_id: str)` at `afk/memory/store/redis.py:60`
  - `sync _state_key(thread_id: str)` at `afk/memory/store/redis.py:64`
  - `sync _memory_hash_key(user_id: str | None)` at `afk/memory/store/redis.py:68`
  - `async append_event(self, event: MemoryEvent)` at `afk/memory/store/redis.py:72`
  - `async get_recent_events(self, thread_id: str, limit: int=50)` at `afk/memory/store/redis.py:92`
  - `async get_events_since(self, thread_id: str, since_ms: int, limit: int=500)` at `afk/memory/store/redis.py:103`
  - `async put_state(self, thread_id: str, key: str, value: JsonValue)` at `afk/memory/store/redis.py:113`
  - `async delete_state(self, thread_id: str, key: str)` at `afk/memory/store/redis.py:118`
  - `async get_state(self, thread_id: str, key: str)` at `afk/memory/store/redis.py:123`
  - `async list_state(self, thread_id: str, prefix: str | None=None)` at `afk/memory/store/redis.py:131`
  - `async replace_thread_events(self, thread_id: str, events: list[MemoryEvent])` at `afk/memory/store/redis.py:144`
  - `async upsert_long_term_memory(self, memory: LongTermMemory, *, embedding: Optional[Sequence[float]]=None)` at `afk/memory/store/redis.py:170`
  - `async delete_long_term_memory(self, user_id: str | None, memory_id: str)` at `afk/memory/store/redis.py:205`
  - `async list_long_term_memories(self, user_id: str | None, *, scope: str | None=None, limit: int=100)` at `afk/memory/store/redis.py:212`
  - `async search_long_term_memory_text(self, user_id: str | None, query: str, *, scope: str | None=None, limit: int=20)` at `afk/memory/store/redis.py:233`
  - `async search_long_term_memory_vector(self, user_id: str | None, query_embedding: Sequence[float], *, scope: str | None=None, limit: int=20, min_score: float | None=None)` at `afk/memory/store/redis.py:260`
  - `sync _deserialize_event(serialized_event: str)` at `afk/memory/store/redis.py:298`
  - `sync _payload_to_memory(payload: dict[str, object])` at `afk/memory/store/redis.py:313`

### Module `afk/memory/store/sqlite.py`

Source: [afk/memory/store/sqlite.py](../../src/afk/memory/store/sqlite.py)

Top-level symbols:

- `class SQLiteMemoryStore` at `afk/memory/store/sqlite.py:29` - Persistent local memory backend backed by SQLite.
  Methods:
  - `sync __init__(self, path: str='afk_memory.sqlite3')` at `afk/memory/store/sqlite.py:36`
  - `async setup(self)` at `afk/memory/store/sqlite.py:41`
  - `async close(self)` at `afk/memory/store/sqlite.py:51`
  - `sync _db(self)` at `afk/memory/store/sqlite.py:57`
  - `async _create_tables(self)` at `afk/memory/store/sqlite.py:64`
  - `sync _user_filter_sql(column_name: str='user_id')` at `afk/memory/store/sqlite.py:116`
  - `async append_event(self, event: MemoryEvent)` at `afk/memory/store/sqlite.py:119`
  - `async get_recent_events(self, thread_id: str, limit: int=50)` at `afk/memory/store/sqlite.py:139`
  - `async get_events_since(self, thread_id: str, since_ms: int, limit: int=500)` at `afk/memory/store/sqlite.py:152`
  - `async put_state(self, thread_id: str, key: str, value: JsonValue)` at `afk/memory/store/sqlite.py:164`
  - `async delete_state(self, thread_id: str, key: str)` at `afk/memory/store/sqlite.py:179`
  - `async get_state(self, thread_id: str, key: str)` at `afk/memory/store/sqlite.py:188`
  - `async list_state(self, thread_id: str, prefix: str | None=None)` at `afk/memory/store/sqlite.py:200`
  - `async replace_thread_events(self, thread_id: str, events: list[MemoryEvent])` at `afk/memory/store/sqlite.py:221`
  - `async upsert_long_term_memory(self, memory: LongTermMemory, *, embedding: Optional[Sequence[float]]=None)` at `afk/memory/store/sqlite.py:247`
  - `async delete_long_term_memory(self, user_id: str | None, memory_id: str)` at `afk/memory/store/sqlite.py:290`
  - `async list_long_term_memories(self, user_id: str | None, *, scope: str | None=None, limit: int=100)` at `afk/memory/store/sqlite.py:307`
  - `async search_long_term_memory_text(self, user_id: str | None, query: str, *, scope: str | None=None, limit: int=20)` at `afk/memory/store/sqlite.py:327`
  - `async search_long_term_memory_vector(self, user_id: str | None, query_embedding: Sequence[float], *, scope: str | None=None, limit: int=20, min_score: float | None=None)` at `afk/memory/store/sqlite.py:354`
  - `sync _row_to_event(row: aiosqlite.Row)` at `afk/memory/store/sqlite.py:398`
  - `sync _row_to_long_term_memory(row: aiosqlite.Row)` at `afk/memory/store/sqlite.py:410`

### Module `afk/memory/vector.py`

Source: [afk/memory/vector.py](../../src/afk/memory/vector.py)

Top-level symbols:

- `def cosine_similarity(a: Sequence[float], b: Sequence[float])` at `afk/memory/vector.py:16` - Compute cosine similarity and return 0.0 for zero-norm vectors.
- `def format_pgvector(vec: Sequence[float])` at `afk/memory/vector.py:34` - Format a Python vector as a pgvector literal string.

## Package: `root`

### Module `afk/__init__.py`

Source: [afk/__init__.py](../../src/afk/__init__.py)

Symbols: none (re-export or namespace module).

## Package: `tools`

### Module `afk/tools/__init__.py`

Source: [afk/tools/__init__.py](../../src/afk/tools/__init__.py)

Symbols: none (re-export or namespace module).

### Module `afk/tools/core/__init__.py`

Source: [afk/tools/core/__init__.py](../../src/afk/tools/core/__init__.py)

Symbols: none (re-export or namespace module).

### Module `afk/tools/core/base.py`

Source: [afk/tools/core/base.py](../../src/afk/tools/core/base.py)

Top-level symbols:

- `class ToolSpec` at `afk/tools/core/base.py:42` - Stable tool metadata used for registry listing + model-facing export.
- `class ToolContext` at `afk/tools/core/base.py:53` - Contextual information available to a tool during its execution.
- `class ToolResult` at `afk/tools/core/base.py:65` - Standardized result object returned by tools after execution.
- `def as_async(fn: ToolFn)` at `afk/tools/core/base.py:82` - Utility function to convert a synchronous function into an asynchronous one.
- `def _infer_call_style(fn: Callable[..., Any])` at `afk/tools/core/base.py:97` - Determine how to call a tool/hook based on the signature.
- `def _infer_middleware_style(fn: Callable[..., Any])` at `afk/tools/core/base.py:141` - Middleware signature inference.
- `class BaseTool` at `afk/tools/core/base.py:188` - Base class for defining a function-based tool/hook.
  Methods:
  - `sync __init__(self, *, spec: ToolSpec, fn: ToolFn, args_model: Type[ArgsT], default_timeout: Optional[float]=None, raise_on_error: bool=False)` at `afk/tools/core/base.py:196`
  - `sync validate(self, raw_args: Dict[str, Any])` at `afk/tools/core/base.py:214`
  - `async _invoke(self, args: ArgsT, ctx: ToolContext)` at `afk/tools/core/base.py:222`
  - `async call(self, raw_args: Dict[str, Any], *, ctx: Optional[ToolContext]=None, timeout: Optional[float]=None, tool_call_id: Optional[str]=None)` at `afk/tools/core/base.py:229`
- `class PreHook` at `afk/tools/core/base.py:298` - Prehooks are executed before the main tool execution.
- `class PostHook` at `afk/tools/core/base.py:306` - Post-hooks are executed after the main tool execution.
- `class Middleware` at `afk/tools/core/base.py:314` - Middleware wraps the tool execution.
  Methods:
  - `sync __init__(self, *, spec: ToolSpec, fn: ToolFn, default_timeout: Optional[float]=None)` at `afk/tools/core/base.py:327`
  - `async call(self, call_next: Callable[[ArgsT, ToolContext], Awaitable[ReturnT]], args: ArgsT, ctx: ToolContext, *, timeout: Optional[float]=None)` at `afk/tools/core/base.py:336`
- `class Tool` at `afk/tools/core/base.py:365` - Main Tool class supporting prehooks, posthooks, and middleware wrapping.
  Methods:
  - `sync __init__(self, *, spec: ToolSpec, fn: ToolFn, args_model: Type[ArgsT], default_timeout: Optional[float]=None, prehooks: Optional[List[PreHook[Any, Any]]]=None, posthooks: Optional[List[PostHook]]=None, middlewares: Optional[List[Middleware[ArgsT, ReturnT]]]=None, raise_on_error: bool=False)` at `afk/tools/core/base.py:370`
  - `async call(self, raw_args: Dict[str, Any], *, ctx: Optional[ToolContext]=None, timeout: Optional[float]=None, tool_call_id: Optional[str]=None)` at `afk/tools/core/base.py:393`

### Module `afk/tools/core/decorator.py`

Source: [afk/tools/core/decorator.py](../../src/afk/tools/core/decorator.py)

Top-level symbols:

- `def _default_description(fn: Callable[..., Any], fallback: str)` at `afk/tools/core/decorator.py:35`
- `def tool(*, args_model: Type[ArgsT], name: str | None=None, description: str | None=None, timeout: float | None=None, prehooks: Optional[list[PreHook[Any, Any]]]=None, posthooks: Optional[list[PostHook]]=None, middlewares: Optional[list[Middleware[Any, Any]]]=None, raise_on_error: bool=False)` at `afk/tools/core/decorator.py:41` - Create an AFK Tool from a sync/async function and a Pydantic v2 args model.
- `def prehook(*, args_model: Type[ArgsT], name: str | None=None, description: str | None=None, timeout: float | None=None, raise_on_error: bool=False)` at `afk/tools/core/decorator.py:91` - Create a PreHook from a sync/async function and a Pydantic v2 args model.
- `def posthook(*, args_model: Type[ArgsT], name: str | None=None, description: str | None=None, timeout: float | None=None, raise_on_error: bool=False)` at `afk/tools/core/decorator.py:123` - Create a PostHook from a sync/async function and a Pydantic v2 args model.
- `def middleware(*, name: str | None=None, description: str | None=None, timeout: float | None=None)` at `afk/tools/core/decorator.py:161` - Create a Middleware from a sync/async function (tool-level middleware).
- `def registry_middleware(*, name: str | None=None, description: str | None=None)` at `afk/tools/core/decorator.py:190` - Create a registry-level middleware wrapper.

### Module `afk/tools/core/errors.py`

Source: [afk/tools/core/errors.py](../../src/afk/tools/core/errors.py)

Module summary: MIT License

Top-level symbols:

- `class AFKToolError` at `afk/tools/core/errors.py:12` - Base exception for all AFK tool-related errors.
- `class ToolValidationError` at `afk/tools/core/errors.py:18`
- `class ToolAlreadyRegisteredError` at `afk/tools/core/errors.py:22`
- `class ToolValidationError` at `afk/tools/core/errors.py:26`
- `class ToolExecutionError` at `afk/tools/core/errors.py:30`
- `class ToolTimeoutError` at `afk/tools/core/errors.py:34`
- `class ToolPolicyError` at `afk/tools/core/errors.py:38`
- `class ToolNotFoundError` at `afk/tools/core/errors.py:42`
- `class ToolPermissionError` at `afk/tools/core/errors.py:46`

### Module `afk/tools/core/export.py`

Source: [afk/tools/core/export.py](../../src/afk/tools/core/export.py)

Top-level symbols:

- `def normalize_json_schema(schema: Dict[str, Any])` at `afk/tools/core/export.py:19` - Pydantic v2's model_json_schema() is generally usable as-is.
- `def toolspec_to_litellm_tool(spec: ToolSpec)` at `afk/tools/core/export.py:33` - Convert a ToolSpec into a LiteLLM tool definition.
- `def tool_to_litellm_tool(tool: Tool[Any, Any])` at `afk/tools/core/export.py:57`
- `def to_litellm_tools(tools: Iterable[Tool[Any, Any]])` at `afk/tools/core/export.py:61` - Export Tool objects to a list of LiteLLM tool definitions.
- `def to_litellm_tools_from_specs(specs: Iterable[ToolSpec])` at `afk/tools/core/export.py:69` - Export ToolSpec objects to a list of LiteLLM tool definitions.
- `def export_tools(tools: Iterable[Tool[Any, Any]], *, format: str='litellm')` at `afk/tools/core/export.py:76` - Generic export entrypoint.

### Module `afk/tools/prebuilts/__init__.py`

Source: [afk/tools/prebuilts/__init__.py](../../src/afk/tools/prebuilts/__init__.py)

Module summary: Prebuilt tool factories.

Symbols: none (re-export or namespace module).

### Module `afk/tools/prebuilts/runtime.py`

Source: [afk/tools/prebuilts/runtime.py](../../src/afk/tools/prebuilts/runtime.py)

Top-level symbols:

- `class _ListDirectoryArgs` at `afk/tools/prebuilts/runtime.py:17`
- `class _ReadFileArgs` at `afk/tools/prebuilts/runtime.py:22`
- `def build_runtime_tools(*, root_dir: Path)` at `afk/tools/prebuilts/runtime.py:27`
- `def _ensure_inside(path: Path, root: Path)` at `afk/tools/prebuilts/runtime.py:71`

### Module `afk/tools/prebuilts/skills.py`

Source: [afk/tools/prebuilts/skills.py](../../src/afk/tools/prebuilts/skills.py)

Top-level symbols:

- `class _EmptyArgs` at `afk/tools/prebuilts/skills.py:19`
- `class _ReadSkillArgs` at `afk/tools/prebuilts/skills.py:23`
- `class _ReadSkillFileArgs` at `afk/tools/prebuilts/skills.py:27`
- `class _RunSkillCommandArgs` at `afk/tools/prebuilts/skills.py:33`
- `def build_skill_tools(*, skills: list[SkillRef], policy: SkillToolPolicy)` at `afk/tools/prebuilts/skills.py:40` - Build runtime-bound skill tools for one run.
- `def _is_command_allowed(command: str, allowlist: list[str])` at `afk/tools/prebuilts/skills.py:158`
- `def _ensure_inside(path: Path, root: Path)` at `afk/tools/prebuilts/skills.py:172`
- `def _is_inside(path: Path, root: Path)` at `afk/tools/prebuilts/skills.py:177`

### Module `afk/tools/registery.py`

Source: [afk/tools/registery.py](../../src/afk/tools/registery.py)

Top-level symbols:

- `class ToolCallRecord` at `afk/tools/registery.py:39`
- `def _infer_registry_middleware_style(fn: Callable[..., Any])` at `afk/tools/registery.py:57` - Supported registry middleware signatures (sync OR async):
- `class RegistryMiddleware` at `afk/tools/registery.py:97` - Wraps ALL tool calls executed through the registry.
  Methods:
  - `sync __init__(self, fn: RegistryMiddlewareFn, *, name: str | None=None)` at `afk/tools/registery.py:107`
  - `async __call__(self, call_next: RegistryCallNext, tool: Tool[Any, Any], raw_args: Dict[str, Any], ctx: ToolContext, timeout: float | None, tool_call_id: str | None)` at `afk/tools/registery.py:116`
- `class ToolRegistry` at `afk/tools/registery.py:174` - Stores tools by name and provides safe async execution with:
  Methods:
  - `sync __init__(self, *, max_concurrency: int=32, default_timeout: float | None=None, policy: ToolPolicy | None=None, enable_plugins: bool=False, plugin_entry_point_group: str='afk.tools', allow_overwrite_plugins: bool=False, middlewares: Optional[List[RegistryMiddleware | RegistryMiddlewareFn]]=None)` at `afk/tools/registery.py:185`
  - `sync register(self, tool: Tool[Any, Any], *, overwrite: bool=False)` at `afk/tools/registery.py:220`
  - `sync register_many(self, tools: Iterable[Tool[Any, Any]], *, overwrite: bool=False)` at `afk/tools/registery.py:226`
  - `sync unregister(self, name: str)` at `afk/tools/registery.py:232`
  - `sync get(self, name: str)` at `afk/tools/registery.py:235`
  - `sync list(self)` at `afk/tools/registery.py:241`
  - `sync names(self)` at `afk/tools/registery.py:244`
  - `sync has(self, name: str)` at `afk/tools/registery.py:247`
  - `sync load_plugins(self, *, entry_point_group: str='afk.tools', overwrite: bool=False)` at `afk/tools/registery.py:250` - Load Tool objects (or factories returning Tool) from Python entry points.
  - `sync add_middleware(self, mw: RegistryMiddleware | RegistryMiddlewareFn)` at `afk/tools/registery.py:289` - Add a registry-level middleware.
  - `sync set_middlewares(self, mws: List[RegistryMiddleware | RegistryMiddlewareFn])` at `afk/tools/registery.py:299`
  - `sync clear_middlewares(self)` at `afk/tools/registery.py:306`
  - `sync list_middlewares(self)` at `afk/tools/registery.py:309`
  - `async call(self, name: str, raw_args: Dict[str, Any], *, ctx: ToolContext | None=None, timeout: float | None=None, tool_call_id: str | None=None)` at `afk/tools/registery.py:316` - Execute a registered tool by name.
  - `async call_many(self, calls: Sequence[tuple[str, Dict[str, Any]]], *, ctx: ToolContext | None=None, timeout: float | None=None, tool_call_id_prefix: str | None=None, return_exceptions: bool=False)` at `afk/tools/registery.py:429` - Execute multiple tool calls concurrently (bounded by registry semaphore).
  - `sync recent_calls(self, limit: int=100)` at `afk/tools/registery.py:460`
  - `sync specs(self)` at `afk/tools/registery.py:467`
  - `sync to_openai_function_tools(self)` at `afk/tools/registery.py:470` - Export registry tools in OpenAI function-tool format:
  - `sync list_tool_summaries(self)` at `afk/tools/registery.py:492` - Lightweight listing for UIs / debugging.

### Module `afk/tools/security.py`

Source: [afk/tools/security.py](../../src/afk/tools/security.py)

Top-level symbols:

- `class SandboxProfile` at `afk/tools/security.py:16`
- `class SecretScopeProvider` at `afk/tools/security.py:28`
  Methods:
  - `sync resolve(self, *, tool_name: str, tool_args: dict[str, Any], run_context: dict[str, Any])` at `afk/tools/security.py:29`
- `class SandboxProfileProvider` at `afk/tools/security.py:39`
  Methods:
  - `sync resolve(self, *, tool_name: str, tool_args: dict[str, Any], run_context: dict[str, Any])` at `afk/tools/security.py:40`
- `def validate_tool_args_against_sandbox(*, tool_name: str, tool_args: dict[str, Any], profile: SandboxProfile, cwd: Path)` at `afk/tools/security.py:50`
- `def build_registry_sandbox_policy(*, profile: SandboxProfile, cwd: Path)` at `afk/tools/security.py:121` - Build a ToolRegistry policy hook enforcing the sandbox profile at call time.
- `def build_registry_output_limit_middleware(*, profile: SandboxProfile)` at `afk/tools/security.py:144` - Registry middleware to cap large tool outputs at execution boundary.
- `def resolve_sandbox_profile(*, tool_name: str, tool_args: dict[str, Any], run_context: dict[str, Any], default_profile: SandboxProfile | None, provider: SandboxProfileProvider | None)` at `afk/tools/security.py:173` - Resolve the effective sandbox profile for a single tool invocation.
- `def apply_tool_output_limits(result: ToolResult[Any], *, profile: SandboxProfile | None)` at `afk/tools/security.py:199` - Apply profile-level output truncation to a tool result.
- `def _looks_like_path_key(key: str)` at `afk/tools/security.py:223`
- `def _resolve_root(value: str, cwd: Path)` at `afk/tools/security.py:228`
- `def _resolve_path(value: str, cwd: Path)` at `afk/tools/security.py:235`
- `def _is_under(path: Path, root: Path)` at `afk/tools/security.py:242`
- `def _iter_leaf_values(payload: dict[str, Any])` at `afk/tools/security.py:250`
- `def _extract_command_parts(tool_args: dict[str, Any])` at `afk/tools/security.py:269`
- `def _is_command_allowed(command: str, allowlist: list[str])` at `afk/tools/security.py:280`
- `def _truncate_text(value: str, *, max_chars: int)` at `afk/tools/security.py:290`
- `def _truncate_json_like(value: Any, *, max_chars: int)` at `afk/tools/security.py:296`
