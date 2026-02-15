# Tested Behaviors and Edge Cases

This document maps the test suite to runtime behavior guarantees. It is intended as a practical specification for developers.

Source tests are under `tests/`.

## How To Use This Document

- Use this file as the "contract" for expected behavior.
- When changing runtime logic, update relevant tests first, then this map.
- For deep implementation details, cross-read:
  - [Architecture](./architecture.md)
  - [Tool Call Lifecycle](./tool-call-lifecycle.md)
  - [LLM Interaction Flow](./llm-interaction.md)

## Quick Coverage Summary

- agent runtime orchestration, failure policies, and resume flows
- policy engine matching and priority behavior
- version migration compatibility
- interaction provider behavior
- LLM adapters and structured output contracts
- memory stores and compaction lifecycle
- tool registry, middleware, and sandbox/security controls

## `tests/agents/test_agent_runtime.py`

Covered behaviors:

- `test_model_resolution_uses_litellm_for_ollama_prefix` (`tests/agents/test_agent_runtime.py:539`): model resolution uses litellm for ollama prefix
- `test_runner_executes_tool_batch_and_finishes` (`tests/agents/test_agent_runtime.py:545`): runner executes tool batch and finishes
- `test_run_handle_cancel_returns_none` (`tests/agents/test_agent_runtime.py:560`): run handle cancel returns none
- `test_missing_skill_raises_configuration_error` (`tests/agents/test_agent_runtime.py:573`): missing skill raises configuration error
- `test_skill_tools_are_auto_registered_and_used` (`tests/agents/test_agent_runtime.py:584`): skill tools are auto registered and used
- `test_policy_can_request_user_input_and_continue_tool_execution` (`tests/agents/test_agent_runtime.py:603`): policy can request user input and continue tool execution
- `test_run_handle_interrupt_raises_interrupted_error_and_emits_event` (`tests/agents/test_agent_runtime.py:630`): run handle interrupt raises interrupted error and emits event
- `test_runner_uses_model_resolver_and_fallback_chain` (`tests/agents/test_agent_runtime.py:654`): runner uses model resolver and fallback chain
- `test_cost_budget_is_enforced` (`tests/agents/test_agent_runtime.py:679`): cost budget is enforced
- `test_approval_denial_policy_fail_run` (`tests/agents/test_agent_runtime.py:690`): approval denial policy fail run
- `test_tool_failure_policy_fail_run` (`tests/agents/test_agent_runtime.py:707`): tool failure policy fail run
- `test_subagent_context_inheritance_and_isolation` (`tests/agents/test_agent_runtime.py:718`): subagent context inheritance and isolation
- `test_subagent_failure_policy_fail_run` (`tests/agents/test_agent_runtime.py:738`): subagent failure policy fail run
- `test_interaction_mode_requires_provider` (`tests/agents/test_agent_runtime.py:754`): interaction mode requires provider
- `test_memory_fallback_emits_warning` (`tests/agents/test_agent_runtime.py:759`): memory fallback emits warning
- `test_resume_completed_run_returns_terminal_result_without_reexecution` (`tests/agents/test_agent_runtime.py:782`): resume completed run returns terminal result without reexecution
- `test_compaction_preserves_resume_contract_for_completed_runs` (`tests/agents/test_agent_runtime.py:805`): compaction preserves resume contract for completed runs
- `test_resume_rejects_incompatible_checkpoint_schema` (`tests/agents/test_agent_runtime.py:836`): resume rejects incompatible checkpoint schema
- `test_effect_replay_skips_side_effect_tool_execution` (`tests/agents/test_agent_runtime.py:865`): effect replay skips side effect tool execution
- `test_effect_replay_input_hash_mismatch_raises` (`tests/agents/test_agent_runtime.py:914`): effect replay input hash mismatch raises
- `test_restart_from_checkpoint_replays_effect_without_duplicate_side_effect` (`tests/agents/test_agent_runtime.py:957`): restart from checkpoint replays effect without duplicate side effect
- `test_policy_engine_denies_tool_and_emits_audit_event` (`tests/agents/test_agent_runtime.py:1037`): policy engine denies tool and emits audit event
- `test_sandbox_profile_denies_disallowed_path_tool_call` (`tests/agents/test_agent_runtime.py:1074`): sandbox profile denies disallowed path tool call
- `test_secret_scope_provider_injects_tool_metadata_secrets` (`tests/agents/test_agent_runtime.py:1102`): secret scope provider injects tool metadata secrets
- `test_tool_output_sanitization_redacts_injection_tokens` (`tests/agents/test_agent_runtime.py:1118`): tool output sanitization redacts injection tokens
- `test_runner_emits_telemetry_events` (`tests/agents/test_agent_runtime.py:1135`): runner emits telemetry events
- `test_runner_emits_telemetry_spans_and_metrics` (`tests/agents/test_agent_runtime.py:1149`): runner emits telemetry spans and metrics
- `test_eval_harness_runs_scenario` (`tests/agents/test_agent_runtime.py:1169`): eval harness runs scenario
- `test_runtime_sandbox_policy_blocks_shell_operator_commands` (`tests/agents/test_agent_runtime.py:1181`): runtime sandbox policy blocks shell operator commands
- `test_per_tool_sandbox_provider_overrides_default_profile` (`tests/agents/test_agent_runtime.py:1208`): per tool sandbox provider overrides default profile
- `test_runtime_output_limit_middleware_truncates_tool_output` (`tests/agents/test_agent_runtime.py:1240`): runtime output limit middleware truncates tool output

## `tests/agents/test_policy_engine.py`

Covered behaviors:

- `test_policy_engine_priority_is_deterministic` (`tests/agents/test_policy_engine.py:22`): policy engine priority is deterministic
- `test_policy_engine_condition_matches_context_and_metadata` (`tests/agents/test_policy_engine.py:47`): policy engine condition matches context and metadata
- `test_infer_policy_subject_mapping` (`tests/agents/test_policy_engine.py:67`): infer policy subject mapping

## `tests/agents/test_versioning.py`

Covered behaviors:

- `test_check_event_schema_version_reports_compatibility` (`tests/agents/test_versioning.py:15`): check event schema version reports compatibility
- `test_check_checkpoint_schema_version_reports_compatibility` (`tests/agents/test_versioning.py:22`): check checkpoint schema version reports compatibility
- `test_migrate_event_record_from_legacy_aliases` (`tests/agents/test_versioning.py:29`): migrate event record from legacy aliases
- `test_migrate_checkpoint_record_from_legacy_aliases` (`tests/agents/test_versioning.py:43`): migrate checkpoint record from legacy aliases
- `test_migrate_checkpoint_record_rejects_unknown_schema` (`tests/agents/test_versioning.py:61`): migrate checkpoint record rejects unknown schema

## `tests/core/test_interaction_provider.py`

Covered behaviors:

- `test_headless_provider_uses_fallback_decisions` (`tests/core/test_interaction_provider.py:18`): headless provider uses fallback decisions
- `test_inmemory_provider_supports_deferred_resolution` (`tests/core/test_interaction_provider.py:44`): inmemory provider supports deferred resolution
- `test_provider_collects_notifications` (`tests/core/test_interaction_provider.py:63`): provider collects notifications

## `tests/core/test_runner_chaos.py`

Covered behaviors:

- `test_runner_handles_many_concurrent_runs` (`tests/core/test_runner_chaos.py:145`): runner handles many concurrent runs
- `test_runner_fallback_chain_recovers_from_chaos_failures` (`tests/core/test_runner_chaos.py:168`): runner fallback chain recovers from chaos failures
- `test_runner_survives_transient_memory_append_failures` (`tests/core/test_runner_chaos.py:190`): runner survives transient memory append failures
- `test_runner_timeout_burst_produces_budget_errors_without_deadlock` (`tests/core/test_runner_chaos.py:198`): runner timeout burst produces budget errors without deadlock

## `tests/evals/test_eval_golden_trace.py`

Covered behaviors:

- `test_eval_event_types_match_golden_trace` (`tests/evals/test_eval_golden_trace.py:45`): eval event types match golden trace

## `tests/llms/test_anthropic_agent_sdk_adapter.py`

Covered behaviors:

- `test_anthropic_adapter_chat_uses_claude_agent_sdk` (`tests/llms/test_anthropic_agent_sdk_adapter.py:124`): anthropic adapter chat uses claude agent sdk
- `test_anthropic_adapter_stream_maps_stream_events` (`tests/llms/test_anthropic_agent_sdk_adapter.py:149`): anthropic adapter stream maps stream events
- `test_anthropic_adapter_stop_is_unsupported` (`tests/llms/test_anthropic_agent_sdk_adapter.py:173`): anthropic adapter stop is unsupported

## `tests/llms/test_litellm_responses_adapter.py`

Covered behaviors:

- `test_litellm_chat_uses_responses_api` (`tests/llms/test_litellm_responses_adapter.py:147`): litellm chat uses responses api
- `test_litellm_stream_uses_responses_events` (`tests/llms/test_litellm_responses_adapter.py:182`): litellm stream uses responses events
- `test_litellm_embed_maps_metadata` (`tests/llms/test_litellm_responses_adapter.py:211`): litellm embed maps metadata

## `tests/llms/test_llm_contract.py`

Covered behaviors:

- `test_llmresponse_does_not_expose_legacy_typo_field` (`tests/llms/test_llm_contract.py:111`): llmresponse does not expose legacy typo field
- `test_chat_structured_validation_from_text` (`tests/llms/test_llm_contract.py:118`): chat structured validation from text
- `test_chat_repair_retry_until_valid` (`tests/llms/test_llm_contract.py:129`): chat repair retry until valid
- `test_chat_structured_raises_when_exhausted_retries` (`tests/llms/test_llm_contract.py:144`): chat structured raises when exhausted retries
- `test_embed_capability_error_when_disabled` (`tests/llms/test_llm_contract.py:158`): embed capability error when disabled
- `test_middleware_order_chat_embed_stream` (`tests/llms/test_llm_contract.py:173`): middleware order chat embed stream
- `test_chat_stream_validates_completion_payload_when_response_model` (`tests/llms/test_llm_contract.py:229`): chat stream validates completion payload when response model
- `test_factory_resolves_adapter_from_env` (`tests/llms/test_llm_contract.py:248`): factory resolves adapter from env
- `test_factory_rejects_unknown_adapter` (`tests/llms/test_llm_contract.py:254`): factory rejects unknown adapter
- `test_factory_supports_openai_adapter` (`tests/llms/test_llm_contract.py:259`): factory supports openai adapter
- `test_factory_passes_thinking_overrides_to_builtin_adapters` (`tests/llms/test_llm_contract.py:264`): factory passes thinking overrides to builtin adapters
- `test_chat_rejects_invalid_thinking_combo` (`tests/llms/test_llm_contract.py:275`): chat rejects invalid thinking combo
- `test_chat_stream_handle_cancel` (`tests/llms/test_llm_contract.py:287`): chat stream handle cancel
- `test_chat_stream_handle_interrupt_unsupported` (`tests/llms/test_llm_contract.py:316`): chat stream handle interrupt unsupported
- `test_start_session_raises_when_unsupported` (`tests/llms/test_llm_contract.py:337`): start session raises when unsupported
- `test_session_pause_resume_behavior` (`tests/llms/test_llm_contract.py:352`): session pause resume behavior
- `test_embed_model_falls_back_to_config` (`tests/llms/test_llm_contract.py:381`): embed model falls back to config
- `test_embed_raises_when_model_unresolved` (`tests/llms/test_llm_contract.py:391`): embed raises when model unresolved
- `test_chat_allows_provider_specific_effort_by_default` (`tests/llms/test_llm_contract.py:398`): chat allows provider specific effort by default
- `test_thinking_effort_alias_is_applied_from_instance_override` (`tests/llms/test_llm_contract.py:412`): thinking effort alias is applied from instance override
- `test_thinking_default_effort_is_overrideable_per_instance` (`tests/llms/test_llm_contract.py:429`): thinking default effort is overrideable per instance
- `test_thinking_effort_rejects_unknown_when_supported_set_defined` (`tests/llms/test_llm_contract.py:442`): thinking effort rejects unknown when supported set defined
- `test_thinking_effort_can_be_overridden_by_subclass_policy` (`tests/llms/test_llm_contract.py:463`): thinking effort can be overridden by subclass policy

## `tests/llms/test_openai_responses_adapter.py`

Covered behaviors:

- `test_openai_chat_uses_responses_api` (`tests/llms/test_openai_responses_adapter.py:160`): openai chat uses responses api
- `test_openai_stream_uses_responses_events` (`tests/llms/test_openai_responses_adapter.py:197`): openai stream uses responses events
- `test_openai_embed_maps_metadata` (`tests/llms/test_openai_responses_adapter.py:226`): openai embed maps metadata

## `tests/memory/test_memory_additional_standard.py`

Covered behaviors:

- `test_memory_module_getattr_errors_for_unknown_attr` (`tests/memory/test_memory_additional_standard.py:17`): memory module getattr errors for unknown attr
- `test_memory_store_capabilities_are_exposed` (`tests/memory/test_memory_additional_standard.py:25`): memory store capabilities are exposed
- `test_memory_store_base_context_manager_with_minimal_impl` (`tests/memory/test_memory_additional_standard.py:34`): memory store base context manager with minimal impl

## `tests/memory/test_memory_lifecycle.py`

Covered behaviors:

- `test_apply_event_retention_keeps_latest_with_priority_types` (`tests/memory/test_memory_lifecycle.py:27`): apply event retention keeps latest with priority types
- `test_apply_state_retention_keeps_resume_safe_checkpoint_keys` (`tests/memory/test_memory_lifecycle.py:43`): apply state retention keeps resume safe checkpoint keys
- `test_compact_thread_memory_applies_event_and_state_retention` (`tests/memory/test_memory_lifecycle.py:105`): compact thread memory applies event and state retention

## `tests/memory/test_memory_models_vector_factory.py`

Covered behaviors:

- `test_now_ms_and_new_id_helpers` (`tests/memory/test_memory_models_vector_factory.py:33`): now ms and new id helpers
- `test_json_helpers_round_trip_with_nested_values` (`tests/memory/test_memory_models_vector_factory.py:46`): json helpers round trip with nested values
- `test_cosine_similarity_and_vector_formatting` (`tests/memory/test_memory_models_vector_factory.py:60`): cosine similarity and vector formatting
- `test_env_bool_parsing` (`tests/memory/test_memory_models_vector_factory.py:73`): env bool parsing
- `test_memory_factory_selects_backends_and_env_options` (`tests/memory/test_memory_models_vector_factory.py:83`): memory factory selects backends and env options
- `test_memory_factory_postgres_requires_vector_dim` (`tests/memory/test_memory_models_vector_factory.py:137`): memory factory postgres requires vector dim
- `test_memory_factory_rejects_unknown_backend` (`tests/memory/test_memory_models_vector_factory.py:144`): memory factory rejects unknown backend

## `tests/memory/test_memory_store_in_memory.py`

Covered behaviors:

- `test_requires_setup_before_use` (`tests/memory/test_memory_store_in_memory.py:49`): requires setup before use
- `test_context_manager_runs_setup_and_close` (`tests/memory/test_memory_store_in_memory.py:55`): context manager runs setup and close
- `test_event_append_recent_and_since_with_limits` (`tests/memory/test_memory_store_in_memory.py:69`): event append recent and since with limits
- `test_state_put_get_and_prefix_listing_sorted` (`tests/memory/test_memory_store_in_memory.py:90`): state put get and prefix listing sorted
- `test_long_term_memory_list_text_search_vector_search_and_delete` (`tests/memory/test_memory_store_in_memory.py:114`): long term memory list text search vector search and delete
- `test_upsert_without_embedding_preserves_existing_embedding` (`tests/memory/test_memory_store_in_memory.py:163`): upsert without embedding preserves existing embedding
- `test_vector_search_ignores_memories_without_embedding_and_applies_scope` (`tests/memory/test_memory_store_in_memory.py:183`): vector search ignores memories without embedding and applies scope

## `tests/memory/test_memory_store_sqlite_integration.py`

Covered behaviors:

- `test_sqlite_store_requires_setup` (`tests/memory/test_memory_store_sqlite_integration.py:49`): sqlite store requires setup
- `test_sqlite_store_end_to_end_crud_and_search` (`tests/memory/test_memory_store_sqlite_integration.py:55`): sqlite store end to end crud and search
- `test_sqlite_upsert_keeps_created_at_and_embedding_when_not_provided` (`tests/memory/test_memory_store_sqlite_integration.py:129`): sqlite upsert keeps created at and embedding when not provided
- `test_sqlite_store_persists_data_across_reopen` (`tests/memory/test_memory_store_sqlite_integration.py:156`): sqlite store persists data across reopen

## `tests/tools/test_tool_security.py`

Covered behaviors:

- `test_validate_tool_args_against_sandbox_denies_path_and_command_operator` (`tests/tools/test_tool_security.py:41`): validate tool args against sandbox denies path and command operator
- `test_registry_sandbox_policy_blocks_at_call_time` (`tests/tools/test_tool_security.py:68`): registry sandbox policy blocks at call time
- `test_registry_output_limit_middleware_truncates_strings` (`tests/tools/test_tool_security.py:91`): registry output limit middleware truncates strings

## `tests/tools/test_tools_additional_standard.py`

Covered behaviors:

- `test_registry_management_and_summary_methods` (`tests/tools/test_tools_additional_standard.py:22`): registry management and summary methods
- `test_registry_rejects_invalid_max_concurrency` (`tests/tools/test_tools_additional_standard.py:48`): registry rejects invalid max concurrency
- `test_registry_call_many_raises_when_return_exceptions_false` (`tests/tools/test_tools_additional_standard.py:53`): registry call many raises when return exceptions false
- `test_registry_recent_calls_limit_and_error_recording` (`tests/tools/test_tools_additional_standard.py:70`): registry recent calls limit and error recording
- `test_normalize_json_schema_handles_non_dict_and_missing_fields` (`tests/tools/test_tools_additional_standard.py:92`): normalize json schema handles non dict and missing fields
- `test_export_tools_format_aliases` (`tests/tools/test_tools_additional_standard.py:101`): export tools format aliases
- `test_tool_middleware_timeout_is_enforced` (`tests/tools/test_tools_additional_standard.py:111`): tool middleware timeout is enforced
- `test_middleware_signature_validation_rejects_varargs` (`tests/tools/test_tools_additional_standard.py:129`): middleware signature validation rejects varargs
- `test_registry_middleware_signature_validation_rejects_varargs` (`tests/tools/test_tools_additional_standard.py:139`): registry middleware signature validation rejects varargs
- `test_registry_plugin_loader_no_group_returns_zero` (`tests/tools/test_tools_additional_standard.py:147`): registry plugin loader no group returns zero

## `tests/tools/test_tools_unit_and_integration.py`

Covered behaviors:

- `test_as_async_supports_sync_and_async_functions` (`tests/tools/test_tools_unit_and_integration.py:53`): as async supports sync and async functions
- `test_tool_decorator_uses_docstring_for_default_description` (`tests/tools/test_tools_unit_and_integration.py:67`): tool decorator uses docstring for default description
- `test_tool_function_signature_variants_are_supported` (`tests/tools/test_tools_unit_and_integration.py:78`): tool function signature variants are supported
- `test_invalid_tool_signature_is_rejected` (`tests/tools/test_tools_unit_and_integration.py:104`): invalid tool signature is rejected
- `test_tool_validation_and_execution_errors_return_failed_tool_result` (`tests/tools/test_tools_unit_and_integration.py:114`): tool validation and execution errors return failed tool result
- `test_tool_raise_on_error_raises_instead_of_returning_failure` (`tests/tools/test_tools_unit_and_integration.py:131`): tool raise on error raises instead of returning failure
- `test_tool_timeout_behavior_with_and_without_raise_on_error` (`tests/tools/test_tools_unit_and_integration.py:140`): tool timeout behavior with and without raise on error
- `test_tool_prehook_posthook_and_middleware_chain` (`tests/tools/test_tools_unit_and_integration.py:159`): tool prehook posthook and middleware chain
- `test_tool_fails_when_prehook_returns_non_dict` (`tests/tools/test_tools_unit_and_integration.py:188`): tool fails when prehook returns non dict
- `test_tool_fails_when_posthook_errors` (`tests/tools/test_tools_unit_and_integration.py:202`): tool fails when posthook errors
- `test_manual_middleware_signatures_supported` (`tests/tools/test_tools_unit_and_integration.py:216`): manual middleware signatures supported
- `test_registry_register_call_and_records_with_integration_chain` (`tests/tools/test_tools_unit_and_integration.py:245`): registry register call and records with integration chain
- `test_registry_duplicate_and_unknown_tool_errors` (`tests/tools/test_tools_unit_and_integration.py:285`): registry duplicate and unknown tool errors
- `test_registry_policy_enforcement_and_wrapping` (`tests/tools/test_tools_unit_and_integration.py:300`): registry policy enforcement and wrapping
- `test_registry_timeout_precedence_and_call_many_exceptions` (`tests/tools/test_tools_unit_and_integration.py:323`): registry timeout precedence and call many exceptions
- `test_registry_sync_middleware_path_and_management_methods` (`tests/tools/test_tools_unit_and_integration.py:346`): registry sync middleware path and management methods
- `test_registry_middleware_style_validation` (`tests/tools/test_tools_unit_and_integration.py:371`): registry middleware style validation
- `test_export_helpers_and_openai_tool_format` (`tests/tools/test_tools_unit_and_integration.py:379`): export helpers and openai tool format
- `test_tool_class_direct_instantiation_for_hook_types` (`tests/tools/test_tools_unit_and_integration.py:399`): tool class direct instantiation for hook types
- `test_registry_set_middlewares_replaces_existing_chain` (`tests/tools/test_tools_unit_and_integration.py:416`): registry set middlewares replaces existing chain
- `test_registry_middleware_decorator_preserves_name_and_description` (`tests/tools/test_tools_unit_and_integration.py:447`): registry middleware decorator preserves name and description

## Change Management Tip

If you change behavior in `afk.core`, `afk.agents`, or `afk.tools`, search this file for the closest existing behavior and either:

1. update the matching test + description, or
2. add a new test and append it in the correct section.
