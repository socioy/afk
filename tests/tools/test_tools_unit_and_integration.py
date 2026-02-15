from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from afk.tools import (
    Middleware,
    PostHook,
    PreHook,
    RegistryMiddleware,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolSpec,
    as_async,
    export_tools,
    middleware,
    posthook,
    prehook,
    registry_middleware,
    tool,
    toolspec_to_litellm_tool,
)
from afk.tools.core.errors import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolPolicyError,
    ToolTimeoutError,
    ToolValidationError,
)


def run_async(coro):
    return asyncio.run(coro)


class EchoArgs(BaseModel):
    text: str


class AddArgs(BaseModel):
    a: int
    b: int


class PostArgs(BaseModel):
    output: str
    tool_name: str | None = None


def test_as_async_supports_sync_and_async_functions():
    def sync_fn(value: int) -> int:
        return value + 1

    async def async_fn(value: int) -> int:
        return value + 2

    sync_wrapped = as_async(sync_fn)
    async_wrapped = as_async(async_fn)

    assert run_async(sync_wrapped(10)) == 11
    assert run_async(async_wrapped(10)) == 12


def test_tool_decorator_uses_docstring_for_default_description():
    @tool(args_model=EchoArgs)
    def doc_tool(args: EchoArgs) -> str:
        """Echoes transformed user text."""
        return args.text

    assert doc_tool.spec.name == "doc_tool"
    assert doc_tool.spec.description == "Echoes transformed user text."
    assert doc_tool.spec.parameters_schema["type"] == "object"


def test_tool_function_signature_variants_are_supported():
    @tool(args_model=EchoArgs, name="args_only")
    def args_only(args: EchoArgs) -> str:
        return args.text

    @tool(args_model=EchoArgs, name="args_ctx")
    def args_ctx(args: EchoArgs, ctx: ToolContext) -> str:
        return f"{ctx.user_id}:{args.text}"

    @tool(args_model=EchoArgs, name="ctx_args")
    def ctx_args(ctx: ToolContext, args: EchoArgs) -> str:
        return f"{ctx.request_id}:{args.text}"

    result_1 = run_async(args_only.call({"text": "hello"}))
    result_2 = run_async(
        args_ctx.call({"text": "hello"}, ctx=ToolContext(user_id="u1"))
    )
    result_3 = run_async(
        ctx_args.call({"text": "hello"}, ctx=ToolContext(request_id="r-123"))
    )

    assert result_1.success and result_1.output == "hello"
    assert result_2.success and result_2.output == "u1:hello"
    assert result_3.success and result_3.output == "r-123:hello"


def test_invalid_tool_signature_is_rejected():
    with pytest.raises(ToolValidationError):

        @tool(args_model=EchoArgs, name="bad")
        def bad(first: EchoArgs, second: str) -> str:
            return first.text + second

        _ = bad


def test_tool_validation_and_execution_errors_return_failed_tool_result():
    @tool(args_model=AddArgs, name="add")
    def add_tool(args: AddArgs) -> int:
        if args.b == 0:
            raise ValueError("b cannot be zero here")
        return args.a + args.b

    bad_validation = run_async(add_tool.call({"a": 1}))
    bad_execution = run_async(add_tool.call({"a": 1, "b": 0}))

    assert bad_validation.success is False
    assert "Invalid arguments" in (bad_validation.error_message or "")

    assert bad_execution.success is False
    assert "Error executing tool 'add'" in (bad_execution.error_message or "")


def test_tool_raise_on_error_raises_instead_of_returning_failure():
    @tool(args_model=AddArgs, name="must_add", raise_on_error=True)
    def must_add(args: AddArgs) -> int:
        return args.a + args.b

    with pytest.raises(ToolValidationError):
        run_async(must_add.call({"a": 1}))


def test_tool_timeout_behavior_with_and_without_raise_on_error():
    @tool(args_model=EchoArgs, name="slow", timeout=0.01)
    async def slow_tool(args: EchoArgs) -> str:
        await asyncio.sleep(0.05)
        return args.text

    failed = run_async(slow_tool.call({"text": "x"}))
    assert failed.success is False
    assert "execution exceeded timeout" in (failed.error_message or "")

    @tool(args_model=EchoArgs, name="slow_raise", timeout=0.01, raise_on_error=True)
    async def slow_raise(args: EchoArgs) -> str:
        await asyncio.sleep(0.05)
        return args.text

    with pytest.raises(ToolTimeoutError):
        run_async(slow_raise.call({"text": "x"}))


def test_tool_prehook_posthook_and_middleware_chain():
    @prehook(args_model=EchoArgs, name="strip")
    def strip_hook(args: EchoArgs) -> dict[str, str]:
        return {"text": args.text.strip()}

    @posthook(args_model=PostArgs, name="decorate")
    def decorate_hook(args: PostArgs) -> str:
        return f"{args.output}|{args.tool_name}"

    @middleware(name="prefix")
    async def prefix_mw(call_next, args: EchoArgs, ctx: ToolContext):
        out = await call_next(EchoArgs(text=f"prefix-{args.text}"), ctx)
        return f"{out}-suffix"

    @tool(
        args_model=EchoArgs,
        name="echo",
        prehooks=[strip_hook],
        middlewares=[prefix_mw],
        posthooks=[decorate_hook],
    )
    def echo_tool(args: EchoArgs) -> str:
        return args.text.upper()

    result = run_async(echo_tool.call({"text": "  hi  "}))
    assert result.success
    assert result.output == "PREFIX-HI-suffix|echo"


def test_tool_fails_when_prehook_returns_non_dict():
    @prehook(args_model=EchoArgs, name="bad_pre")
    def bad_pre(args: EchoArgs) -> str:
        return args.text

    @tool(args_model=EchoArgs, name="echo_with_bad_pre", prehooks=[bad_pre])
    def echo_with_bad_pre(args: EchoArgs) -> str:
        return args.text

    result = run_async(echo_with_bad_pre.call({"text": "hello"}))
    assert result.success is False
    assert "must return a dict of args" in (result.error_message or "")


def test_tool_fails_when_posthook_errors():
    @posthook(args_model=PostArgs, name="explode")
    def explode(args: PostArgs) -> str:
        raise RuntimeError("boom")

    @tool(args_model=EchoArgs, name="echo_post_fail", posthooks=[explode])
    def echo_post_fail(args: EchoArgs) -> str:
        return args.text

    result = run_async(echo_post_fail.call({"text": "a"}))
    assert result.success is False
    assert "Post-hook 'explode' failed" in (result.error_message or "")


def test_manual_middleware_signatures_supported():
    async def core(args: EchoArgs, ctx: ToolContext) -> str:
        return f"{ctx.user_id}:{args.text}"

    async def next_args(call_next, args: EchoArgs):
        out = await call_next(EchoArgs(text=args.text + "_1"), ToolContext(user_id="u"))
        return out + "_A"

    async def ctx_args_next(ctx: ToolContext, args: EchoArgs, call_next):
        out = await call_next(EchoArgs(text=args.text + "_2"), ctx)
        return out + "_B"

    mw_1 = Middleware(
        spec=ToolSpec(name="mw1", description="mw1", parameters_schema={}), fn=next_args
    )
    mw_2 = Middleware(
        spec=ToolSpec(name="mw2", description="mw2", parameters_schema={}),
        fn=ctx_args_next,
    )

    first = run_async(mw_1.call(core, EchoArgs(text="x"), ToolContext(user_id="base")))
    second = run_async(
        mw_2.call(core, EchoArgs(text="x"), ToolContext(user_id="ctx-user"))
    )

    assert first == "u:x_1_A"
    assert second == "ctx-user:x_2_B"


def test_registry_register_call_and_records_with_integration_chain():
    @tool(args_model=AddArgs, name="add_two")
    def add_two(args: AddArgs) -> int:
        return args.a + args.b

    call_sequence: list[str] = []

    @registry_middleware(name="audit")
    async def audit_mw(call_next, tool_obj, raw_args, ctx, timeout, tool_call_id):
        call_sequence.append("before")
        raw_args = {"a": raw_args["a"] + 1, "b": raw_args["b"]}
        result = await call_next(tool_obj, raw_args, ctx, timeout, tool_call_id)
        call_sequence.append("after")
        result.metadata["mw"] = "audit"
        return result

    registry = ToolRegistry(default_timeout=1.0, middlewares=[audit_mw])
    registry.register(add_two)

    result = run_async(
        registry.call(
            "add_two",
            {"a": 1, "b": 3},
            ctx=ToolContext(user_id="u1"),
            tool_call_id="tcid-1",
        )
    )

    assert result.success
    assert result.output == 5
    assert result.metadata["mw"] == "audit"
    assert call_sequence == ["before", "after"]

    calls = registry.recent_calls(limit=1)
    assert len(calls) == 1
    assert calls[0].tool_name == "add_two"
    assert calls[0].ok is True
    assert calls[0].tool_call_id == "tcid-1"


def test_registry_duplicate_and_unknown_tool_errors():
    @tool(args_model=EchoArgs, name="echo_dup")
    def echo_dup(args: EchoArgs) -> str:
        return args.text

    registry = ToolRegistry()
    registry.register(echo_dup)

    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(echo_dup)

    with pytest.raises(ToolNotFoundError):
        run_async(registry.call("missing", {"text": "x"}))


def test_registry_policy_enforcement_and_wrapping():
    @tool(args_model=EchoArgs, name="echo_policy")
    def echo_policy(args: EchoArgs) -> str:
        return args.text

    def deny_policy(name: str, raw_args: dict, ctx: ToolContext):
        raise ToolPolicyError(f"blocked:{name}")

    def crashing_policy(name: str, raw_args: dict, ctx: ToolContext):
        raise RuntimeError("policy crash")

    denied_registry = ToolRegistry(policy=deny_policy)
    denied_registry.register(echo_policy)

    with pytest.raises(ToolPolicyError, match="blocked:echo_policy"):
        run_async(denied_registry.call("echo_policy", {"text": "x"}))

    wrapped_registry = ToolRegistry(policy=crashing_policy)
    wrapped_registry.register(echo_policy)
    with pytest.raises(ToolPolicyError, match="policy crash"):
        run_async(wrapped_registry.call("echo_policy", {"text": "x"}))


def test_registry_timeout_precedence_and_call_many_exceptions():
    @tool(args_model=EchoArgs, name="slow_tool", timeout=0.5)
    async def slow_tool(args: EchoArgs) -> str:
        await asyncio.sleep(0.05)
        return args.text

    registry = ToolRegistry(default_timeout=0.01)
    registry.register(slow_tool)

    with pytest.raises(ToolTimeoutError):
        run_async(registry.call("slow_tool", {"text": "x"}, timeout=0.001))

    results = run_async(
        registry.call_many(
            [("slow_tool", {"text": "ok"}), ("does_not_exist", {"x": 1})],
            return_exceptions=True,
        )
    )
    assert len(results) == 2
    assert getattr(results[0], "success", False) is True
    assert isinstance(results[1], ToolNotFoundError)


def test_registry_sync_middleware_path_and_management_methods():
    @tool(args_model=AddArgs, name="sum")
    def sum_tool(args: AddArgs) -> int:
        return args.a + args.b

    def sync_registry_mw(tool_obj, raw_args, ctx, call_next, timeout, tool_call_id):
        modified = dict(raw_args)
        modified["a"] += 10
        result = call_next(tool_obj, modified, ctx, timeout, tool_call_id)
        result.metadata["sync"] = True
        return result

    registry = ToolRegistry()
    registry.register(sum_tool)
    registry.add_middleware(sync_registry_mw)

    assert registry.list_middlewares() == ["sync_registry_mw"]
    result = run_async(registry.call("sum", {"a": 1, "b": 2}))
    assert result.output == 13
    assert result.metadata["sync"] is True

    registry.clear_middlewares()
    assert registry.list_middlewares() == []


def test_registry_middleware_style_validation():
    def bad_registry_mw(a, b, c):  # wrong arity
        return None

    with pytest.raises(ValueError):
        RegistryMiddleware(bad_registry_mw)


def test_export_helpers_and_openai_tool_format():
    @tool(args_model=EchoArgs, name="exportable", description="export me")
    def exportable(args: EchoArgs) -> str:
        return args.text

    registry = ToolRegistry()
    registry.register(exportable)

    from_registry = registry.to_openai_function_tools()
    from_export = export_tools(registry.list(), format="openai")
    from_spec = toolspec_to_litellm_tool(exportable.spec)

    assert from_registry[0]["function"]["name"] == "exportable"
    assert from_export[0]["function"]["description"] == "export me"
    assert from_spec["function"]["parameters"]["type"] == "object"

    with pytest.raises(ValueError, match="Unknown export format"):
        export_tools(registry.list(), format="unknown")


def test_tool_class_direct_instantiation_for_hook_types():
    def core_fn(args: EchoArgs) -> str:
        return args.text

    def post_fn(args: PostArgs) -> str:
        return args.output

    spec = ToolSpec(name="raw_tool", description="raw", parameters_schema={})
    tool_obj = Tool(spec=spec, fn=core_fn, args_model=EchoArgs)
    pre_obj = PreHook(spec=spec, fn=core_fn, args_model=EchoArgs)
    post_obj = PostHook(spec=spec, fn=post_fn, args_model=PostArgs)

    assert run_async(tool_obj.call({"text": "x"})).output == "x"
    assert run_async(pre_obj.call({"text": "x"})).output == "x"
    assert run_async(post_obj.call({"output": "x", "tool_name": "raw"})).output == "x"


def test_registry_set_middlewares_replaces_existing_chain():
    @tool(args_model=EchoArgs, name="echo_replace")
    def echo_replace(args: EchoArgs) -> str:
        return args.text

    @registry_middleware(name="first")
    async def first(call_next, tool_obj, raw_args, ctx):
        result = await call_next(tool_obj, raw_args, ctx, None, None)
        result.metadata["chain"] = "first"
        return result

    @registry_middleware(name="second")
    async def second(call_next, tool_obj, raw_args, ctx):
        result = await call_next(tool_obj, raw_args, ctx, None, None)
        result.metadata["chain"] = "second"
        return result

    registry = ToolRegistry(middlewares=[first])
    registry.register(echo_replace)
    assert (
        run_async(registry.call("echo_replace", {"text": "x"})).metadata["chain"]
        == "first"
    )

    registry.set_middlewares([second])
    assert (
        run_async(registry.call("echo_replace", {"text": "x"})).metadata["chain"]
        == "second"
    )


def test_registry_middleware_decorator_preserves_name_and_description():
    @registry_middleware(name="custom_name", description="custom description")
    async def mw(call_next, tool_obj, raw_args, ctx):
        return await call_next(tool_obj, raw_args, ctx, None, None)

    assert mw.name == "custom_name"
    assert getattr(mw, "description") == "custom description"
