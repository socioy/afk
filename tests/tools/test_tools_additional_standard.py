from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from afk.tools import ToolContext, ToolRegistry, ToolSpec, export_tools, tool
from afk.tools.core.base import Middleware
from afk.tools.core.export import normalize_json_schema
from afk.tools.registery import RegistryMiddleware


def run_async(coro):
    return asyncio.run(coro)


class EchoArgs(BaseModel):
    text: str


def test_registry_management_and_summary_methods():
    @tool(args_model=EchoArgs, name="one", description="first")
    def one(args: EchoArgs) -> str:
        return args.text

    @tool(args_model=EchoArgs, name="two", description="second")
    def two(args: EchoArgs) -> str:
        return args.text

    registry = ToolRegistry()
    registry.register_many([one, two])

    assert set(registry.names()) == {"one", "two"}
    assert registry.has("one") is True
    assert registry.has("missing") is False
    assert {summary["name"] for summary in registry.list_tool_summaries()} == {
        "one",
        "two",
    }
    assert {spec.name for spec in registry.specs()} == {"one", "two"}

    registry.unregister("one")
    assert registry.has("one") is False
    assert registry.names() == ["two"]


def test_registry_rejects_invalid_max_concurrency():
    with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
        ToolRegistry(max_concurrency=0)


def test_registry_call_many_raises_when_return_exceptions_false():
    @tool(args_model=EchoArgs, name="echo")
    def echo(args: EchoArgs) -> str:
        return args.text

    registry = ToolRegistry()
    registry.register(echo)

    with pytest.raises(Exception, match="Unknown tool"):
        run_async(
            registry.call_many(
                [("echo", {"text": "ok"}), ("missing", {"text": "x"})],
                return_exceptions=False,
            )
        )


def test_registry_recent_calls_limit_and_error_recording():
    @tool(args_model=EchoArgs, name="echo")
    def echo(args: EchoArgs) -> str:
        return args.text

    registry = ToolRegistry()
    registry.register(echo)

    run_async(registry.call("echo", {"text": "1"}, tool_call_id="t1"))
    run_async(registry.call("echo", {"text": "2"}, tool_call_id="t2"))

    with pytest.raises(Exception):
        run_async(registry.call("missing", {"text": "x"}, tool_call_id="t3"))

    limited = registry.recent_calls(limit=2)
    assert len(limited) == 2
    assert limited[0].tool_call_id == "t1"
    assert limited[1].tool_call_id == "t2"
    # Unknown-tool failures happen before execution and are not recorded.
    assert all(record.ok is True for record in limited)


def test_normalize_json_schema_handles_non_dict_and_missing_fields():
    assert normalize_json_schema("bad") == {"type": "object", "properties": {}}
    assert normalize_json_schema({}) == {"type": "object", "properties": {}}
    assert normalize_json_schema({"type": "array"}) == {
        "type": "array",
        "properties": {},
    }


def test_export_tools_format_aliases():
    @tool(args_model=EchoArgs, name="echo_alias")
    def echo_alias(args: EchoArgs) -> str:
        return args.text

    exported = export_tools([echo_alias], format="function")
    assert exported[0]["type"] == "function"
    assert exported[0]["function"]["name"] == "echo_alias"


def test_tool_middleware_timeout_is_enforced():
    async def slow_mw(call_next, args, ctx):
        await asyncio.sleep(0.05)
        return await call_next(args, ctx)

    mw = Middleware(
        spec=ToolSpec(name="slow_mw", description="", parameters_schema={}),
        fn=slow_mw,
        default_timeout=0.001,
    )

    async def core(args, ctx):
        return args.text

    with pytest.raises(Exception, match="exceeded timeout"):
        run_async(mw.call(core, EchoArgs(text="x"), ToolContext()))


def test_middleware_signature_validation_rejects_varargs():
    def bad(*args):  # noqa: ANN002
        return None

    with pytest.raises(Exception, match="cannot have \\*args or \\*\\*kwargs"):
        Middleware(
            spec=ToolSpec(name="bad", description="", parameters_schema={}), fn=bad
        )


def test_registry_middleware_signature_validation_rejects_varargs():
    def bad(*args):  # noqa: ANN002
        return None

    with pytest.raises(ValueError, match="cannot use \\*args/\\*\\*kwargs"):
        RegistryMiddleware(bad)


def test_registry_plugin_loader_no_group_returns_zero():
    registry = ToolRegistry()
    loaded = registry.load_plugins(entry_point_group="definitely.not.a.real.group")
    assert loaded == 0
