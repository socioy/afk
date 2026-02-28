from __future__ import annotations

import asyncio

import pytest

from afk.llms.utils import run_sync


def test_run_sync_executes_coroutine_from_sync_context():
    async def _coro() -> int:
        await asyncio.sleep(0)
        return 42

    assert run_sync(_coro()) == 42


def test_run_sync_supports_repeated_calls_in_cli_style_loop():
    async def _echo(value: int) -> int:
        await asyncio.sleep(0)
        return value

    assert [run_sync(_echo(i)) for i in range(3)] == [0, 1, 2]


def test_run_sync_raises_inside_running_event_loop():
    async def _inside() -> None:
        coro = asyncio.sleep(0)
        with pytest.raises(RuntimeError, match=r"Cannot use \*_sync methods"):
            run_sync(coro)
        coro.close()

    asyncio.run(_inside())
