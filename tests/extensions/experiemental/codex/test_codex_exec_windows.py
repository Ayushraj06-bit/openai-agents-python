from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agents.extensions.experimental.codex.exec import CodexExec, CodexExecArgs

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Requires a Windows batch launcher")

# The socket makes the exit of the fake Codex observable, and lets a failing test release it.
_FAKE_CODEX = r"""
import os, socket
connection = socket.create_connection(("127.0.0.1", {port}))
os.write(1, b"started\n")
connection.recv(1)
"""


async def _stop_with_signal(stream: AsyncGenerator[str, None], args: Any) -> None:
    args.signal.set()
    with pytest.raises(RuntimeError, match="Codex exec exited with code"):
        await anext(stream)


async def _stop_with_idle_timeout(stream: AsyncGenerator[str, None], args: Any) -> None:
    args.idle_timeout_seconds = 0.05
    with pytest.raises(RuntimeError, match="Codex stream idle"):
        await anext(stream)


async def _stop_with_cancellation(stream: AsyncGenerator[str, None], _args: Any) -> None:
    read_task = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    read_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await read_task


async def _stop_with_aclose(stream: AsyncGenerator[str, None], _args: Any) -> None:
    await stream.aclose()


@pytest.mark.parametrize(
    "stop",
    [
        pytest.param(_stop_with_signal, id="signal"),
        pytest.param(_stop_with_idle_timeout, id="idle-timeout"),
        pytest.param(_stop_with_cancellation, id="cancellation"),
        pytest.param(_stop_with_aclose, id="aclose"),
    ],
)
def test_stopping_run_stops_codex_started_by_batch_launcher(
    tmp_path: Path, stop: Callable[[AsyncGenerator[str, None], Any], Awaitable[None]]
) -> None:
    # Importing Modal installs a selector event loop policy, which cannot start subprocesses.
    loop = asyncio.ProactorEventLoop()
    try:
        loop.run_until_complete(_assert_stop_ends_codex(tmp_path, stop))
    finally:
        loop.close()


async def _assert_stop_ends_codex(
    tmp_path: Path, stop: Callable[[AsyncGenerator[str, None], Any], Awaitable[None]]
) -> None:
    connected: asyncio.Future[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
        asyncio.get_running_loop().create_future()
    )
    server = await asyncio.start_server(
        lambda reader, writer: connected.set_result((reader, writer)), "127.0.0.1", 0
    )
    port = server.sockets[0].getsockname()[1]
    (tmp_path / "fake_codex.py").write_text(_FAKE_CODEX.format(port=port), encoding="utf-8")
    launcher = tmp_path / "codex.cmd"
    launcher.write_text(f'@"{sys.executable}" "%~dp0fake_codex.py" %*\n', encoding="utf-8")

    # Mutable arguments keep process startup out of the idle timeout, which starts after line one.
    args: Any = SimpleNamespace(**vars(CodexExecArgs(input="hello", signal=asyncio.Event())))
    stream = CodexExec(executable_path=str(launcher)).run(args)
    stop_task: asyncio.Future[None] | None = None
    try:
        assert await anext(stream) == "started"
        reader, _ = await asyncio.wait_for(asyncio.shield(connected), timeout=10)

        stop_task = asyncio.ensure_future(stop(stream, args))
        done, _ = await asyncio.wait({stop_task}, timeout=10)
        assert stop_task in done, "Stopping the run did not stop Codex."
        await stop_task
        with contextlib.suppress(ConnectionError):
            assert await asyncio.wait_for(reader.read(), timeout=10) == b""
    finally:
        server.close()
        if connected.done():
            connected.result()[1].close()
        if stop_task is not None:
            await asyncio.gather(stop_task, return_exceptions=True)
        await stream.aclose()
