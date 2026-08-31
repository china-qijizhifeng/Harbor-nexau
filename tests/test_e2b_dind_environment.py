from __future__ import annotations

import asyncio
import logging
from pathlib import Path, PurePath
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.environments.base import ExecResult, ResourceMode
from harbor.models.task.config import EnvironmentConfig
from harbor.utils.import_path import import_class

from nexau_harbor.constants import NEXAU_DETACHED_EXEC_ENV
from nexau_harbor.e2b_dind_environment import (
    E2B_DIND_DEFAULT_TEMPLATE,
    NexAUE2BDinDEnvironment,
    _AgentLogDownloadError,
    _compose_env_file_content,
    _compose_project_name,
    _diagnostic_tail,
    _E2BDinDCompose,
    _PendingAgentLogDownload,
    _remote_pack_command_with_exclusions,
    _reserve_sandbox_start_delay,
)
from nexau_harbor.e2b_environment import NexAUE2BEnvironment


class _AsyncChunkStream:
    def __init__(self, chunks: list[bytes | BaseException]) -> None:
        self._chunks = iter(chunks)

    async def __aenter__(self) -> _AsyncChunkStream:
        return self

    async def __aexit__(self, *exc_info) -> None:  # noqa: ANN002
        del exc_info

    def __aiter__(self) -> _AsyncChunkStream:
        return self

    async def __anext__(self) -> bytes:
        try:
            chunk = next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk


def test_custom_dind_environment_import_path_is_loadable() -> None:
    environment_class = import_class(
        "nexau_harbor.e2b_dind_environment:NexAUE2BDinDEnvironment",
        label="environment",
    )

    assert environment_class is NexAUE2BDinDEnvironment


def test_dind_environment_declares_compose_and_resource_capabilities() -> None:
    environment = object.__new__(NexAUE2BDinDEnvironment)

    assert environment.capabilities.docker_compose
    assert environment.capabilities.dynamic_network_policy
    assert environment.capabilities.disable_internet
    assert environment.capabilities.network_allowlist
    assert not environment.capabilities.gpus
    assert environment.resource_capabilities().model_dump() == {
        "cpu_limit": True,
        "cpu_request": True,
        "memory_limit": True,
        "memory_request": True,
    }


def test_compose_project_name_is_safe_stable_and_bounded() -> None:
    raw = "Frontier.Task/Trial.With Spaces_" + "x" * 100
    first = _compose_project_name(raw)
    second = _compose_project_name(raw)

    assert first == second
    assert len(first) <= 63
    assert first[0].isalnum()
    assert set(first) <= set("abcdefghijklmnopqrstuvwxyz0123456789_-")


def test_compose_env_file_quotes_interpolation_and_special_characters() -> None:
    content = _compose_env_file_content(
        {
            "CONTEXT_DIR": "/harbor/environment",
            "PRIVATE_TOKEN": "literal-$HOME-a'b\\c",
        }
    )

    assert content == (
        "CONTEXT_DIR='/harbor/environment'\nPRIVATE_TOKEN='literal-$HOME-a\\'b\\\\c'\n"
    )


def test_compose_env_file_rejects_invalid_variable_names() -> None:
    with pytest.raises(ValueError, match="Invalid Docker Compose environment variable"):
        _compose_env_file_content({"INVALID-NAME": "value"})


def test_sandbox_start_slots_are_reserved_at_the_configured_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nexau_harbor.e2b_environment._next_sandbox_start_at",
        0.0,
    )
    monkeypatch.setattr("nexau_harbor.e2b_environment.time.monotonic", lambda: 100.0)

    assert _reserve_sandbox_start_delay(5) == 0
    assert _reserve_sandbox_start_delay(5) == 5
    assert _reserve_sandbox_start_delay(5) == 10


def _fake_super_init(
    environment_dir: Path,
    task_env_config: EnvironmentConfig,
):
    def init(self: NexAUE2BDinDEnvironment, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        del args, kwargs
        self.environment_dir = environment_dir
        self.environment_name = "compose-task"
        self.session_id = "compose-task__trial__env"
        self.task_env_config = task_env_config
        self.extra_docker_compose_paths = []
        self._persistent_env = {}
        self._mounts = []
        self._cpu_resource_mode = ResourceMode.AUTO
        self._memory_resource_mode = ResourceMode.AUTO
        self._sandbox = None
        self.logger = logging.getLogger("test-e2b-dind")

    return init


def test_dind_environment_uses_versioned_template_and_accepts_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert E2B_DIND_DEFAULT_TEMPLATE == "e2b-nex/harbor-nexau-dind:v1"
    (tmp_path / "Dockerfile").write_text("FROM alpine:3.22\n")
    monkeypatch.setattr(
        NexAUE2BEnvironment,
        "__init__",
        _fake_super_init(
            tmp_path,
            EnvironmentConfig(cpus=8, memory_mb=32_768, storage_mb=40_960),
        ),
    )

    environment = NexAUE2BDinDEnvironment()

    assert environment._dind_template == E2B_DIND_DEFAULT_TEMPLATE
    assert environment._template_name == E2B_DIND_DEFAULT_TEMPLATE
    assert environment._workdir is None


def test_custom_dind_template_requires_explicit_capacity_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "Dockerfile").write_text("FROM alpine:3.22\n")
    monkeypatch.setattr(
        NexAUE2BEnvironment,
        "__init__",
        _fake_super_init(tmp_path, EnvironmentConfig(cpus=2)),
    )

    with pytest.raises(ValueError, match="requires explicit capacity metadata"):
        NexAUE2BDinDEnvironment(dind_template="custom-dind:v2")

    environment = NexAUE2BDinDEnvironment(
        dind_template="custom-dind:v2",
        dind_cpu_capacity=8,
        dind_memory_capacity_mb=32_768,
        dind_storage_capacity_mb=41_062,
    )
    assert environment._dind_template == "custom-dind:v2"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (EnvironmentConfig(cpus=9), "requests 9 CPU"),
        (EnvironmentConfig(memory_mb=32_769), "requests 32769 memory"),
        (EnvironmentConfig(storage_mb=41_063), "requests 41063 storage"),
    ],
)
def test_dind_environment_rejects_requirements_above_template_capacity(
    config: EnvironmentConfig,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "Dockerfile").write_text("FROM alpine:3.22\n")
    monkeypatch.setattr(
        NexAUE2BEnvironment,
        "__init__",
        _fake_super_init(tmp_path, config),
    )

    with pytest.raises(ValueError, match=message):
        NexAUE2BDinDEnvironment()


async def test_exec_targets_compose_main_and_merges_environment() -> None:
    expected = ExecResult(stdout="ok", stderr="", return_code=0)
    compose_exec = AsyncMock(return_value=expected)
    environment = object.__new__(NexAUE2BDinDEnvironment)
    environment._compose = SimpleNamespace(exec=compose_exec)
    environment.task_env_config = EnvironmentConfig(workdir="/workspace")
    environment._workdir = None
    environment._merge_env = MagicMock(side_effect=lambda env: {"PERSISTENT": "yes", **(env or {})})
    environment._resolve_user = MagicMock(return_value="agent")

    result = await environment.exec(
        "python runner.py",
        env={"ONE_SHOT": "yes"},
        timeout_sec=321,
    )

    assert result is expected
    compose_exec.assert_awaited_once_with(
        "python runner.py",
        cwd="/workspace",
        env={"PERSISTENT": "yes", "ONE_SHOT": "yes"},
        timeout_sec=321,
        user="agent",
    )


async def test_exec_routes_detached_marker_without_forwarding_it() -> None:
    expected = ExecResult(stdout="done", stderr="", return_code=0)
    detached_exec = AsyncMock(return_value=expected)
    environment = object.__new__(NexAUE2BDinDEnvironment)
    environment._exec_detached = detached_exec

    result = await environment.exec(
        "long-agent",
        env={NEXAU_DETACHED_EXEC_ENV: "1", "VISIBLE": "yes"},
        timeout_sec=7200,
        user="agent",
    )

    assert result is expected
    detached_exec.assert_awaited_once_with(
        "long-agent",
        cwd=None,
        env={"VISIBLE": "yes"},
        timeout_sec=7200,
        user="agent",
    )


def test_detached_status_is_polled_as_outer_host_root() -> None:
    environment = object.__new__(NexAUE2BDinDEnvironment)

    assert environment._detached_status_user("container-only-user") == "root"


async def test_compose_build_retries_only_transient_network_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transient = ExecResult(stdout="", stderr="failed to copy: unexpected EOF", return_code=1)
    success = ExecResult(stdout="done", stderr="", return_code=0)
    detached_exec = AsyncMock(side_effect=[transient, success])
    compose = object.__new__(_E2BDinDCompose)
    compose._env = SimpleNamespace(
        _exec_on_host_detached=detached_exec,
        logger=logging.getLogger("test-e2b-dind"),
    )
    compose._compose_env_vars = MagicMock(return_value={"BUILDKIT_PROGRESS": "plain"})
    compose._compose_cmd = MagicMock(return_value="docker compose build")
    sleep = AsyncMock()
    monkeypatch.setattr("nexau_harbor.e2b_dind_environment.asyncio.sleep", sleep)

    result = await compose._compose_build_exec(["build"], phase="build")

    assert result is success
    assert detached_exec.await_count == 2
    assert detached_exec.await_args_list[0].kwargs["diagnostic_label"] == (
        "docker compose build attempt 1/3"
    )
    assert detached_exec.await_args_list[1].kwargs["diagnostic_label"] == (
        "docker compose build attempt 2/3"
    )
    sleep.assert_awaited_once_with(5)


async def test_compose_build_does_not_retry_a_deterministic_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = ExecResult(stdout="", stderr="Dockerfile parse error", return_code=1)
    detached_exec = AsyncMock(return_value=failure)
    compose = object.__new__(_E2BDinDCompose)
    compose._env = SimpleNamespace(
        _exec_on_host_detached=detached_exec,
        logger=logging.getLogger("test-e2b-dind"),
    )
    compose._compose_env_vars = MagicMock(return_value={})
    compose._compose_cmd = MagicMock(return_value="docker compose build")
    sleep = AsyncMock()
    monkeypatch.setattr("nexau_harbor.e2b_dind_environment.asyncio.sleep", sleep)

    result = await compose._compose_build_exec(["build"], phase="build")

    assert result is failure
    detached_exec.assert_awaited_once()
    sleep.assert_not_awaited()


async def test_detached_host_build_has_no_inner_command_timeout() -> None:
    expected = ExecResult(stdout="built", stderr="", return_code=0)
    handle = SimpleNamespace(pid=42, disconnect=AsyncMock())
    environment = object.__new__(NexAUE2BDinDEnvironment)
    environment.logger = logging.getLogger("test-e2b-dind")
    environment._dispatch_command = AsyncMock(return_value=handle)
    environment._poll_detached_command = AsyncMock(return_value=expected)

    result = await environment._exec_on_host_detached(
        "docker compose build",
        env={"BUILDKIT_PROGRESS": "plain"},
    )

    assert result is expected
    environment._dispatch_command.assert_awaited_once()
    assert environment._dispatch_command.await_args.kwargs["timeout_sec"] is None
    dispatched_wrapper = environment._dispatch_command.await_args.args[0]
    assert "/tmp/harbor-nexau-exec/harbor-exec-" in dispatched_wrapper
    assert "/logs/agent/harbor-exec-" not in dispatched_wrapper


async def test_cancelled_detached_host_build_logs_saved_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handle = SimpleNamespace(pid=73, disconnect=AsyncMock())
    command_kill = AsyncMock()
    environment = object.__new__(NexAUE2BDinDEnvironment)
    environment.logger = logging.getLogger("test-e2b-dind-cancel-diagnostic")
    environment._sandbox = SimpleNamespace(commands=SimpleNamespace(kill=command_kill))
    environment._dispatch_command = AsyncMock(return_value=handle)
    environment._poll_detached_command = AsyncMock(side_effect=asyncio.CancelledError())
    environment._detached_exec_result = AsyncMock(
        return_value=ExecResult(
            stdout="pulling layer 17/20\n",
            stderr="registry connection stalled\n",
            return_code=-1,
        )
    )

    with caplog.at_level(logging.ERROR, logger=environment.logger.name):
        with pytest.raises(asyncio.CancelledError):
            await environment._exec_on_host_detached(
                "docker compose build",
                diagnostic_label="docker compose build attempt 1/3",
            )

    command_kill.assert_awaited_once_with(
        73,
        request_timeout=30,
    )
    snapshot_call = environment._detached_exec_result.await_args
    assert snapshot_call.kwargs["user"] == "root"
    assert snapshot_call.kwargs["return_code"] == -1
    assert "Cancelled before" in snapshot_call.kwargs["diagnostic"]
    assert "docker compose build attempt 1/3" in caplog.text
    assert "process 73" in caplog.text
    assert "pulling layer 17/20" in caplog.text
    assert "registry connection stalled" in caplog.text


def test_diagnostic_tail_keeps_only_bounded_suffix() -> None:
    long_output = "prefix-marker\n" + ("x" * 20_000) + "\nlast-progress"

    captured = _diagnostic_tail(long_output)

    assert "prefix-marker" not in captured
    assert "omitted" in captured
    assert captured.endswith("last-progress")


def test_remote_log_archive_can_exclude_the_prefetched_trace() -> None:
    command = _remote_pack_command_with_exclusions(
        "/logs/agent",
        "/tmp/agent-logs.tar.gz",
        ("nexau_trace_events.jsonl",),
    )

    assert command == (
        "tar -czf /tmp/agent-logs.tar.gz --exclude=./nexau_trace_events.jsonl -C /logs/agent ."
    )


async def test_host_file_download_streams_to_an_atomic_target(tmp_path: Path) -> None:
    stream = _AsyncChunkStream([b"first-", b"second"])
    read = AsyncMock(return_value=stream)
    environment = object.__new__(NexAUE2BDinDEnvironment)
    environment._sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
    target = tmp_path / "nexau_trace_events.jsonl"
    target.write_bytes(b"old-content")

    await environment._stream_file_from_host("/logs/agent/trace.jsonl", target)

    assert target.read_bytes() == b"first-second"
    assert list(tmp_path.glob("*.part")) == []
    read.assert_awaited_once_with(
        "/logs/agent/trace.jsonl",
        format="stream",
        user="root",
        gzip=True,
        stream_idle_timeout=300,
    )


async def test_failed_stream_download_keeps_the_previous_atomic_target(tmp_path: Path) -> None:
    stream = _AsyncChunkStream([b"partial", ConnectionError("stream reset")])
    environment = object.__new__(NexAUE2BDinDEnvironment)
    environment._sandbox = SimpleNamespace(
        files=SimpleNamespace(read=AsyncMock(return_value=stream))
    )
    target = tmp_path / "nexau_trace_events.jsonl"
    target.write_bytes(b"previous-complete-copy")

    with pytest.raises(ConnectionError, match="stream reset"):
        await environment._stream_file_from_host("/logs/agent/trace.jsonl", target)

    assert target.read_bytes() == b"previous-complete-copy"
    assert list(tmp_path.glob("*.part")) == []


async def test_agent_log_download_keeps_trace_when_remaining_archive_fails(
    tmp_path: Path,
) -> None:
    trace_payload = b'{"event":"start"}\n'

    async def fetch_trace(source: str, target: Path | str) -> None:
        assert source == "/logs/agent/nexau_trace_events.jsonl"
        Path(target).write_bytes(trace_payload)

    compose = object.__new__(_E2BDinDCompose)
    compose._env = SimpleNamespace(
        logger=MagicMock(),
        _exec_on_host=AsyncMock(
            return_value=ExecResult(
                stdout="file\tnexau_trace_events.jsonl\n",
                stderr="",
                return_code=0,
            )
        ),
    )
    compose._fetch_file_from_host = AsyncMock(side_effect=fetch_trace)
    compose._fetch_dir_from_host = AsyncMock(side_effect=ConnectionError("archive reset"))

    await compose.download_agent_logs("/logs/agent", tmp_path)

    assert (tmp_path / "nexau_trace_events.jsonl").read_bytes() == trace_payload
    compose._fetch_dir_from_host.assert_awaited_once_with(
        "/logs/agent",
        tmp_path,
        exclude_relative_paths=("nexau_trace_events.jsonl",),
    )
    compose._env.logger.exception.assert_called_once()


async def test_agent_log_download_reports_when_both_trace_paths_fail(
    tmp_path: Path,
) -> None:
    compose = object.__new__(_E2BDinDCompose)
    compose._env = SimpleNamespace(
        logger=MagicMock(),
        _exec_on_host=AsyncMock(
            return_value=ExecResult(
                stdout="file\tnexau_trace_events.jsonl\n",
                stderr="",
                return_code=0,
            )
        ),
    )
    compose._fetch_file_from_host = AsyncMock(side_effect=ConnectionError("trace reset"))
    compose._fetch_dir_from_host = AsyncMock(side_effect=ConnectionError("archive reset"))

    with pytest.raises(_AgentLogDownloadError) as error:
        await compose.download_agent_logs("/logs/agent", tmp_path)

    assert error.value.trace_expected
    assert "Both priority trace download" in str(error.value)


async def test_agent_log_download_prioritizes_codex_events_and_rollouts(
    tmp_path: Path,
) -> None:
    payloads = {
        "codex_events.jsonl": b'{"type":"thread.started"}\n',
        "codex_stderr.log": b"warning\n",
        "northgate-responses-bridge.log": b"bridge ready\n",
        "codex_result.json": b'{"status":"running"}\n',
        "codex_history.jsonl": b'{"session_id":"thread"}\n',
    }

    async def fetch_file(source: str, target: Path | str) -> None:
        Path(target).write_bytes(payloads[PurePath(source).name])

    async def fetch_dir(
        source: str,
        target: Path | str,
        *,
        exclude_relative_paths: tuple[str, ...] = (),
        detached_pack: bool = False,
    ) -> None:
        if source == "/logs/agent/sessions":
            assert detached_pack
            rollout = Path(target) / "2026/08/19/rollout-test.jsonl"
            rollout.parent.mkdir(parents=True)
            rollout.write_text('{"type":"session_meta"}\n')
            return
        assert source == "/logs/agent"
        assert exclude_relative_paths == (
            "codex_events.jsonl",
            "codex_stderr.log",
            "northgate-responses-bridge.log",
            "codex_result.json",
            "codex_history.jsonl",
            "sessions",
        )
        raise ConnectionError("secondary archive reset")

    compose = object.__new__(_E2BDinDCompose)
    compose._env = SimpleNamespace(
        logger=MagicMock(),
        _exec_on_host=AsyncMock(
            return_value=ExecResult(
                stdout=(
                    "file\tcodex_events.jsonl\n"
                    "file\tcodex_stderr.log\n"
                    "file\tnorthgate-responses-bridge.log\n"
                    "file\tcodex_result.json\n"
                    "file\tcodex_history.jsonl\n"
                    "dir\tsessions\n"
                ),
                stderr="",
                return_code=0,
            )
        ),
    )
    compose._fetch_file_from_host = AsyncMock(side_effect=fetch_file)
    compose._fetch_dir_from_host = AsyncMock(side_effect=fetch_dir)

    await compose.download_agent_logs("/logs/agent", tmp_path)

    assert (tmp_path / "codex_events.jsonl").read_bytes() == payloads[
        "codex_events.jsonl"
    ]
    assert (tmp_path / "codex_stderr.log").read_bytes() == payloads["codex_stderr.log"]
    assert (tmp_path / "northgate-responses-bridge.log").read_bytes() == payloads[
        "northgate-responses-bridge.log"
    ]
    assert (tmp_path / "codex_result.json").read_bytes() == payloads[
        "codex_result.json"
    ]
    assert (tmp_path / "codex_history.jsonl").read_bytes() == payloads[
        "codex_history.jsonl"
    ]
    assert (tmp_path / "sessions/2026/08/19/rollout-test.jsonl").is_file()
    compose._env.logger.exception.assert_called_once()


async def test_failed_agent_log_download_is_retried_before_cleanup(
    tmp_path: Path,
) -> None:
    environment = object.__new__(NexAUE2BDinDEnvironment)
    environment._compose = SimpleNamespace(
        download_agent_logs=AsyncMock(
            side_effect=_AgentLogDownloadError("unavailable", trace_expected=True)
        )
    )
    environment._pending_agent_log_download = None
    environment.logger = MagicMock()

    with pytest.raises(_AgentLogDownloadError):
        await environment.download_dir("/logs/agent", tmp_path)

    pending = environment._pending_agent_log_download
    assert pending == _PendingAgentLogDownload(
        "/logs/agent",
        tmp_path,
        trace_expected=True,
    )


async def test_unrecovered_trace_preserves_sandbox_for_manual_recovery(
    tmp_path: Path,
) -> None:
    set_timeout = AsyncMock()
    sandbox = SimpleNamespace(sandbox_id="sandbox-with-trace", set_timeout=set_timeout)
    environment = object.__new__(NexAUE2BDinDEnvironment)
    environment._sandbox = sandbox
    environment._pending_agent_log_download = _PendingAgentLogDownload(
        "/logs/agent",
        tmp_path,
        trace_expected=True,
    )
    environment._retry_pending_agent_log_download = AsyncMock(return_value=False)
    environment._stop_keepalive = AsyncMock()
    environment._keepalive_deadline = None
    environment._compose_files_staged = True
    environment.logger = MagicMock()

    await environment.stop(delete=True)

    environment._stop_keepalive.assert_awaited_once_with()
    set_timeout.assert_awaited_once_with(900)
    assert environment._sandbox is None
    assert environment._pending_agent_log_download is None
    environment.logger.error.assert_called_once_with(
        "E2B sandbox %s was not deleted because its Agent trace could not be "
        "recovered; it remains available for up to %s seconds for manual recovery",
        "sandbox-with-trace",
        900,
    )
