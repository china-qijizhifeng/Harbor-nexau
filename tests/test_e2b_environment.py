from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from harbor.environments.base import ExecResult, ResourceMode
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.utils.import_path import import_class

from nexau_harbor import e2b_environment
from nexau_harbor.constants import NEXAU_DETACHED_EXEC_ENV
from nexau_harbor.e2b_environment import (
    E2B_DEFAULT_MAX_LIFETIME_SEC,
    E2B_DETACHED_MAX_CONSECUTIVE_POLL_ERRORS,
    E2B_DETACHED_POLL_INTERVAL_SEC,
    E2B_DETACHED_REQUEST_TIMEOUT_SEC,
    E2B_KEEPALIVE_INTERVAL_SEC,
    E2B_MAX_LIFETIME_ENV,
    E2B_NO_PROXY_HOSTS_ENV,
    E2B_SANDBOX_TIMEOUT_SEC,
    E2B_START_RAMP_DELAY_ENV,
    NexAUE2BEnvironment,
)

UV_COPY = "COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/"


def test_custom_environment_import_path_is_loadable() -> None:
    environment_class = import_class(
        "nexau_harbor.e2b_environment:NexAUE2BEnvironment",
        label="environment",
    )

    assert environment_class is NexAUE2BEnvironment


def test_missing_default_template_tag_detection_is_narrow() -> None:
    assert e2b_environment._is_missing_default_template_tag(
        RuntimeError(
            "404: tag 'default' does not exist for template 'e2b-nex/ico-path-patch__2676acce2372'"
        )
    )
    assert not e2b_environment._is_missing_default_template_tag(
        RuntimeError("404: sandbox not found")
    )
    assert not e2b_environment._is_missing_default_template_tag(
        RuntimeError("500: tag 'default' does not exist for template 'example/task'")
    )


async def test_start_rebuilds_once_when_cached_template_has_no_launchable_default(
    monkeypatch,
) -> None:
    missing_default = RuntimeError(
        "404: tag 'default' does not exist for template 'e2b-nex/ico-path-patch__2676acce2372'"
    )
    upstream_start = AsyncMock(side_effect=[missing_default, None])
    monkeypatch.setattr(e2b_environment.harbor_e2b.E2BEnvironment, "start", upstream_start)

    environment = object.__new__(NexAUE2BEnvironment)
    environment._template_name = "ico-path-patch__2676acce2372"
    environment.logger = MagicMock()

    await environment.start(force_build=False)

    assert upstream_start.await_args_list == [call(False), call(force_build=True)]
    environment.logger.warning.assert_called_once()


async def test_start_propagates_failed_automatic_rebuild_without_looping(monkeypatch) -> None:
    missing_default = RuntimeError(
        "404: tag 'default' does not exist for template 'e2b-nex/ico-path-patch__2676acce2372'"
    )
    upstream_start = AsyncMock(side_effect=[missing_default, missing_default])
    monkeypatch.setattr(e2b_environment.harbor_e2b.E2BEnvironment, "start", upstream_start)

    environment = object.__new__(NexAUE2BEnvironment)
    environment._template_name = "ico-path-patch__2676acce2372"
    environment.logger = MagicMock()

    with pytest.raises(RuntimeError, match="tag 'default' does not exist"):
        await environment.start(force_build=False)

    assert upstream_start.await_args_list == [call(False), call(force_build=True)]
    environment.logger.warning.assert_called_once()


@pytest.mark.parametrize(
    ("force_build", "error"),
    [
        (False, RuntimeError("404: sandbox not found")),
        (
            True,
            RuntimeError(
                "404: tag 'default' does not exist for template "
                "'e2b-nex/ico-path-patch__2676acce2372'"
            ),
        ),
    ],
)
async def test_start_does_not_rebuild_unrelated_errors_or_loop_after_forced_build(
    monkeypatch,
    force_build: bool,
    error: RuntimeError,
) -> None:
    upstream_start = AsyncMock(side_effect=error)
    monkeypatch.setattr(e2b_environment.harbor_e2b.E2BEnvironment, "start", upstream_start)

    environment = object.__new__(NexAUE2BEnvironment)
    environment._template_name = "ico-path-patch__2676acce2372"
    environment.logger = MagicMock()

    with pytest.raises(RuntimeError, match=str(error)):
        await environment.start(force_build=force_build)

    upstream_start.assert_awaited_once_with(force_build)
    environment.logger.warning.assert_not_called()


async def test_exec_routes_internal_marker_to_detached_mode_without_forwarding_it() -> None:
    expected = ExecResult(stdout="done", stderr="", return_code=0)
    detached_exec = AsyncMock(return_value=expected)
    environment = object.__new__(NexAUE2BEnvironment)
    environment._exec_detached = detached_exec

    result = await environment.exec(
        "long-running-agent",
        cwd="/root",
        env={NEXAU_DETACHED_EXEC_ENV: "1", "VISIBLE": "yes"},
        timeout_sec=123,
        user="root",
    )

    assert result is expected
    detached_exec.assert_awaited_once_with(
        "long-running-agent",
        cwd="/root",
        env={"VISIBLE": "yes"},
        timeout_sec=123,
        user="root",
    )


async def test_exec_without_internal_marker_uses_standard_harbor_path(monkeypatch) -> None:
    expected = ExecResult(stdout="setup complete", stderr="", return_code=0)
    standard_exec = AsyncMock(return_value=expected)
    monkeypatch.setattr(
        e2b_environment.harbor_e2b.E2BEnvironment,
        "exec",
        standard_exec,
    )
    environment = object.__new__(NexAUE2BEnvironment)

    result = await environment.exec(
        "setup-command",
        cwd="/root",
        env={"VISIBLE": "yes"},
        timeout_sec=123,
        user="root",
    )

    assert result is expected
    standard_exec.assert_awaited_once_with(
        "setup-command",
        cwd="/root",
        env={"VISIBLE": "yes"},
        timeout_sec=123,
        user="root",
    )


def _detached_test_environment(
    command_run: AsyncMock,
    file_read: AsyncMock,
) -> tuple[NexAUE2BEnvironment, AsyncMock]:
    command_kill = AsyncMock(return_value=True)
    environment = object.__new__(NexAUE2BEnvironment)
    environment._sandbox = SimpleNamespace(
        sandbox_id="sandbox-detached",
        commands=SimpleNamespace(run=command_run, kill=command_kill),
        files=SimpleNamespace(read=file_read),
    )
    environment.task_env_config = EnvironmentConfig()
    environment._workdir = None
    environment._resolve_user = MagicMock(side_effect=lambda user: user or "root")
    environment._merge_env = MagicMock(side_effect=lambda env: env)
    environment.logger = logging.getLogger("test-e2b-detached")
    return environment, command_kill


async def test_detached_exec_disconnects_and_polls_short_commands(monkeypatch) -> None:
    handle = SimpleNamespace(pid=321, disconnect=AsyncMock())
    command_run = AsyncMock(
        side_effect=[
            handle,
            SimpleNamespace(stdout="__HARBOR_NEXAU_STATUS__:RUNNING\n"),
            SimpleNamespace(stdout="__HARBOR_NEXAU_STATUS__:DONE:0\n"),
        ]
    )

    async def read_file(path: str, **kwargs) -> str:  # noqa: ANN003
        del kwargs
        if path.endswith("/stdout"):
            return "captured stdout"
        if path.endswith("/stderr"):
            return "captured stderr"
        raise AssertionError(f"Unexpected detached output path: {path}")

    file_read = AsyncMock(side_effect=read_file)
    environment, _ = _detached_test_environment(command_run, file_read)
    monkeypatch.setattr(e2b_environment, "E2B_DETACHED_POLL_INTERVAL_SEC", 0)

    result = await environment._exec_detached(
        "python /nexau-workspace/runner.py",
        cwd="/root",
        env={"VISIBLE": "yes"},
        timeout_sec=None,
        user=None,
    )

    assert result == ExecResult(
        stdout="captured stdout",
        stderr="captured stderr",
        return_code=0,
    )
    handle.disconnect.assert_awaited_once_with()
    assert command_run.await_count == 3
    launch = command_run.await_args_list[0].kwargs
    assert launch["background"] is True
    assert launch["timeout"] == 0
    assert launch["envs"] == {"VISIBLE": "yes"}
    assert "python /nexau-workspace/runner.py" in launch["cmd"]
    assert "exitcode.tmp" in launch["cmd"]
    for poll in command_run.await_args_list[1:]:
        assert poll.kwargs["timeout"] == E2B_DETACHED_REQUEST_TIMEOUT_SEC
        assert poll.kwargs["request_timeout"] == E2B_DETACHED_REQUEST_TIMEOUT_SEC


async def test_detached_exec_preserves_nonzero_exit_code(monkeypatch) -> None:
    handle = SimpleNamespace(pid=654, disconnect=AsyncMock())
    command_run = AsyncMock(
        side_effect=[
            handle,
            SimpleNamespace(stdout="__HARBOR_NEXAU_STATUS__:DONE:137\n"),
        ]
    )
    file_read = AsyncMock(side_effect=["", "killed"])
    environment, _ = _detached_test_environment(command_run, file_read)
    monkeypatch.setattr(e2b_environment, "E2B_DETACHED_POLL_INTERVAL_SEC", 0)

    result = await environment._exec_detached(
        "long-agent",
        cwd=None,
        env=None,
        timeout_sec=None,
        user="root",
    )

    assert result.return_code == 137
    assert result.stderr == "killed"


async def test_detached_exec_reports_process_disappearing_without_status(monkeypatch) -> None:
    handle = SimpleNamespace(pid=987, disconnect=AsyncMock())
    command_run = AsyncMock(
        side_effect=[
            handle,
            SimpleNamespace(stdout="__HARBOR_NEXAU_STATUS__:CRASHED\n"),
        ]
    )
    file_read = AsyncMock(side_effect=["partial output", ""])
    environment, _ = _detached_test_environment(command_run, file_read)
    monkeypatch.setattr(e2b_environment, "E2B_DETACHED_POLL_INTERVAL_SEC", 0)

    result = await environment._exec_detached(
        "long-agent",
        cwd=None,
        env=None,
        timeout_sec=None,
        user="root",
    )

    assert result.return_code == -1
    assert result.stdout == "partial output"
    assert "process 987 disappeared without writing an exit status" in result.stderr


async def test_detached_exec_retries_transient_poll_failure(monkeypatch) -> None:
    handle = SimpleNamespace(pid=246, disconnect=AsyncMock())
    command_run = AsyncMock(
        side_effect=[
            handle,
            ConnectionError("temporary poll failure"),
            SimpleNamespace(stdout="__HARBOR_NEXAU_STATUS__:DONE:0\n"),
        ]
    )
    file_read = AsyncMock(side_effect=["", ""])
    environment, _ = _detached_test_environment(command_run, file_read)
    monkeypatch.setattr(e2b_environment, "E2B_DETACHED_POLL_INTERVAL_SEC", 0)

    result = await environment._exec_detached(
        "long-agent",
        cwd=None,
        env=None,
        timeout_sec=None,
        user="root",
    )

    assert result.return_code == 0
    assert command_run.await_count == 3
    assert E2B_DETACHED_MAX_CONSECUTIVE_POLL_ERRORS == 5
    assert E2B_DETACHED_POLL_INTERVAL_SEC == 30


async def test_detached_exec_kills_remote_process_when_cancelled(monkeypatch) -> None:
    handle = SimpleNamespace(pid=135, disconnect=AsyncMock())
    command_run = AsyncMock(return_value=handle)
    file_read = AsyncMock()
    environment, command_kill = _detached_test_environment(command_run, file_read)
    poll = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(environment, "_poll_detached_command", poll)

    with pytest.raises(asyncio.CancelledError):
        await environment._exec_detached(
            "long-agent",
            cwd=None,
            env=None,
            timeout_sec=None,
            user="root",
        )

    command_kill.assert_awaited_once_with(
        135,
        request_timeout=E2B_DETACHED_REQUEST_TIMEOUT_SEC,
    )


def test_rewrites_astral_uv_copy_for_python_stage_without_mutating_other_lines() -> None:
    dockerfile = f"""\
FROM python:3.12-slim-bookworm
{UV_COPY}
RUN uv pip install --system pytest==9.1.1
"""

    rewritten, rewrite_count = e2b_environment._rewrite_e2b_incompatible_uv_copy(dockerfile)

    assert rewrite_count == 1
    assert UV_COPY not in rewritten
    assert "RUN python -m pip install --no-cache-dir uv==0.9.7" in rewritten
    assert "RUN uv pip install --system pytest==9.1.1" in rewritten


@pytest.mark.parametrize(
    "dockerfile",
    [
        UV_COPY,
        f"FROM debian:bookworm-slim\n{UV_COPY}\n",
        (
            "FROM python:3.12-slim-bookworm\n"
            "COPY --from=example.invalid/uv:0.9.7 /uv /uvx /usr/local/bin/\n"
        ),
    ],
)
def test_does_not_rewrite_uv_copy_outside_supported_python_case(dockerfile: str) -> None:
    rewritten, rewrite_count = e2b_environment._rewrite_e2b_incompatible_uv_copy(dockerfile)

    assert rewrite_count == 0
    assert rewritten == dockerfile


async def test_create_template_passes_compatible_dockerfile_without_changing_source(
    tmp_path, monkeypatch
) -> None:
    dockerfile_path = tmp_path / "Dockerfile"
    original_dockerfile = f"FROM python:3.12-slim-bookworm\n{UV_COPY}\n"
    dockerfile_path.write_text(original_dockerfile)

    parsed_template = object()
    template_builder = MagicMock()
    template_builder.from_dockerfile.return_value = parsed_template
    template_factory = MagicMock(return_value=template_builder)
    build = AsyncMock()
    monkeypatch.setattr(e2b_environment.harbor_e2b, "Template", template_factory)
    monkeypatch.setattr(e2b_environment.harbor_e2b.AsyncTemplate, "build", build)

    environment = object.__new__(NexAUE2BEnvironment)
    environment.environment_dir = tmp_path
    environment.task_env_config = EnvironmentConfig(cpus=2, memory_mb=4096)
    environment._cpu_resource_mode = ResourceMode.AUTO
    environment._memory_resource_mode = ResourceMode.AUTO
    environment._template_name = "ico-path-patch__template"
    environment.logger = logging.getLogger("test-e2b-template-compatibility")

    await environment._create_template()

    template_factory.assert_called_once_with(file_context_path=str(tmp_path))
    compatible_dockerfile = template_builder.from_dockerfile.call_args.kwargs[
        "dockerfile_content_or_path"
    ]
    assert UV_COPY not in compatible_dockerfile
    assert "RUN python -m pip install --no-cache-dir uv==0.9.7" in compatible_dockerfile
    build.assert_awaited_once_with(
        template=parsed_template,
        alias="ico-path-patch__template",
        cpu_count=2,
        memory_mb=4096,
        on_build_logs=build.await_args.kwargs["on_build_logs"],
    )
    assert callable(build.await_args.kwargs["on_build_logs"])
    assert dockerfile_path.read_text() == original_dockerfile


async def test_create_template_preserves_prebuilt_image_behavior(monkeypatch) -> None:
    parsed_template = object()
    template_builder = MagicMock()
    template_builder.from_image.return_value = parsed_template
    template_factory = MagicMock(return_value=template_builder)
    build = AsyncMock()
    monkeypatch.setattr(e2b_environment.harbor_e2b, "Template", template_factory)
    monkeypatch.setattr(e2b_environment.harbor_e2b.AsyncTemplate, "build", build)

    environment = object.__new__(NexAUE2BEnvironment)
    environment.task_env_config = EnvironmentConfig(docker_image="example.invalid/task:1")
    environment._cpu_resource_mode = ResourceMode.IGNORE
    environment._memory_resource_mode = ResourceMode.IGNORE
    environment._template_name = "prebuilt-task__template"

    await environment._create_template()

    template_factory.assert_called_once_with()
    template_builder.from_image.assert_called_once_with(image="example.invalid/task:1")
    build.assert_awaited_once_with(
        template=parsed_template,
        alias="prebuilt-task__template",
        on_build_logs=build.await_args.kwargs["on_build_logs"],
    )
    assert callable(build.await_args.kwargs["on_build_logs"])


async def test_template_build_log_is_attached_to_failure(monkeypatch) -> None:
    failure = RuntimeError("error waiting for provisioning sandbox: exit status: 1")

    async def build(**kwargs) -> None:
        kwargs["on_build_logs"](
            SimpleNamespace(
                level="debug",
                message="/usr/local/bin/provision.sh: line 61: apt-get: command not found",
            )
        )
        raise failure

    parsed_template = object()
    template_builder = MagicMock()
    template_builder.from_image.return_value = parsed_template
    monkeypatch.setattr(
        e2b_environment.harbor_e2b,
        "Template",
        MagicMock(return_value=template_builder),
    )
    monkeypatch.setattr(e2b_environment.harbor_e2b.AsyncTemplate, "build", build)

    environment = object.__new__(NexAUE2BEnvironment)
    environment.task_env_config = EnvironmentConfig(docker_image="fedora:42")
    environment._cpu_resource_mode = ResourceMode.IGNORE
    environment._memory_resource_mode = ResourceMode.IGNORE
    environment._template_name = "retro-console-soc__template"
    environment._e2b_template_build_attempts = 1
    environment._last_template_build_log_tail = ""

    with pytest.raises(RuntimeError) as exc_info:
        await environment._create_template()

    assert exc_info.value is failure
    assert "apt-get: command not found" in failure.e2b_build_log_tail


async def test_template_build_attempt_count_can_be_reduced_for_preflight(monkeypatch) -> None:
    parsed_template = object()
    template_builder = MagicMock()
    template_builder.from_image.return_value = parsed_template
    build = AsyncMock(side_effect=RuntimeError("template failed"))
    monkeypatch.setattr(
        e2b_environment.harbor_e2b,
        "Template",
        MagicMock(return_value=template_builder),
    )
    monkeypatch.setattr(e2b_environment.harbor_e2b.AsyncTemplate, "build", build)

    environment = object.__new__(NexAUE2BEnvironment)
    environment.task_env_config = EnvironmentConfig(docker_image="example.invalid/task:1")
    environment._cpu_resource_mode = ResourceMode.IGNORE
    environment._memory_resource_mode = ResourceMode.IGNORE
    environment._template_name = "single-attempt__template"
    environment._e2b_template_build_attempts = 1
    environment._last_template_build_log_tail = ""

    with pytest.raises(RuntimeError, match="template failed"):
        await environment._create_template()

    assert build.await_count == 1


async def test_template_build_timeout_wraps_only_the_remote_build(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class RecordingTimeout:
        def __init__(self, timeout_sec: float) -> None:
            observed["timeout_sec"] = timeout_sec

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    parsed_template = object()
    template_builder = MagicMock()
    template_builder.from_image.return_value = parsed_template
    build = AsyncMock()
    monkeypatch.setattr(
        e2b_environment.harbor_e2b,
        "Template",
        MagicMock(return_value=template_builder),
    )
    monkeypatch.setattr(e2b_environment.harbor_e2b.AsyncTemplate, "build", build)
    monkeypatch.setattr(e2b_environment.asyncio, "timeout", RecordingTimeout)

    environment = object.__new__(NexAUE2BEnvironment)
    environment.task_env_config = EnvironmentConfig(docker_image="example.invalid/task:1")
    environment._cpu_resource_mode = ResourceMode.IGNORE
    environment._memory_resource_mode = ResourceMode.IGNORE
    environment._template_name = "bounded-build__template"
    environment._e2b_template_build_attempts = 1
    environment._e2b_template_build_timeout_sec = 17
    environment._last_template_build_log_tail = ""

    await environment._create_template()

    assert observed["timeout_sec"] == 17
    assert build.await_count == 1


def test_max_lifetime_defaults_to_24_hours_and_accepts_overrides(monkeypatch) -> None:
    monkeypatch.delenv(E2B_MAX_LIFETIME_ENV, raising=False)
    assert e2b_environment._resolve_max_lifetime_sec(None) == 86_400

    monkeypatch.setenv(E2B_MAX_LIFETIME_ENV, "7200")
    assert e2b_environment._resolve_max_lifetime_sec(None) == 7_200
    assert e2b_environment._resolve_max_lifetime_sec("1800") == 1_800
    assert E2B_DEFAULT_MAX_LIFETIME_SEC == 86_400


def test_environment_kwarg_takes_precedence_over_env(monkeypatch) -> None:
    monkeypatch.setenv(E2B_MAX_LIFETIME_ENV, "7200")
    monkeypatch.setattr(
        e2b_environment.harbor_e2b.E2BEnvironment,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    environment = NexAUE2BEnvironment(max_lifetime_sec="1800")

    assert environment._max_lifetime_sec == 1_800


def test_start_ramp_delay_defaults_to_env_and_accepts_kwarg(monkeypatch) -> None:
    monkeypatch.setenv(E2B_START_RAMP_DELAY_ENV, "5")
    monkeypatch.setattr(
        e2b_environment.harbor_e2b.E2BEnvironment,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    from_env = NexAUE2BEnvironment()
    from_kwarg = NexAUE2BEnvironment(e2b_start_ramp_delay_sec="2.5")

    assert from_env._e2b_start_ramp_delay_sec == 5
    assert from_kwarg._e2b_start_ramp_delay_sec == 2.5


def test_template_build_controls_accept_preflight_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        e2b_environment.harbor_e2b.E2BEnvironment,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    environment = NexAUE2BEnvironment(
        e2b_template_build_attempts="1",
        e2b_template_build_timeout_sec="17.5",
    )

    assert environment._e2b_template_build_attempts == 1
    assert environment._e2b_template_build_timeout_sec == 17.5


@pytest.mark.parametrize("value", [0, -1, "invalid", True, float("inf")])
def test_template_build_timeout_rejects_invalid_values(value, monkeypatch) -> None:
    monkeypatch.setattr(
        e2b_environment.harbor_e2b.E2BEnvironment,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="e2b_template_build_timeout_sec"):
        NexAUE2BEnvironment(e2b_template_build_timeout_sec=value)


@pytest.mark.parametrize("value", [-1, "invalid", True, float("inf")])
def test_start_ramp_delay_rejects_invalid_values(value, monkeypatch) -> None:
    monkeypatch.setattr(
        e2b_environment.harbor_e2b.E2BEnvironment,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match=E2B_START_RAMP_DELAY_ENV):
        NexAUE2BEnvironment(e2b_start_ramp_delay_sec=value)


def test_environment_merges_configured_hosts_into_both_no_proxy_spellings(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        E2B_NO_PROXY_HOSTS_ENV,
        "api.e2b.app,.e2b.app,api.e2b.app",
    )
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,api.e2b.app")
    monkeypatch.setenv("no_proxy", "localhost")
    monkeypatch.setattr(
        e2b_environment.harbor_e2b.E2BEnvironment,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    NexAUE2BEnvironment()

    assert os.environ["NO_PROXY"] == "127.0.0.1,api.e2b.app,.e2b.app"
    assert os.environ["no_proxy"] == "localhost,api.e2b.app,.e2b.app"


@pytest.mark.parametrize("value", [0, -1, "0", "invalid", True, 1.5])
def test_max_lifetime_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match=E2B_MAX_LIFETIME_ENV):
        e2b_environment._resolve_max_lifetime_sec(value)


@pytest.mark.parametrize(
    ("remaining_lifetime_sec", "expected_timeout_sec"),
    [
        (90_000, 3_600),
        (3_599.9, 3_599),
        (61.2, 61),
        (1.1, 1),
        (0.9, None),
        (0, None),
    ],
)
def test_lease_timeout_never_exceeds_remaining_lifetime(
    remaining_lifetime_sec: float,
    expected_timeout_sec: int | None,
) -> None:
    assert e2b_environment._bounded_lease_timeout(remaining_lifetime_sec) == expected_timeout_sec


async def test_create_sandbox_uses_and_renews_one_hour_timeout(monkeypatch) -> None:
    renewed_twice = asyncio.Event()
    renewal_count = 0

    async def set_timeout(timeout: int) -> None:
        nonlocal renewal_count
        assert timeout == E2B_SANDBOX_TIMEOUT_SEC
        renewal_count += 1
        if renewal_count >= 2:
            renewed_twice.set()

    sandbox = SimpleNamespace(
        sandbox_id="sandbox-1",
        set_timeout=AsyncMock(side_effect=set_timeout),
    )
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(
        e2b_environment.harbor_e2b.AsyncSandbox,
        "create",
        create,
    )
    monkeypatch.setattr(e2b_environment, "E2B_KEEPALIVE_INTERVAL_SEC", 0.001)
    reserve_start = MagicMock(return_value=0.0)
    monkeypatch.setattr(e2b_environment, "_reserve_sandbox_start_delay", reserve_start)

    environment = object.__new__(NexAUE2BEnvironment)
    environment.environment_name = "terminal-bench-task"
    environment.session_id = "trial-session"
    environment._template_name = "terminal-bench-template"
    environment.task_env_config = EnvironmentConfig(env={"TASK_ENV": "task"})
    environment._persistent_env = {"TRIAL_ENV": "trial"}
    environment._network_policy = NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    environment._max_lifetime_sec = E2B_DEFAULT_MAX_LIFETIME_SEC
    environment._e2b_start_ramp_delay_sec = 5
    environment._keepalive_task = None
    environment._keepalive_deadline = None
    environment.logger = logging.getLogger("test-e2b-keepalive")

    await environment._create_sandbox()
    try:
        await asyncio.wait_for(renewed_twice.wait(), timeout=1)

        assert environment._sandbox is sandbox
        create.assert_awaited_once_with(
            template="terminal-bench-template",
            metadata={
                "environment_name": "terminal-bench-task",
                "session_id": "trial-session",
            },
            envs={"TASK_ENV": "task", "TRIAL_ENV": "trial"},
            timeout=E2B_SANDBOX_TIMEOUT_SEC,
            allow_internet_access=True,
            network=None,
        )
        assert sandbox.set_timeout.await_count >= 2
        reserve_start.assert_called_once_with(5)
        assert E2B_SANDBOX_TIMEOUT_SEC == 3_600
        assert E2B_KEEPALIVE_INTERVAL_SEC == 60
    finally:
        await environment._stop_keepalive()


async def test_initial_lease_does_not_exceed_smaller_lifetime_cap(monkeypatch) -> None:
    sandbox = SimpleNamespace(
        sandbox_id="sandbox-1",
        set_timeout=AsyncMock(),
    )
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(e2b_environment.harbor_e2b.AsyncSandbox, "create", create)

    environment = object.__new__(NexAUE2BEnvironment)
    environment.environment_name = "terminal-bench-task"
    environment.session_id = "trial-session"
    environment._template_name = "terminal-bench-template"
    environment.task_env_config = EnvironmentConfig()
    environment._persistent_env = {}
    environment._network_policy = NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    environment._max_lifetime_sec = 1_800
    environment._e2b_start_ramp_delay_sec = 0
    environment._keepalive_task = None
    environment._keepalive_deadline = None
    environment.logger = logging.getLogger("test-e2b-short-cap")

    await environment._create_sandbox()
    try:
        assert create.await_args.kwargs["timeout"] == 1_800
    finally:
        await environment._stop_keepalive()


async def test_stop_cancels_keepalive_before_killing_sandbox() -> None:
    sandbox = SimpleNamespace(
        sandbox_id="sandbox-1",
        set_timeout=AsyncMock(),
        kill=AsyncMock(return_value=True),
    )
    environment = object.__new__(NexAUE2BEnvironment)
    environment.session_id = "trial-session"
    environment.logger = logging.getLogger("test-e2b-stop")
    environment._sandbox = sandbox
    environment._max_lifetime_sec = E2B_DEFAULT_MAX_LIFETIME_SEC
    environment._keepalive_task = None
    environment._keepalive_deadline = None
    environment._start_keepalive()
    keepalive_task = environment._keepalive_task

    async def kill() -> bool:
        assert keepalive_task.done()
        return True

    sandbox.kill.side_effect = kill

    await asyncio.sleep(0)
    await environment.stop(delete=True)

    assert keepalive_task.cancelled()
    assert environment._keepalive_task is None
    assert environment._sandbox is None
    sandbox.kill.assert_awaited_once_with()
