from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.environments.base import ResourceMode
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.utils.import_path import import_class

from nexau_harbor import e2b_environment
from nexau_harbor.e2b_environment import (
    E2B_DEFAULT_MAX_LIFETIME_SEC,
    E2B_KEEPALIVE_INTERVAL_SEC,
    E2B_MAX_LIFETIME_ENV,
    E2B_SANDBOX_TIMEOUT_SEC,
    NexAUE2BEnvironment,
)

UV_COPY = "COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/"


def test_custom_environment_import_path_is_loadable() -> None:
    environment_class = import_class(
        "nexau_harbor.e2b_environment:NexAUE2BEnvironment",
        label="environment",
    )

    assert environment_class is NexAUE2BEnvironment


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
    )
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
    )


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

    environment = object.__new__(NexAUE2BEnvironment)
    environment.environment_name = "terminal-bench-task"
    environment.session_id = "trial-session"
    environment._template_name = "terminal-bench-template"
    environment.task_env_config = EnvironmentConfig(env={"TASK_ENV": "task"})
    environment._persistent_env = {"TRIAL_ENV": "trial"}
    environment._network_policy = NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    environment._max_lifetime_sec = E2B_DEFAULT_MAX_LIFETIME_SEC
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
