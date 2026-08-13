from __future__ import annotations

import logging
from pathlib import Path
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
    _compose_env_file_content,
    _compose_project_name,
)
from nexau_harbor.e2b_environment import NexAUE2BEnvironment


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
