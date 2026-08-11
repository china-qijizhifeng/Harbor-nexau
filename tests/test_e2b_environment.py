from __future__ import annotations

from unittest.mock import AsyncMock

from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.utils.import_path import import_class

from nexau_harbor import e2b_environment
from nexau_harbor.e2b_environment import (
    E2B_SANDBOX_TIMEOUT_SEC,
    NexAUE2BEnvironment,
)


def test_custom_environment_import_path_is_loadable() -> None:
    environment_class = import_class(
        "nexau_harbor.e2b_environment:NexAUE2BEnvironment",
        label="environment",
    )

    assert environment_class is NexAUE2BEnvironment


async def test_create_sandbox_uses_one_hour_timeout(monkeypatch) -> None:
    sandbox = object()
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(
        e2b_environment.harbor_e2b.AsyncSandbox,
        "create",
        create,
    )

    environment = object.__new__(NexAUE2BEnvironment)
    environment.environment_name = "terminal-bench-task"
    environment.session_id = "trial-session"
    environment._template_name = "terminal-bench-template"
    environment.task_env_config = EnvironmentConfig(env={"TASK_ENV": "task"})
    environment._persistent_env = {"TRIAL_ENV": "trial"}
    environment._network_policy = NetworkPolicy(network_mode=NetworkMode.PUBLIC)

    await environment._create_sandbox()

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
    assert E2B_SANDBOX_TIMEOUT_SEC == 3_600
