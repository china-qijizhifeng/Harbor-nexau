from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.utils.import_path import import_class

from nexau_harbor import e2b_environment
from nexau_harbor.e2b_environment import (
    E2B_KEEPALIVE_INTERVAL_SEC,
    E2B_SANDBOX_TIMEOUT_SEC,
    NexAUE2BEnvironment,
)


def test_custom_environment_import_path_is_loadable() -> None:
    environment_class = import_class(
        "nexau_harbor.e2b_environment:NexAUE2BEnvironment",
        label="environment",
    )

    assert environment_class is NexAUE2BEnvironment


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
