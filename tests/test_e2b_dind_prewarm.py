from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexau_harbor import e2b_dind_prewarm
from nexau_harbor.e2b_dind_prewarm import discover_prewarm_plan, prewarm_images
from nexau_harbor.e2b_dind_template import E2BConnection


def _task(task_store: Path, digest: str, name: str) -> Path:
    task = task_store / digest
    task.mkdir()
    (task / "task.toml").write_text(
        f'[task]\nname = "frontier-bench/{name}"\n',
        encoding="utf-8",
    )
    (task / "environment").mkdir()
    return task


def test_discovery_selects_tasks_and_finds_docker_hub_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POSTGRES_TAG", raising=False)
    task_store = tmp_path / "task-store"
    task_store.mkdir()
    selected = _task(task_store, "selected-digest", "selected")
    skipped = _task(task_store, "skipped-digest", "skipped")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "selected-human").symlink_to(selected, target_is_directory=True)
    (dataset / "skipped-human").symlink_to(skipped, target_is_directory=True)

    environment = selected / "environment"
    (environment / ".env").write_text("POSTGRES_TAG=18\n", encoding="utf-8")
    (environment / "Dockerfile").write_text(
        """ARG PYTHON_TAG=3.12-slim
FROM python:${PYTHON_TAG} AS builder
FROM builder AS final
COPY --from=0 /tmp/a /tmp/a
COPY --from=ghcr.io/example/tool:1 /tool /tool
RUN <<'PY'
from pathlib import Path
PY
""",
        encoding="utf-8",
    )
    nested = environment / "loadgen"
    nested.mkdir()
    (nested / "Dockerfile.worker").write_text(
        "FROM rust:1-slim-bookworm\nFROM ${UNRESOLVED_BASE}\n",
        encoding="utf-8",
    )
    (environment / "docker-compose.yaml").write_text(
        """services:
  db:
    image: postgres:${POSTGRES_TAG:-17}
  app:
    build: .
    image: local/app:latest
  browser:
    image: mcr.microsoft.com/playwright:v1
""",
        encoding="utf-8",
    )
    (skipped / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n",
        encoding="utf-8",
    )

    plan = discover_prewarm_plan(dataset, includes=("selected-digest",))

    assert plan.task_paths == (selected,)
    assert plan.images == ("postgres:18", "python:3.12-slim", "rust:1-slim-bookworm")
    assert plan.skipped_external_images == (
        "ghcr.io/example/tool:1",
        "mcr.microsoft.com/playwright:v1",
    )
    assert len(plan.unresolved_references) == 1
    assert "UNRESOLVED_BASE" in plan.unresolved_references[0]


def test_discovery_applies_name_glob_exclusions(tmp_path: Path) -> None:
    task_store = tmp_path / "task-store"
    task_store.mkdir()
    keep = _task(task_store, "keep-digest", "keep-me")
    skip = _task(task_store, "skip-digest", "skip-me")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "keep-human").symlink_to(keep, target_is_directory=True)
    (dataset / "skip-human").symlink_to(skip, target_is_directory=True)
    (keep / "environment" / "Dockerfile").write_text("FROM fedora:42\n", encoding="utf-8")
    (skip / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")

    plan = discover_prewarm_plan(
        dataset,
        includes=("frontier-bench/*",),
        excludes=("*skip-me",),
    )

    assert plan.task_paths == (keep,)
    assert plan.images == ("fedora:42",)


class _FakeSandbox:
    def __init__(self, *, fail_pulls: bool = False) -> None:
        self.sandbox_id = "prewarm-sandbox"
        self.fail_pulls = fail_pulls
        self.active_pulls = 0
        self.max_active_pulls = 0
        self.pull_commands: list[str] = []
        self.kill = AsyncMock()
        self.set_timeout = AsyncMock()
        self.commands = SimpleNamespace(run=self.run)

    async def run(self, command: str, **kwargs) -> SimpleNamespace:  # noqa: ANN003
        del kwargs
        if command == "harbor-nexau-dind-ready":
            return SimpleNamespace(exit_code=0, stdout="ready", stderr="")
        self.pull_commands.append(command)
        self.active_pulls += 1
        self.max_active_pulls = max(self.max_active_pulls, self.active_pulls)
        await asyncio.sleep(0)
        self.active_pulls -= 1
        if self.fail_pulls:
            raise _PullError("registry unavailable")
        return SimpleNamespace(exit_code=0, stdout="pulled", stderr="")


class _PullError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.exit_code = 1
        self.stdout = ""
        self.stderr = message


async def test_prewarm_uses_bounded_concurrency_and_destroys_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeSandbox()

    class FakeSandboxApi:
        create = AsyncMock(return_value=sandbox)

    monkeypatch.setattr(e2b_dind_prewarm, "_load_async_sandbox_api", lambda: FakeSandboxApi)

    await prewarm_images(
        ("fedora:42", "python:3.12-slim", "ubuntu:24.04"),
        E2BConnection(api_key="test-key"),
        template="example/dind:v1",
        concurrency=2,
        timeout_sec=60,
        attempts=1,
    )

    assert sandbox.max_active_pulls == 2
    assert len(sandbox.pull_commands) == 3
    assert all("docker pull" in command for command in sandbox.pull_commands)
    assert all("docker image rm" in command for command in sandbox.pull_commands)
    sandbox.kill.assert_awaited_once_with()


async def test_pull_retries_a_failed_image(monkeypatch: pytest.MonkeyPatch) -> None:
    command_run = AsyncMock(
        side_effect=[
            _PullError("temporary registry failure"),
            SimpleNamespace(exit_code=0, stdout="pulled", stderr=""),
        ]
    )
    sandbox = SimpleNamespace(commands=SimpleNamespace(run=command_run))
    sleep = AsyncMock()
    monkeypatch.setattr(e2b_dind_prewarm.asyncio, "sleep", sleep)

    failure = await e2b_dind_prewarm._pull_image(
        sandbox,
        "fedora:42",
        timeout_sec=60,
        attempts=2,
    )

    assert failure is None
    assert command_run.await_count == 2
    sleep.assert_awaited_once_with(5)


async def test_prewarm_failure_still_destroys_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeSandbox(fail_pulls=True)

    class FakeSandboxApi:
        create = AsyncMock(return_value=sandbox)

    monkeypatch.setattr(e2b_dind_prewarm, "_load_async_sandbox_api", lambda: FakeSandboxApi)

    with pytest.raises(RuntimeError, match="prewarm failed for 1 image"):
        await prewarm_images(
            ("fedora:42",),
            E2BConnection(api_key="test-key"),
            template="example/dind:v1",
            concurrency=1,
            timeout_sec=60,
            attempts=1,
        )

    sandbox.kill.assert_awaited_once_with()
