from __future__ import annotations

from pathlib import Path

import pytest

from nexau_harbor import e2b_preflight
from nexau_harbor.e2b_preflight import (
    PreflightUnresolvedError,
    RouteDecision,
    build_route_plan,
)


def _create_dataset(tmp_path: Path, *, shell_instruction: bool = False) -> Path:
    task_store = tmp_path / "task-store"
    task_store.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    plain = task_store / "plain-digest"
    plain_environment = plain / "environment"
    plain_environment.mkdir(parents=True)
    (plain / "task.toml").write_text('version = "1.0"\n')
    (plain / "instruction.md").write_text("Do the task.\n")
    dockerfile = "FROM python:3.12-slim\n"
    if shell_instruction:
        dockerfile += 'SHELL ["/bin/bash", "-lc"]\n'
    dockerfile += "WORKDIR /app\n"
    (plain_environment / "Dockerfile").write_text(dockerfile)

    compose = task_store / "compose-digest"
    compose_environment = compose / "environment"
    compose_environment.mkdir(parents=True)
    (compose / "task.toml").write_text('version = "1.0"\n')
    (compose / "instruction.md").write_text("Do the compose task.\n")
    (compose_environment / "docker-compose.yaml").write_text("services: {}\n")

    (dataset / "plain-human").symlink_to(plain, target_is_directory=True)
    (dataset / "compose-human").symlink_to(compose, target_is_directory=True)
    return dataset


@pytest.mark.asyncio
async def test_route_plan_probes_plain_tasks_and_caches_decisions(tmp_path: Path) -> None:
    dataset = _create_dataset(tmp_path)
    cache_path = tmp_path / "routes.json"
    probed: list[str] = []

    async def probe(candidate, _ramp_delay, _timeout):
        probed.append(candidate.dataset_name)
        return RouteDecision(route="plain-e2b", reason="probe passed")

    plan = await build_route_plan(
        dataset,
        cache_path,
        concurrency=2,
        start_ramp_delay_sec=0,
        probe=probe,
    )

    assert probed == ["plain-human"]
    assert plan["plain_e2b_selectors"] == ["plain-digest"]
    assert plan["dind_selectors"] == ["compose-digest"]
    assert cache_path.is_file()

    async def unexpected_probe(*_args):
        raise AssertionError("cached plain task should not be probed again")

    cached_plan = await build_route_plan(
        dataset,
        cache_path,
        concurrency=2,
        start_ramp_delay_sec=0,
        probe=unexpected_probe,
    )

    plain_record = next(
        record for record in cached_plan["tasks"] if record["dataset_name"] == "plain-human"
    )
    assert plain_record["source"] == "cache"


@pytest.mark.asyncio
async def test_route_plan_refresh_ignores_cached_decision(tmp_path: Path) -> None:
    dataset = _create_dataset(tmp_path)
    cache_path = tmp_path / "routes.json"
    calls = 0

    async def probe(_candidate, _ramp_delay, _timeout):
        nonlocal calls
        calls += 1
        return RouteDecision(route="plain-e2b", reason=f"probe {calls}")

    for refresh in (False, True):
        await build_route_plan(
            dataset,
            cache_path,
            concurrency=1,
            start_ramp_delay_sec=0,
            refresh=refresh,
            probe=probe,
        )

    assert calls == 2


@pytest.mark.asyncio
async def test_route_cache_is_scoped_to_the_e2b_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _create_dataset(tmp_path)
    cache_path = tmp_path / "routes.json"
    calls = 0

    async def probe(_candidate, _ramp_delay, _timeout):
        nonlocal calls
        calls += 1
        return RouteDecision(route="plain-e2b", reason="probe passed")

    monkeypatch.setenv("E2B_API_URL", "https://e2b-one.example")
    await build_route_plan(
        dataset,
        cache_path,
        concurrency=1,
        start_ramp_delay_sec=0,
        probe=probe,
    )
    monkeypatch.setenv("E2B_API_URL", "https://e2b-two.example")
    await build_route_plan(
        dataset,
        cache_path,
        concurrency=1,
        start_ramp_delay_sec=0,
        probe=probe,
    )

    assert calls == 2


@pytest.mark.asyncio
async def test_unresolved_preflight_is_not_misclassified_as_dind(tmp_path: Path) -> None:
    dataset = _create_dataset(tmp_path)
    cache_path = tmp_path / "routes.json"

    async def probe(_candidate, _ramp_delay, _timeout):
        raise RuntimeError("temporary provisioning failure")

    with pytest.raises(PreflightUnresolvedError, match="plain-human"):
        await build_route_plan(
            dataset,
            cache_path,
            concurrency=1,
            start_ramp_delay_sec=0,
            probe=probe,
        )

    assert not cache_path.exists()


def test_lossy_shell_instruction_routes_to_dind_without_a_remote_probe(tmp_path: Path) -> None:
    dataset = _create_dataset(tmp_path, shell_instruction=True)
    plain_candidate = next(
        candidate
        for candidate in e2b_preflight._discover_tasks(dataset)
        if candidate.dataset_name == "plain-human"
    )

    assert (
        e2b_preflight._lossy_dockerfile_reason(plain_candidate)
        == "Dockerfile SHELL semantics are silently ignored by the E2B parser"
    )


def test_deterministic_build_error_and_transient_control_plane_error_are_separate() -> None:
    build_exception = type("BuildException", (RuntimeError,), {})

    deterministic = build_exception("invalid image reference '--platform=linux/amd64'")
    transient = build_exception("error waiting for provisioning sandbox: exit status 1")

    assert "template build rejected" in (
        e2b_preflight._deterministic_incompatibility_reason(deterministic) or ""
    )
    assert e2b_preflight._deterministic_incompatibility_reason(transient) is None
    assert e2b_preflight._is_transient_exception(transient)


@pytest.mark.asyncio
async def test_plain_e2b_probe_retries_transient_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _create_dataset(tmp_path)
    candidate = next(
        candidate
        for candidate in e2b_preflight._discover_tasks(dataset)
        if candidate.dataset_name == "plain-human"
    )
    calls = 0

    async def probe_once(_candidate, _ramp_delay, _timeout):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("temporary sandbox timeout")

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(e2b_preflight, "_probe_once", probe_once)
    monkeypatch.setattr(e2b_preflight.asyncio, "sleep", no_sleep)

    decision = await e2b_preflight._probe_plain_e2b(candidate, 0, 30, attempts=3)

    assert calls == 3
    assert decision.route == "plain-e2b"


@pytest.mark.asyncio
async def test_plain_e2b_probe_immediately_routes_deterministic_build_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _create_dataset(tmp_path)
    candidate = next(
        candidate
        for candidate in e2b_preflight._discover_tasks(dataset)
        if candidate.dataset_name == "plain-human"
    )
    build_exception = type("BuildException", (RuntimeError,), {})
    calls = 0

    async def probe_once(_candidate, _ramp_delay, _timeout):
        nonlocal calls
        calls += 1
        raise build_exception("invalid image reference")

    monkeypatch.setattr(e2b_preflight, "_probe_once", probe_once)

    decision = await e2b_preflight._probe_plain_e2b(candidate, 0, 30, attempts=3)

    assert calls == 1
    assert decision.route == "dind"


def test_selector_file_contains_one_auditable_route_per_task(tmp_path: Path) -> None:
    output = tmp_path / "selectors.tsv"
    e2b_preflight._write_selector_file(
        output,
        {
            "tasks": [
                {"route": "plain-e2b", "selector": "plain-digest"},
                {"route": "dind", "selector": "compose-digest"},
            ]
        },
    )

    assert output.read_text() == "plain-e2b\tplain-digest\ndind\tcompose-digest\n"
