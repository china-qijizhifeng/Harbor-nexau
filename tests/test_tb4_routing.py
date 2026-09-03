from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexau_harbor.tb4_routing import (
    TB4RoutingError,
    discover_routes,
    main,
    materialize_route_views,
)


def _create_task(
    root: Path,
    name: str,
    *,
    agent_gpus: int = 0,
    agent_gpu_types: tuple[str, ...] = (),
    verifier_gpus: int | None = None,
    verifier_gpu_types: tuple[str, ...] = (),
    compose: bool = False,
) -> Path:
    task = root / name
    environment = task / "environment"
    tests = task / "tests"
    environment.mkdir(parents=True)
    tests.mkdir()
    (task / "instruction.md").write_text("Solve it.\n", encoding="utf-8")
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    (tests / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    if compose:
        (environment / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")

    def gpu_types(values: tuple[str, ...]) -> str:
        return json.dumps(list(values))

    verifier_environment = ""
    if verifier_gpus is not None:
        verifier_environment = (
            "\n[verifier.environment]\n"
            "cpus = 4\n"
            "memory_mb = 8192\n"
            "storage_mb = 20480\n"
            f"gpus = {verifier_gpus}\n"
            f"gpu_types = {gpu_types(verifier_gpu_types)}\n"
        )
    (task / "task.toml").write_text(
        (
            f'[task]\nname = "terminal-bench/{name}"\n\n'
            '[verifier]\nenvironment_mode = "separate"\n'
            f"{verifier_environment}\n"
            "[environment]\n"
            "cpus = 8\n"
            "memory_mb = 16384\n"
            "storage_mb = 30720\n"
            f"gpus = {agent_gpus}\n"
            f"gpu_types = {gpu_types(agent_gpu_types)}\n"
        ),
        encoding="utf-8",
    )
    return task


def test_discover_routes_splits_cpu_and_h100_tasks(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _create_task(dataset, "cpu-compose", compose=True)
    _create_task(dataset, "gpu", agent_gpus=1, agent_gpu_types=("H100",))

    routes = discover_routes(dataset)

    assert [(route.dataset_name, route.route) for route in routes] == [
        ("cpu-compose", "cpu"),
        ("gpu", "basestream"),
    ]
    gpu = routes[1]
    assert [resource.label for resource in gpu.environments] == ["agent", "verifier"]
    assert all(resource.gpus == 1 for resource in gpu.environments)
    assert all(resource.gpu_types == ("H100",) for resource in gpu.environments)


def test_discover_routes_rejects_non_h100_gpu(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _create_task(dataset, "gpu", agent_gpus=1, agent_gpu_types=("A100",))

    with pytest.raises(TB4RoutingError, match="explicit gpu_types"):
        discover_routes(dataset)


def test_discover_routes_rejects_gpu_only_verifier(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _create_task(
        dataset,
        "mixed",
        verifier_gpus=1,
        verifier_gpu_types=("H100",),
    )

    with pytest.raises(TB4RoutingError, match="GPU-only verifier"):
        discover_routes(dataset)


def test_discover_routes_rejects_cpu_only_separate_verifier_for_gpu_agent(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _create_task(
        dataset,
        "mixed",
        agent_gpus=1,
        agent_gpu_types=("H100",),
        verifier_gpus=0,
    )

    with pytest.raises(TB4RoutingError, match="missing GPU"):
        discover_routes(dataset)


def test_materialize_views_keeps_original_tasks_unchanged(tmp_path: Path) -> None:
    pvc = tmp_path / "pvc"
    dataset = pvc / "dataset"
    dataset.mkdir(parents=True)
    cpu = _create_task(dataset, "cpu")
    gpu = _create_task(dataset, "gpu", agent_gpus=1, agent_gpu_types=("H100",))
    routes = discover_routes(dataset)

    views = materialize_route_views(routes, pvc / "routes", gpu_host_root=pvc)

    assert (views["cpu"] / "cpu").is_symlink()
    assert (views["cpu"] / "cpu").resolve() == cpu.resolve()
    assert (views["basestream"] / "gpu").is_symlink()
    assert (views["basestream"] / "gpu").resolve() == gpu.resolve()
    assert sorted(path.name for path in dataset.iterdir()) == ["cpu", "gpu"]


def test_materialize_views_rejects_gpu_task_outside_pvc(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _create_task(dataset, "gpu", agent_gpus=1, agent_gpu_types=("H100",))
    routes = discover_routes(dataset)
    pvc = tmp_path / "other-pvc"
    pvc.mkdir()

    with pytest.raises(TB4RoutingError, match="outside the Basestream PVC"):
        materialize_route_views(routes, pvc / "routes", gpu_host_root=pvc)


def test_cli_writes_auditable_plan_and_machine_readable_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pvc = tmp_path / "pvc"
    dataset = pvc / "dataset"
    dataset.mkdir(parents=True)
    _create_task(dataset, "cpu")
    _create_task(dataset, "gpu", agent_gpus=1, agent_gpu_types=("H100",))
    plan_path = pvc / "plan.json"

    result = main(
        [
            "--dataset",
            str(dataset),
            "--materialize-root",
            str(pvc / "views"),
            "--output",
            str(plan_path),
            "--gpu-host-root",
            str(pvc),
        ]
    )

    assert result == 0
    fields = capsys.readouterr().out.strip().split("\t")
    assert fields[:2] == ["1", "1"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema_version"] == "harbor-nexau-tb4-routing-v1"
    assert plan["counts"] == {"basestream": 1, "cpu": 1}
    assert {task["route"] for task in plan["tasks"]} == {"cpu", "basestream"}
