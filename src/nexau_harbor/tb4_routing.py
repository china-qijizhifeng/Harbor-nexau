"""Plan TB4 CPU/E2B and H100/Basestream execution without editing tasks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config

Route = Literal["cpu", "basestream"]
_COMPOSE_FILENAMES = (
    "docker-compose.yaml",
    "docker-compose.yml",
    "compose.yaml",
    "compose.yml",
)


class TB4RoutingError(RuntimeError):
    """The local task set cannot be routed without changing its semantics."""


@dataclass(frozen=True)
class EnvironmentResources:
    label: str
    cpus: int
    memory_mb: int
    storage_mb: int
    gpus: int
    gpu_types: tuple[str, ...]

    @classmethod
    def from_config(cls, label: str, config: EnvironmentConfig) -> EnvironmentResources:
        return cls(
            label=label,
            cpus=int(config.cpus or 1),
            memory_mb=int(config.memory_mb or 2048),
            storage_mb=int(config.storage_mb or 10240),
            gpus=int(config.gpus or 0),
            gpu_types=tuple(str(item) for item in (config.gpu_types or ())),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "gpus": self.gpus,
            "gpu_types": list(self.gpu_types),
        }


@dataclass(frozen=True)
class TaskRoute:
    dataset_name: str
    task_name: str
    selector: str
    source_path: Path
    route: Route
    environments: tuple[EnvironmentResources, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "selector": self.selector,
            "source_path": self.source_path.as_posix(),
            "route": self.route,
            "environments": [resource.as_dict() for resource in self.environments],
        }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_compose(path: Path) -> bool:
    return any((path / filename).is_file() for filename in _COMPOSE_FILENAMES)


def _environment_contracts(task: Task) -> tuple[EnvironmentResources, ...]:
    contracts = [EnvironmentResources.from_config("agent", task.config.environment)]
    if task.config.steps:
        for step in task.config.steps:
            verifier = resolve_effective_verifier_env_config(task.config, step)
            if verifier is not None:
                contracts.append(
                    EnvironmentResources.from_config(f"verifier:{step.name}", verifier)
                )
    else:
        verifier = resolve_effective_verifier_env_config(task.config, None)
        if verifier is not None:
            contracts.append(EnvironmentResources.from_config("verifier", verifier))
    return tuple(contracts)


def _validate_gpu_contract(task: Task, contracts: tuple[EnvironmentResources, ...]) -> None:
    agent = contracts[0]
    if agent.gpus < 1:
        raise TB4RoutingError(
            f"{task.name}: a GPU-only verifier with a CPU agent cannot use the current "
            "single-provider Harbor routing contract"
        )
    cpu_only_environments = [contract.label for contract in contracts if contract.gpus < 1]
    if cpu_only_environments:
        raise TB4RoutingError(
            f"{task.name}: Basestream-routed separate environments must also request an "
            f"H100; missing GPU in {cpu_only_environments!r}"
        )
    for contract in contracts:
        if contract.gpus < 1:
            continue
        normalized = tuple(value.upper() for value in contract.gpu_types)
        if not normalized or any(value != "H100" for value in normalized):
            raise TB4RoutingError(
                f"{task.name} {contract.label}: Basestream routing requires explicit "
                f"gpu_types=['H100'], got {contract.gpu_types!r}"
            )
    if _has_compose(task.paths.environment_dir):
        raise TB4RoutingError(
            f"{task.name}: BasestreamEnvironment does not support Compose GPU tasks"
        )
    if task.config.steps:
        for step in task.config.steps:
            verifier = resolve_effective_verifier_env_config(task.config, step)
            if verifier is None:
                continue
            context = task.paths.step_tests_dir(step.name)
            if context.exists() and _has_compose(context):
                raise TB4RoutingError(f"{task.name} verifier {step.name}: Compose is not supported")
    elif any(resource.label == "verifier" for resource in contracts) and _has_compose(
        task.paths.tests_dir
    ):
        raise TB4RoutingError(f"{task.name} verifier: Compose is not supported")


def discover_routes(dataset_path: Path) -> list[TaskRoute]:
    """Classify every direct task entry in a resolved local dataset."""

    dataset = dataset_path.expanduser().resolve()
    if not dataset.is_dir():
        raise TB4RoutingError(f"dataset directory does not exist: {dataset}")
    routes: list[TaskRoute] = []
    selectors: set[str] = set()
    for entry in sorted(dataset.iterdir(), key=lambda path: path.name):
        if not entry.is_dir() or not (entry / "task.toml").is_file():
            continue
        if any(character in entry.name for character in "\t\r\n"):
            raise TB4RoutingError(f"task directory has an unsafe name: {entry.name!r}")
        task = Task(entry, disable_verification=False)
        contracts = _environment_contracts(task)
        has_gpu = any(resource.gpus > 0 for resource in contracts)
        if has_gpu:
            _validate_gpu_contract(task, contracts)
        selector = task.task_dir.name
        if selector in selectors:
            raise TB4RoutingError(f"duplicate resolved task selector: {selector!r}")
        selectors.add(selector)
        routes.append(
            TaskRoute(
                dataset_name=entry.name,
                task_name=task.name,
                selector=selector,
                source_path=entry.resolve(),
                route="basestream" if has_gpu else "cpu",
                environments=contracts,
            )
        )
    if not routes:
        raise TB4RoutingError(f"no Harbor tasks found directly under dataset: {dataset}")
    return routes


def materialize_route_views(
    routes: list[TaskRoute],
    output_root: Path,
    *,
    gpu_host_root: Path | None = None,
) -> dict[Route, Path]:
    """Create immutable symlink views consumed by independent Harbor child jobs."""

    root = output_root.expanduser()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise TB4RoutingError(f"route output must be an empty directory: {root}")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    views: dict[Route, Path] = {
        "cpu": root / "cpu",
        "basestream": root / "basestream",
    }
    for view in views.values():
        view.mkdir(mode=0o700)

    resolved_gpu_root = gpu_host_root.expanduser().resolve() if gpu_host_root else None
    for route in routes:
        source = route.source_path.resolve()
        if route.route == "basestream" and resolved_gpu_root is not None:
            if not _is_relative_to(source, resolved_gpu_root):
                raise TB4RoutingError(
                    f"GPU task {route.task_name} resolves outside the Basestream PVC host "
                    f"root {resolved_gpu_root}: {source}"
                )
        (views[route.route] / route.dataset_name).symlink_to(source, target_is_directory=True)
    return views


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--materialize-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-host-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        routes = discover_routes(args.dataset)
        views = materialize_route_views(
            routes,
            args.materialize_root,
            gpu_host_root=args.gpu_host_root,
        )
        cpu_count = sum(route.route == "cpu" for route in routes)
        basestream_count = len(routes) - cpu_count
        payload = {
            "schema_version": "harbor-nexau-tb4-routing-v1",
            "dataset": args.dataset.expanduser().resolve().as_posix(),
            "counts": {"cpu": cpu_count, "basestream": basestream_count},
            "views": {key: value.resolve().as_posix() for key, value in views.items()},
            "tasks": [route.as_dict() for route in routes],
        }
        _write_json_atomic(args.output, payload)
    except (OSError, ValueError, TB4RoutingError) as exc:
        raise SystemExit(f"TB4 routing failed: {exc}") from exc

    fields = (
        str(cpu_count),
        str(basestream_count),
        views["cpu"].resolve().as_posix(),
        views["basestream"].resolve().as_posix(),
    )
    if any(any(character in field for character in "\t\r\n") for field in fields):
        raise SystemExit("TB4 routing output contains an unsafe path")
    print("\t".join(fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
