"""Discover and prewarm Docker Hub images for selected E2B DinD tasks."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fnmatch
import json
import os
import re
import shlex
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from nexau_harbor.e2b_dind_environment import (
    E2B_DIND_DEFAULT_TEMPLATE,
    E2B_DIND_TEMPLATE_ENV,
)
from nexau_harbor.e2b_dind_template import E2BConnection, load_e2b_connection

DEFAULT_CONCURRENCY = 2
DEFAULT_PULL_TIMEOUT_SEC = 60
DEFAULT_PULL_ATTEMPTS = 1
DEFAULT_SANDBOX_TIMEOUT_SEC = 3_600
_READY_TIMEOUT_SEC = 180
_KEEPALIVE_INTERVAL_SEC = 60
_PULL_KILL_GRACE_SEC = 30
_PULL_RETRY_BASE_DELAY_SEC = 5
_OUTPUT_TAIL_CHARS = 2_000
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_FILENAMES = {
    "compose.yaml",
    "compose.yml",
    "docker-compose.yaml",
    "docker-compose.yml",
}
_VARIABLE_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:(?P<operator>:-|-)(?P<default>[^}]*))?\}|"
    r"(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
_ARG_PATTERN = re.compile(
    r"^ARG\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:=(?P<default>.*))?$",
    re.IGNORECASE,
)
_COPY_FROM_PATTERN = re.compile(r"(?:^|\s)--from=(?P<source>[^\s]+)", re.IGNORECASE)
_HEREDOC_PATTERN = re.compile(r"<<-?\s*['\"]?(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)['\"]?")


@dataclass(frozen=True)
class PrewarmPlan:
    task_paths: tuple[Path, ...]
    images: tuple[str, ...]
    skipped_external_images: tuple[str, ...]
    unresolved_references: tuple[str, ...]


@dataclass(frozen=True)
class EnvironmentImagePlan:
    images: tuple[str, ...]
    skipped_external_images: tuple[str, ...]
    unresolved_references: tuple[str, ...]


@dataclass(frozen=True)
class PullFailure:
    image: str
    attempts: int
    detail: str


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"{name} must be a positive integer")
    try:
        resolved = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {resolved}")
    return resolved


def _task_name(task_path: Path) -> str | None:
    try:
        task_config = tomllib.loads((task_path / "task.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    name = task_config.get("task", {}).get("name")
    return str(name) if name else None


def _task_candidates(dataset_path: Path) -> tuple[tuple[Path, tuple[str, ...]], ...]:
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset path is not a directory: {dataset_path}")

    entries = (
        [dataset_path] if (dataset_path / "task.toml").is_file() else sorted(dataset_path.iterdir())
    )
    candidates: list[tuple[Path, tuple[str, ...]]] = []
    seen: set[Path] = set()
    for entry in entries:
        if not entry.is_dir() or not (entry / "task.toml").is_file():
            continue
        resolved = entry.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        identities = {entry.name, resolved.name}
        if name := _task_name(resolved):
            identities.add(name)
            identities.add(name.rsplit("/", 1)[-1])
        candidates.append((resolved, tuple(sorted(identities))))
    if not candidates:
        raise ValueError(f"No Harbor tasks found under dataset path: {dataset_path}")
    return tuple(candidates)


def _matches_selector(identities: tuple[str, ...], selector: str) -> bool:
    return any(fnmatch.fnmatchcase(identity, selector) for identity in identities)


def _selected_task_paths(
    dataset_path: Path,
    *,
    includes: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
) -> tuple[Path, ...]:
    selected: list[Path] = []
    for task_path, identities in _task_candidates(dataset_path):
        if includes and not any(_matches_selector(identities, selector) for selector in includes):
            continue
        if any(_matches_selector(identities, selector) for selector in excludes):
            continue
        selected.append(task_path)
    if includes and not selected:
        raise ValueError(
            "No tasks matched the DinD prewarm include selector(s): " + ", ".join(includes)
        )
    return tuple(selected)


def _resolve_variables(value: str, variables: dict[str, str]) -> str | None:
    if "$$" in value:
        return None
    resolved = value
    for _ in range(5):
        unresolved = False

        def replace(match: re.Match[str]) -> str:
            nonlocal unresolved
            name = match.group("braced") or match.group("plain")
            operator = match.group("operator")
            default = match.group("default") or ""
            configured = variables.get(name)
            if operator == ":-" and not configured:
                return default
            if operator == "-" and configured is None:
                return default
            if configured is None:
                unresolved = True
                return match.group(0)
            return configured

        updated = _VARIABLE_PATTERN.sub(replace, resolved)
        if unresolved:
            return None
        if updated == resolved:
            return updated if "$" not in updated else None
        resolved = updated
    return resolved if "$" not in resolved else None


def _dockerfile_instructions(content: str) -> tuple[str, ...]:
    instructions: list[str] = []
    buffered = ""
    heredoc_delimiters: list[str] = []
    for raw_line in content.splitlines():
        if heredoc_delimiters:
            if raw_line.strip() == heredoc_delimiters[0]:
                heredoc_delimiters.pop(0)
            continue
        stripped = raw_line.strip()
        if not buffered and (not stripped or stripped.startswith("#")):
            continue
        continuation = stripped.endswith("\\")
        fragment = stripped[:-1].rstrip() if continuation else stripped
        buffered = f"{buffered} {fragment}".strip()
        if not continuation:
            instructions.append(buffered)
            heredoc_delimiters.extend(
                match.group("delimiter") for match in _HEREDOC_PATTERN.finditer(buffered)
            )
            buffered = ""
    if buffered:
        instructions.append(buffered)
    return tuple(instructions)


def _docker_hub_image(image: str) -> str | None:
    normalized = image.strip()
    if not normalized or normalized.lower() == "scratch" or "$" in normalized:
        return None
    parts = normalized.split("/", 1)
    if len(parts) == 1:
        return normalized
    first = parts[0].lower()
    if first in {"docker.io", "index.docker.io", "registry-1.docker.io"}:
        return normalized
    if "." in first or ":" in first or first == "localhost":
        return None
    return normalized


def docker_hub_official_reference(image: str) -> str:
    """Return an explicit Docker Hub Registry API reference that bypasses mirrors."""

    normalized = image.strip()
    for prefix in ("registry-1.docker.io/", "docker.io/", "index.docker.io/"):
        if normalized.startswith(prefix):
            path = normalized.removeprefix(prefix)
            break
    else:
        path = normalized
    if "/" not in path:
        path = f"library/{path}"
    return f"registry-1.docker.io/{path}"


def docker_hub_local_reference(image: str) -> str:
    """Return the local tag expected by an unqualified Dockerfile/Compose reference."""

    normalized = image.strip()
    for prefix in ("registry-1.docker.io/", "docker.io/", "index.docker.io/"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    if normalized.startswith("library/"):
        normalized = normalized.removeprefix("library/")
    return normalized.split("@", 1)[0]


def _dockerfile_images(path: Path) -> tuple[set[str], set[str], set[str]]:
    images: set[str] = set()
    external: set[str] = set()
    unresolved: set[str] = set()
    variables: dict[str, str] = {}
    stage_aliases: set[str] = set()
    try:
        instructions = _dockerfile_instructions(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return images, external, {f"{path}: {exc}"}

    for instruction in instructions:
        if arg_match := _ARG_PATTERN.match(instruction):
            if arg_match.group("default") is not None:
                variables[arg_match.group("name")] = arg_match.group("default").strip()
            continue

        try:
            tokens = shlex.split(instruction, posix=True)
        except ValueError:
            unresolved.add(f"{path}: {instruction}")
            continue
        if not tokens:
            continue

        if tokens[0].upper() == "FROM":
            image_index = 1
            while image_index < len(tokens) and tokens[image_index].startswith("--"):
                image_index += 1
            if image_index >= len(tokens):
                unresolved.add(f"{path}: {instruction}")
                continue
            raw_image = tokens[image_index]
            resolved_image = _resolve_variables(raw_image, variables)
            if resolved_image is None:
                unresolved.add(f"{path}: {raw_image}")
            elif resolved_image.lower() not in stage_aliases:
                if hub_image := _docker_hub_image(resolved_image):
                    images.add(hub_image)
                elif resolved_image.lower() != "scratch":
                    external.add(resolved_image)
            if image_index + 2 < len(tokens) and tokens[image_index + 1].upper() == "AS":
                stage_aliases.add(tokens[image_index + 2].lower())
            continue

        if tokens[0].upper() != "COPY":
            continue
        copy_from = _COPY_FROM_PATTERN.search(instruction)
        if copy_from is None:
            continue
        raw_source = copy_from.group("source")
        if raw_source.isdigit():
            continue
        resolved_source = _resolve_variables(raw_source, variables)
        if resolved_source is None:
            unresolved.add(f"{path}: {raw_source}")
        elif resolved_source.lower() not in stage_aliases:
            if hub_image := _docker_hub_image(resolved_source):
                images.add(hub_image)
            else:
                external.add(resolved_source)

    return images, external, unresolved


def _compose_variables(environment_dir: Path) -> dict[str, str]:
    variables = {
        name: value
        for name, value in dotenv_values(environment_dir / ".env").items()
        if value is not None
    }
    variables.update(os.environ)
    return variables


def _compose_images(path: Path, variables: dict[str, str]) -> tuple[set[str], set[str], set[str]]:
    images: set[str] = set()
    external: set[str] = set()
    unresolved: set[str] = set()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return images, external, {f"{path}: {exc}"}
    if not isinstance(document, dict) or not isinstance(document.get("services"), dict):
        return images, external, unresolved

    for service_name, service in document["services"].items():
        if not isinstance(service, dict) or "build" in service or "image" not in service:
            continue
        raw_image = service["image"]
        if not isinstance(raw_image, str):
            unresolved.add(f"{path}: service {service_name!r} has a non-string image")
            continue
        resolved_image = _resolve_variables(raw_image, variables)
        if resolved_image is None:
            unresolved.add(f"{path}: service {service_name!r}: {raw_image}")
        elif hub_image := _docker_hub_image(resolved_image):
            images.add(hub_image)
        else:
            external.add(resolved_image)
    return images, external, unresolved


def discover_prewarm_plan(
    dataset_path: Path | str,
    *,
    includes: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
) -> PrewarmPlan:
    task_paths = _selected_task_paths(Path(dataset_path), includes=includes, excludes=excludes)
    images: set[str] = set()
    external: set[str] = set()
    unresolved: set[str] = set()
    for task_path in task_paths:
        environment_dir = task_path / "environment"
        if not environment_dir.is_dir():
            continue
        environment_plan = discover_environment_images(environment_dir)
        images.update(environment_plan.images)
        external.update(environment_plan.skipped_external_images)
        unresolved.update(environment_plan.unresolved_references)
    return PrewarmPlan(
        task_paths=task_paths,
        images=tuple(sorted(images)),
        skipped_external_images=tuple(sorted(external)),
        unresolved_references=tuple(sorted(unresolved)),
    )


def discover_environment_images(
    environment_dir: Path | str,
    *,
    extra_variables: dict[str, str] | None = None,
) -> EnvironmentImagePlan:
    environment_path = Path(environment_dir)
    images: set[str] = set()
    external: set[str] = set()
    unresolved: set[str] = set()
    for dockerfile in sorted(environment_path.rglob("Dockerfile*")):
        if not dockerfile.is_file():
            continue
        found, skipped, missing = _dockerfile_images(dockerfile)
        images.update(found)
        external.update(skipped)
        unresolved.update(missing)
    variables = _compose_variables(environment_path)
    variables.update(extra_variables or {})
    for compose_path in sorted(environment_path.rglob("*")):
        if not compose_path.is_file() or compose_path.name not in _COMPOSE_FILENAMES:
            continue
        found, skipped, missing = _compose_images(compose_path, variables)
        images.update(found)
        external.update(skipped)
        unresolved.update(missing)
    return EnvironmentImagePlan(
        images=tuple(sorted(images)),
        skipped_external_images=tuple(sorted(external)),
        unresolved_references=tuple(sorted(unresolved)),
    )


def _load_async_sandbox_api() -> Any:
    try:
        from e2b import AsyncSandbox
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise RuntimeError(
            "E2B SDK is not installed; run `uv sync --locked --extra cloud`"
        ) from exc
    return AsyncSandbox


def _failure_detail(exc: Exception) -> str:
    stdout = str(getattr(exc, "stdout", "") or "")
    stderr = str(getattr(exc, "stderr", "") or "")
    exit_code = getattr(exc, "exit_code", None)
    details = "\n".join(part for part in (stderr.rstrip(), stdout.rstrip()) if part)
    if not details:
        details = str(exc)
    prefix = f"exit code {exit_code}: " if exit_code is not None else ""
    return f"{prefix}{details[-_OUTPUT_TAIL_CHARS:]}"


async def _pull_image(
    sandbox: Any,
    image: str,
    *,
    timeout_sec: int,
    attempts: int,
) -> PullFailure | None:
    last_detail = "no diagnostic"
    for attempt in range(1, attempts + 1):
        print(f"Prewarming {image} (attempt {attempt}/{attempts})", file=sys.stderr, flush=True)
        command = (
            f"timeout --signal=TERM --kill-after={_PULL_KILL_GRACE_SEC} "
            f"{timeout_sec} docker pull -- {shlex.quote(image)}; "
            "harbor_nexau_pull_rc=$?; "
            'if test "$harbor_nexau_pull_rc" -eq 0; then '
            f"docker image rm -- {shlex.quote(image)} >/dev/null 2>&1 || true; "
            'fi; exit "$harbor_nexau_pull_rc"'
        )
        try:
            result = await sandbox.commands.run(
                command,
                user="root",
                timeout=timeout_sec + _PULL_KILL_GRACE_SEC + 10,
                request_timeout=timeout_sec + _PULL_KILL_GRACE_SEC + 40,
            )
            if result.exit_code == 0:
                print(f"Prewarmed {image}", file=sys.stderr, flush=True)
                return None
            last_detail = (
                result.stderr or result.stdout or f"docker pull exited {result.exit_code}"
            )[-_OUTPUT_TAIL_CHARS:]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_detail = _failure_detail(exc)
        if attempt < attempts:
            print(
                f"Prewarm attempt {attempt}/{attempts} failed for {image}: "
                f"{last_detail[-_OUTPUT_TAIL_CHARS:]}",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(_PULL_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)))
    return PullFailure(image=image, attempts=attempts, detail=last_detail)


async def _keep_sandbox_alive(sandbox: Any) -> None:
    while True:
        await asyncio.sleep(_KEEPALIVE_INTERVAL_SEC)
        try:
            await sandbox.set_timeout(DEFAULT_SANDBOX_TIMEOUT_SEC)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(
                f"Failed to refresh DinD prewarm sandbox lease; retrying: {exc}",
                file=sys.stderr,
                flush=True,
            )


async def prewarm_images(
    images: tuple[str, ...],
    connection: E2BConnection,
    *,
    template: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout_sec: int = DEFAULT_PULL_TIMEOUT_SEC,
    attempts: int = DEFAULT_PULL_ATTEMPTS,
) -> None:
    concurrency = _positive_int(concurrency, name="concurrency")
    timeout_sec = _positive_int(timeout_sec, name="timeout_sec")
    attempts = _positive_int(attempts, name="attempts")
    if not images:
        return

    sandbox_api = _load_async_sandbox_api()
    sandbox = None
    keepalive_task: asyncio.Task[None] | None = None
    try:
        sandbox = await sandbox_api.create(
            template,
            timeout=DEFAULT_SANDBOX_TIMEOUT_SEC,
            metadata={
                "purpose": "harbor-nexau-dind-prewarm",
                "image_count": str(len(images)),
            },
            **connection.sdk_options(),
        )
        keepalive_task = asyncio.create_task(_keep_sandbox_alive(sandbox))
        ready = await sandbox.commands.run(
            "harbor-nexau-dind-ready",
            user="root",
            timeout=_READY_TIMEOUT_SEC,
            request_timeout=_READY_TIMEOUT_SEC + 30,
        )
        if ready.exit_code != 0:
            raise RuntimeError(
                "E2B DinD prewarm sandbox did not become ready: "
                + (ready.stderr or ready.stdout or f"exit code {ready.exit_code}")
            )

        semaphore = asyncio.Semaphore(min(concurrency, len(images)))

        async def pull(image: str) -> PullFailure | None:
            async with semaphore:
                return await _pull_image(
                    sandbox,
                    image,
                    timeout_sec=timeout_sec,
                    attempts=attempts,
                )

        failures = [
            failure
            for failure in await asyncio.gather(*(pull(image) for image in images))
            if failure
        ]
        if failures:
            detail = "\n".join(
                f"- {failure.image} after {failure.attempts} attempts: {failure.detail}"
                for failure in failures
            )
            raise RuntimeError(f"DinD image prewarm failed for {len(failures)} image(s):\n{detail}")
    finally:
        if keepalive_task is not None:
            keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keepalive_task
        if sandbox is not None:
            sandbox_id = sandbox.sandbox_id
            try:
                await sandbox.kill()
            except Exception as exc:
                print(
                    f"Failed to destroy DinD prewarm sandbox {sandbox_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            else:
                print(f"Destroyed DinD prewarm sandbox {sandbox_id}", file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument(
        "--template",
        default=os.environ.get(E2B_DIND_TEMPLATE_ENV, E2B_DIND_DEFAULT_TEMPLATE),
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_PULL_TIMEOUT_SEC)
    parser.add_argument("--attempts", type=int, default=DEFAULT_PULL_ATTEMPTS)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(os.environ.get("HARBOR_NEXAU_ENV_FILE", _PROJECT_ROOT / ".env")),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = discover_prewarm_plan(
            args.dataset,
            includes=tuple(args.include),
            excludes=tuple(args.exclude),
        )
        summary = {
            "task_count": len(plan.task_paths),
            "images": list(plan.images),
            "skipped_external_images": list(plan.skipped_external_images),
            "unresolved_references": list(plan.unresolved_references),
        }
        if args.dry_run:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        print(
            f"Discovered {len(plan.images)} Docker Hub image(s) across "
            f"{len(plan.task_paths)} selected DinD task(s)",
            file=sys.stderr,
        )
        for image in plan.images:
            print(f"  {image}", file=sys.stderr)
        if plan.skipped_external_images:
            print(
                "Skipping non-Docker-Hub image(s): " + ", ".join(plan.skipped_external_images),
                file=sys.stderr,
            )
        if plan.unresolved_references:
            print(
                f"Skipping {len(plan.unresolved_references)} unresolved image reference(s)",
                file=sys.stderr,
            )
        if not plan.images:
            print("No Docker Hub images require prewarming", file=sys.stderr)
            return 0
        connection = load_e2b_connection(args.env_file)
        asyncio.run(
            prewarm_images(
                plan.images,
                connection,
                template=args.template,
                concurrency=args.concurrency,
                timeout_sec=args.timeout_sec,
                attempts=args.attempts,
            )
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"DinD prewarm failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
