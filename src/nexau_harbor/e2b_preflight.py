"""Cache-backed capability preflight for routing Harbor tasks to E2B or DinD."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from uuid import uuid4

from harbor.environments.definition import environment_content_hash
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from nexau_harbor.e2b_environment import NexAUE2BEnvironment

Route = Literal["plain-e2b", "dind"]
Probe = Callable[["TaskCandidate", float, float], Awaitable["RouteDecision"]]

ROUTING_SCHEMA_VERSION = 1
DEFAULT_PREFLIGHT_TIMEOUT_SEC = 7_200.0
DEFAULT_PREFLIGHT_ATTEMPTS = 3
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "harbor-nexau" / "e2b-routes.json"

_COMPOSE_FILENAMES = (
    "docker-compose.yaml",
    "docker-compose.yml",
    "compose.yaml",
    "compose.yml",
)
_TRANSIENT_EXCEPTION_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "PoolTimeout",
    "RateLimitException",
    "ReadError",
    "ReadTimeout",
    "TimeoutError",
    "TimeoutException",
}
_TRANSIENT_MESSAGE_MARKERS = (
    "429",
    "bad gateway",
    "connection reset",
    "connection refused",
    "connection timed out",
    "could not resolve host",
    "error waiting for provisioning sandbox",
    "gateway timeout",
    "internal server error",
    "rate limit",
    "name or service not known",
    "network is unreachable",
    "sandbox timeout",
    "service unavailable",
    "temporarily unavailable",
    "temporary failure resolving",
    "tls handshake timeout",
    "too many requests",
    "timed out",
    "timeout",
    "unexpected eof",
)
_DOCKERFILE_VALUE_ERROR_MARKERS = (
    "dockerfile must contain a from instruction",
    "multi-stage dockerfiles are not supported",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskCandidate:
    """One task and the stable fields needed to route it."""

    dataset_name: str
    selector: str
    task: Task
    environment_hash: str
    cache_key: str
    has_compose: bool


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    source: Literal["cache", "preflight", "structure"] = "preflight"


class PreflightUnresolvedError(RuntimeError):
    """Raised when infrastructure failures prevent a trustworthy route decision."""

    def __init__(self, failures: dict[str, str]) -> None:
        self.failures = failures
        details = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
        super().__init__(f"E2B preflight left {len(failures)} task(s) unresolved: {details}")


def _provider_fingerprint() -> str:
    backend = "\0".join(
        (
            os.environ.get("E2B_API_URL", "").strip(),
            os.environ.get("E2B_DOMAIN", "").strip(),
        )
    )
    backend_hash = hashlib.sha256(backend.encode()).hexdigest()[:12]
    return f"route-v{ROUTING_SCHEMA_VERSION}:e2b-{version('e2b')}:backend-{backend_hash}"


def _default_cache_path() -> Path:
    configured = os.environ.get("HARBOR_NEXAU_E2B_ROUTE_CACHE")
    return Path(configured).expanduser() if configured else DEFAULT_CACHE_PATH


def _discover_tasks(dataset_path: Path) -> list[TaskCandidate]:
    dataset_path = dataset_path.expanduser().resolve()
    candidates: list[TaskCandidate] = []
    seen_selectors: set[str] = set()
    provider = _provider_fingerprint()

    for dataset_entry in sorted(dataset_path.iterdir(), key=lambda path: path.name):
        if not dataset_entry.is_dir() or not (dataset_entry / "task.toml").is_file():
            continue

        task = Task(dataset_entry, disable_verification=True)
        environment_dir = task.paths.environment_dir
        environment_hash = environment_content_hash(
            environment_dir,
            docker_image=task.config.environment.docker_image,
            truncate=64,
        )
        selector = task.task_dir.name
        if selector in seen_selectors:
            raise ValueError(
                f"Dataset {dataset_path} resolves more than one task to selector {selector!r}"
            )
        seen_selectors.add(selector)
        candidates.append(
            TaskCandidate(
                dataset_name=dataset_entry.name,
                selector=selector,
                task=task,
                environment_hash=environment_hash,
                cache_key=f"{provider}:{environment_hash}",
                has_compose=any(
                    (environment_dir / filename).is_file() for filename in _COMPOSE_FILENAMES
                ),
            )
        )

    if not candidates:
        raise ValueError(f"No Harbor tasks found directly under dataset path: {dataset_path}")
    return candidates


def _load_cache(cache_path: Path) -> dict[str, object]:
    if not cache_path.exists():
        return {"schema_version": ROUTING_SCHEMA_VERSION, "routes": {}}
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read E2B route cache {cache_path}: {exc}") from exc
    if not isinstance(cache, dict) or not isinstance(cache.get("routes"), dict):
        raise ValueError(f"Invalid E2B route cache structure: {cache_path}")
    return cache


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_selector_file(path: Path, plan: dict[str, object]) -> None:
    task_routes = plan.get("tasks")
    if not isinstance(task_routes, list):
        raise ValueError("E2B route plan has no task list")
    lines = []
    for task_route in task_routes:
        if not isinstance(task_route, dict):
            raise ValueError("E2B route plan contains an invalid task record")
        route = task_route.get("route")
        selector = task_route.get("selector")
        if route not in ("plain-e2b", "dind") or not isinstance(selector, str):
            raise ValueError("E2B route plan contains an invalid route or selector")
        if "\t" in selector or "\n" in selector:
            raise ValueError(f"Task selector cannot be represented safely: {selector!r}")
        lines.append(f"{route}\t{selector}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.writelines(lines)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def _exclusive_cache_lock(cache_path: Path):
    import fcntl

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_name(f"{cache_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cached_decision(
    cache: dict[str, object],
    candidate: TaskCandidate,
) -> RouteDecision | None:
    routes = cache.get("routes")
    if not isinstance(routes, dict):
        return None
    raw_record = routes.get(candidate.cache_key)
    if not isinstance(raw_record, dict):
        return None
    route = raw_record.get("route")
    reason = raw_record.get("reason")
    if route not in ("plain-e2b", "dind") or not isinstance(reason, str):
        return None
    return RouteDecision(route=route, reason=reason, source="cache")


def _record_decision(
    cache_path: Path,
    cache: dict[str, object],
    candidate: TaskCandidate,
    decision: RouteDecision,
) -> None:
    routes = cache.setdefault("routes", {})
    if not isinstance(routes, dict):
        raise ValueError(f"Invalid E2B route cache structure: {cache_path}")
    routes[candidate.cache_key] = {
        "checked_at": datetime.now(UTC).isoformat(),
        "dataset_name": candidate.dataset_name,
        "environment_hash": candidate.environment_hash,
        "environment_name": candidate.task.short_name,
        "provider": _provider_fingerprint(),
        "reason": decision.reason,
        "route": decision.route,
    }
    cache["schema_version"] = ROUTING_SCHEMA_VERSION
    _write_json_atomic(cache_path, cache)


def _lossy_dockerfile_reason(candidate: TaskCandidate) -> str | None:
    """Catch instructions the E2B parser accepts by silently discarding semantics."""

    dockerfile = candidate.task.paths.environment_dir / "Dockerfile"
    if not dockerfile.is_file():
        return None

    from dockerfile_parse import DockerfileParser

    with tempfile.TemporaryDirectory(prefix="harbor-nexau-dockerfile-") as temporary_dir:
        Path(temporary_dir, "Dockerfile").write_text(
            dockerfile.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        structure = DockerfileParser(path=temporary_dir).structure

    if any(item.get("instruction") == "SHELL" for item in structure):
        return "Dockerfile SHELL semantics are silently ignored by the E2B parser"
    return None


def _short_exception(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    if not message:
        message = exc.__class__.__name__
    return message[:500]


def _is_transient_exception(exc: BaseException) -> bool:
    if exc.__class__.__name__ in _TRANSIENT_EXCEPTION_NAMES:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MESSAGE_MARKERS)


def _deterministic_incompatibility_reason(exc: BaseException) -> str | None:
    if _is_transient_exception(exc):
        return None

    exception_name = exc.__class__.__name__
    message = str(exc).lower()
    if exception_name == "ValueError" and any(
        marker in message for marker in _DOCKERFILE_VALUE_ERROR_MARKERS
    ):
        return f"E2B Dockerfile parser rejected the task: {_short_exception(exc)}"
    if exception_name == "BuildException":
        return f"E2B template build rejected the task: {_short_exception(exc)}"
    return None


async def _stop_environment(environment: NexAUE2BEnvironment) -> None:
    cleanup = asyncio.create_task(environment.stop(delete=True))
    try:
        await asyncio.wait_for(asyncio.shield(cleanup), timeout=120)
    except asyncio.CancelledError:
        try:
            await asyncio.wait_for(asyncio.shield(cleanup), timeout=120)
        finally:
            raise


async def _probe_once(
    candidate: TaskCandidate,
    start_ramp_delay_sec: float,
    timeout_sec: float,
) -> None:
    with tempfile.TemporaryDirectory(prefix="harbor-nexau-e2b-preflight-") as temporary_dir:
        trial_paths = TrialPaths(Path(temporary_dir) / "trial")
        trial_paths.mkdir()
        environment = NexAUE2BEnvironment(
            environment_dir=candidate.task.paths.environment_dir,
            environment_name=candidate.task.short_name,
            session_id=f"e2b-preflight-{uuid4().hex}",
            trial_paths=trial_paths,
            task_env_config=candidate.task.config.environment.model_copy(deep=True),
            network_policy=candidate.task.config.environment.resolve_baseline(),
            e2b_start_ramp_delay_sec=start_ramp_delay_sec,
        )
        start_error: BaseException | None = None
        try:
            async with asyncio.timeout(timeout_sec):
                await environment.start(force_build=False)
        except BaseException as exc:
            start_error = exc
            raise
        finally:
            try:
                await _stop_environment(environment)
            except BaseException as cleanup_exc:
                if start_error is None:
                    raise
                logger.warning(
                    "E2B preflight cleanup also failed for %s: %s",
                    candidate.dataset_name,
                    _short_exception(cleanup_exc),
                )


async def _probe_plain_e2b(
    candidate: TaskCandidate,
    start_ramp_delay_sec: float,
    timeout_sec: float,
    *,
    attempts: int = DEFAULT_PREFLIGHT_ATTEMPTS,
) -> RouteDecision:
    lossy_reason = _lossy_dockerfile_reason(candidate)
    if lossy_reason is not None:
        return RouteDecision(route="dind", reason=lossy_reason, source="structure")

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            await _probe_once(candidate, start_ramp_delay_sec, timeout_sec)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            last_error = exc
            deterministic_reason = _deterministic_incompatibility_reason(exc)
            if deterministic_reason is not None:
                return RouteDecision(route="dind", reason=deterministic_reason)
            if attempt < attempts:
                delay_sec = min(10.0, float(2 ** (attempt - 1)))
                logger.warning(
                    "E2B preflight attempt %s/%s for %s was inconclusive: %s; retrying in %.1fs",
                    attempt,
                    attempts,
                    candidate.dataset_name,
                    _short_exception(exc),
                    delay_sec,
                )
                await asyncio.sleep(delay_sec)
                continue
            break
        else:
            return RouteDecision(
                route="plain-e2b",
                reason="E2B template and sandbox preflight succeeded",
            )

    if last_error is None:
        raise RuntimeError(f"E2B preflight for {candidate.dataset_name} produced no result")
    raise RuntimeError(
        f"inconclusive after {attempts} attempt(s): "
        f"{last_error.__class__.__name__}: {_short_exception(last_error)}"
    ) from last_error


async def build_route_plan(
    dataset_path: Path,
    cache_path: Path,
    *,
    concurrency: int,
    start_ramp_delay_sec: float,
    timeout_sec: float = DEFAULT_PREFLIGHT_TIMEOUT_SEC,
    attempts: int = DEFAULT_PREFLIGHT_ATTEMPTS,
    refresh: bool = False,
    probe: Probe | None = None,
) -> dict[str, object]:
    """Return a complete route plan, caching only trustworthy decisions."""

    if concurrency <= 0:
        raise ValueError(f"Preflight concurrency must be positive, got {concurrency}")
    if timeout_sec <= 0:
        raise ValueError(f"Preflight timeout must be positive, got {timeout_sec}")
    if attempts <= 0:
        raise ValueError(f"Preflight attempts must be positive, got {attempts}")

    candidates = _discover_tasks(dataset_path)
    cache = _load_cache(cache_path)
    probe_impl = probe
    if probe_impl is None:

        async def probe_impl(
            candidate: TaskCandidate,
            ramp_delay_sec: float,
            task_timeout_sec: float,
        ) -> RouteDecision:
            return await _probe_plain_e2b(
                candidate,
                ramp_delay_sec,
                task_timeout_sec,
                attempts=attempts,
            )

    semaphore = asyncio.Semaphore(concurrency)
    decisions: dict[str, RouteDecision] = {}
    unknown: list[TaskCandidate] = []

    for candidate in candidates:
        if candidate.has_compose:
            decisions[candidate.selector] = RouteDecision(
                route="dind",
                reason="Task defines Docker Compose services",
                source="structure",
            )
            continue
        cached = None if refresh else _cached_decision(cache, candidate)
        if cached is not None:
            decisions[candidate.selector] = cached
        else:
            unknown.append(candidate)

    logger.info(
        "E2B route plan: %s task(s), %s cached/structural, %s requiring preflight",
        len(candidates),
        len(decisions),
        len(unknown),
    )

    async def run_probe(candidate: TaskCandidate) -> tuple[TaskCandidate, RouteDecision]:
        async with semaphore:
            decision = await probe_impl(candidate, start_ramp_delay_sec, timeout_sec)
        _record_decision(cache_path, cache, candidate, decision)
        logger.info(
            "E2B preflight routed %s to %s: %s",
            candidate.dataset_name,
            decision.route,
            decision.reason,
        )
        return candidate, decision

    probe_results = await asyncio.gather(
        *(run_probe(candidate) for candidate in unknown),
        return_exceptions=True,
    )
    unresolved: dict[str, str] = {}
    for candidate, result in zip(unknown, probe_results, strict=True):
        if isinstance(result, BaseException):
            if isinstance(result, asyncio.CancelledError):
                raise result
            unresolved[candidate.dataset_name] = _short_exception(result)
            continue
        resolved_candidate, decision = result
        decisions[resolved_candidate.selector] = decision

    if unresolved:
        raise PreflightUnresolvedError(unresolved)

    task_routes = []
    for candidate in candidates:
        decision = decisions[candidate.selector]
        task_routes.append(
            {
                "dataset_name": candidate.dataset_name,
                "environment_hash": candidate.environment_hash,
                "reason": decision.reason,
                "route": decision.route,
                "selector": candidate.selector,
                "source": decision.source,
            }
        )

    return {
        "dataset_path": str(dataset_path.expanduser().resolve()),
        "dind_selectors": [route["selector"] for route in task_routes if route["route"] == "dind"],
        "generated_at": datetime.now(UTC).isoformat(),
        "plain_e2b_selectors": [
            route["selector"] for route in task_routes if route["route"] == "plain-e2b"
        ],
        "provider": _provider_fingerprint(),
        "schema_version": ROUTING_SCHEMA_VERSION,
        "tasks": task_routes,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selectors-output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--concurrency", type=_positive_int, default=8)
    parser.add_argument(
        "--start-ramp-delay-sec",
        type=_non_negative_float,
        default=0.0,
    )
    parser.add_argument(
        "--timeout-sec",
        type=_positive_float,
        default=DEFAULT_PREFLIGHT_TIMEOUT_SEC,
    )
    parser.add_argument(
        "--attempts",
        type=_positive_int,
        default=DEFAULT_PREFLIGHT_ATTEMPTS,
    )
    parser.add_argument("--refresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    cache_path = (args.cache or _default_cache_path()).expanduser()
    try:
        NexAUE2BEnvironment.preflight()
        with _exclusive_cache_lock(cache_path):
            plan = asyncio.run(
                build_route_plan(
                    args.dataset,
                    cache_path,
                    concurrency=args.concurrency,
                    start_ramp_delay_sec=args.start_ramp_delay_sec,
                    timeout_sec=args.timeout_sec,
                    attempts=args.attempts,
                    refresh=args.refresh,
                )
            )
        _write_json_atomic(args.output, plan)
        _write_selector_file(args.selectors_output, plan)
    except PreflightUnresolvedError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("E2B preflight interrupted")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
