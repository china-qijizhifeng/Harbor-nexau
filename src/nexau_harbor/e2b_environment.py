"""E2B compatibility environment for Harbor 0.21.x."""

import asyncio
import math
import os
import re
import shlex
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import override
from uuid import uuid4

from harbor.environments import e2b as harbor_e2b
from harbor.environments.base import ExecResult
from harbor.environments.definition import effective_exec_cwd
from harbor.models.task.config import NetworkMode
from tenacity import retry, stop_after_attempt, wait_exponential

from nexau_harbor.constants import NEXAU_DETACHED_EXEC_ENV, REMOTE_LOG_DIR
from nexau_harbor.trace_artifacts import (
    AgentLogDownloadError,
    PendingAgentLogDownload,
    PriorityTraceAssets,
    parse_priority_trace_probe,
    priority_trace_probe_command,
)

_AgentLogDownloadError = AgentLogDownloadError
_PendingAgentLogDownload = PendingAgentLogDownload

E2B_SANDBOX_TIMEOUT_SEC = 3_600
E2B_KEEPALIVE_INTERVAL_SEC = 60
E2B_DEFAULT_MAX_LIFETIME_SEC = 24 * 60 * 60
E2B_MAX_LIFETIME_ENV = "HARBOR_NEXAU_E2B_MAX_LIFETIME_SEC"
E2B_NO_PROXY_HOSTS_ENV = "HARBOR_NEXAU_NO_PROXY_HOSTS"
E2B_START_RAMP_DELAY_ENV = "HARBOR_NEXAU_E2B_START_RAMP_DELAY_SEC"
E2B_DETACHED_POLL_INTERVAL_SEC = 30
E2B_DETACHED_REQUEST_TIMEOUT_SEC = 30
E2B_DETACHED_MAX_CONSECUTIVE_POLL_ERRORS = 5
E2B_DEFAULT_TEMPLATE_BUILD_ATTEMPTS = 2
E2B_TEMPLATE_BUILD_LOG_MAX_ENTRIES = 256
E2B_TEMPLATE_BUILD_LOG_TAIL_CHARS = 16_384
E2B_AGENT_TRACE_RECOVERY_GRACE_SEC = 15 * 60

_DETACHED_STATUS_PREFIX = "__HARBOR_NEXAU_STATUS__:"
_MISSING_DEFAULT_TEMPLATE_TAG = "tag 'default' does not exist for template"
_sandbox_start_lock = threading.Lock()
_next_sandbox_start_at = 0.0

_DOCKERFILE_FROM_PATTERN = re.compile(
    r"^[ \t]*FROM(?:[ \t]+--platform=\S+)?[ \t]+(?P<image>\S+)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_ASTRAL_UV_COPY_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)COPY[ \t]+--from=ghcr\.io/astral-sh/uv:"
    r"(?P<version>[0-9][0-9A-Za-z.+-]*)[ \t]+/uv[ \t]+/uvx[ \t]+"
    r"/usr/local/bin/[ \t]*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class _DetachedExecPaths:
    directory: str
    pid: str
    stdout: str
    stderr: str
    exit_code: str


class E2BTemplateBuildTimeoutError(TimeoutError):
    """A configured local deadline expired while E2B was building a template."""


def _detached_exec_paths(
    root: PurePosixPath = REMOTE_LOG_DIR,
) -> _DetachedExecPaths:
    directory = (root / f"harbor-exec-{uuid4().hex}").as_posix()
    return _DetachedExecPaths(
        directory=directory,
        pid=f"{directory}/pid",
        stdout=f"{directory}/stdout",
        stderr=f"{directory}/stderr",
        exit_code=f"{directory}/exitcode",
    )


def _detached_command_wrapper(command: str, paths: _DetachedExecPaths) -> str:
    quote = shlex.quote
    exit_code_tmp = f"{paths.exit_code}.tmp"
    return "\n".join(
        (
            f"mkdir -p {quote(paths.directory)}",
            f"printf '%s\\n' \"$$\" > {quote(paths.pid)}",
            "set +e",
            "(",
            command,
            f") > {quote(paths.stdout)} 2> {quote(paths.stderr)}",
            "harbor_nexau_rc=$?",
            f"printf '%s\\n' \"$harbor_nexau_rc\" > {quote(exit_code_tmp)}",
            f"mv {quote(exit_code_tmp)} {quote(paths.exit_code)}",
            'exit "$harbor_nexau_rc"',
        )
    )


def _detached_status_command(pid: int, paths: _DetachedExecPaths) -> str:
    quote = shlex.quote
    return " ".join(
        (
            f"if test -f {quote(paths.exit_code)}; then",
            f"printf '%s' {quote(f'{_DETACHED_STATUS_PREFIX}DONE:')};",
            f"tr -d '\\r\\n' < {quote(paths.exit_code)};",
            "printf '\\n';",
            f"elif kill -0 {pid} 2>/dev/null; then",
            f"printf '%s\\n' {quote(f'{_DETACHED_STATUS_PREFIX}RUNNING')};",
            "else",
            f"printf '%s\\n' {quote(f'{_DETACHED_STATUS_PREFIX}CRASHED')};",
            "fi",
        )
    )


def _parse_detached_status(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_DETACHED_STATUS_PREFIX):
            return line.removeprefix(_DETACHED_STATUS_PREFIX)
    raise ValueError(f"Detached E2B status marker missing from output: {stdout!r}")


def _is_python_image(image: str) -> bool:
    normalized = image.lower()
    return normalized == "python" or normalized.startswith(
        ("python:", "python@", "docker.io/library/python:", "docker.io/library/python@")
    )


def _rewrite_e2b_incompatible_uv_copy(dockerfile: str) -> tuple[str, int]:
    """Replace the uv image copy form that E2B's Dockerfile parser rejects.

    E2B 2.x rejects absolute sources in ``COPY --from``, even though Docker accepts
    them. For Python-based stages, installing the same pinned uv package exposes both
    ``uv`` and ``uvx`` in /usr/local/bin without changing the source task on disk.
    """

    rewrite_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal rewrite_count

        preceding_from = tuple(_DOCKERFILE_FROM_PATTERN.finditer(dockerfile, 0, match.start()))
        if not preceding_from or not _is_python_image(preceding_from[-1].group("image")):
            return match.group(0)

        rewrite_count += 1
        indent = match.group("indent")
        version = match.group("version")
        return f"{indent}RUN python -m pip install --no-cache-dir uv=={version}"

    return _ASTRAL_UV_COPY_PATTERN.sub(replace, dockerfile), rewrite_count


def _resolve_max_lifetime_sec(configured: int | str | None) -> int:
    raw_value: object = configured
    if raw_value is None:
        raw_value = os.environ.get(
            E2B_MAX_LIFETIME_ENV,
            str(E2B_DEFAULT_MAX_LIFETIME_SEC),
        )

    if isinstance(raw_value, bool) or not isinstance(raw_value, int | str):
        raise ValueError(f"{E2B_MAX_LIFETIME_ENV} must be a positive integer")

    try:
        max_lifetime_sec = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{E2B_MAX_LIFETIME_ENV} must be a positive integer, got {raw_value!r}"
        ) from exc

    if max_lifetime_sec <= 0:
        raise ValueError(f"{E2B_MAX_LIFETIME_ENV} must be positive, got {max_lifetime_sec}")
    return max_lifetime_sec


def _resolve_start_ramp_delay_sec(configured: int | float | str | None) -> float:
    raw_value: object = configured
    if raw_value is None:
        raw_value = os.environ.get(E2B_START_RAMP_DELAY_ENV, "0")
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float | str):
        raise ValueError(f"{E2B_START_RAMP_DELAY_ENV} must be a finite non-negative number")
    try:
        resolved = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{E2B_START_RAMP_DELAY_ENV} must be a finite non-negative number, got {raw_value!r}"
        ) from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(
            f"{E2B_START_RAMP_DELAY_ENV} must be a finite non-negative number, got {raw_value!r}"
        )
    return resolved


def _resolve_template_build_attempts(configured: int | str | None) -> int:
    raw_value: object = E2B_DEFAULT_TEMPLATE_BUILD_ATTEMPTS if configured is None else configured
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | str):
        raise ValueError("e2b_template_build_attempts must be a positive integer")
    try:
        attempts = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"e2b_template_build_attempts must be a positive integer, got {raw_value!r}"
        ) from exc
    if attempts <= 0:
        raise ValueError(f"e2b_template_build_attempts must be positive, got {attempts}")
    return attempts


def _resolve_template_build_timeout_sec(
    configured: int | float | str | None,
) -> float | None:
    if configured is None:
        return None
    if isinstance(configured, bool) or not isinstance(configured, int | float | str):
        raise ValueError("e2b_template_build_timeout_sec must be a finite positive number")
    try:
        timeout_sec = float(configured)
    except ValueError as exc:
        raise ValueError(
            f"e2b_template_build_timeout_sec must be a finite positive number, got {configured!r}"
        ) from exc
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError(
            f"e2b_template_build_timeout_sec must be a finite positive number, got {configured!r}"
        )
    return timeout_sec


def _reserve_sandbox_start_delay(ramp_delay_sec: float) -> float:
    """Reserve one process-wide E2B creation slot and return its wait time."""

    global _next_sandbox_start_at

    if ramp_delay_sec <= 0:
        return 0.0
    with _sandbox_start_lock:
        now = time.monotonic()
        start_at = max(now, _next_sandbox_start_at)
        _next_sandbox_start_at = start_at + ramp_delay_sec
    return max(0.0, start_at - now)


def _merge_configured_no_proxy_hosts() -> None:
    """Append configured direct-connect hosts to both conventional spellings."""

    configured = [
        host.strip()
        for host in os.environ.get(E2B_NO_PROXY_HOSTS_ENV, "").split(",")
        if host.strip()
    ]
    if not configured:
        return

    for env_name in ("NO_PROXY", "no_proxy"):
        existing = [
            host.strip() for host in os.environ.get(env_name, "").split(",") if host.strip()
        ]
        merged = list(existing)
        for host in configured:
            if host not in merged:
                merged.append(host)
        os.environ[env_name] = ",".join(merged)


def _bounded_lease_timeout(remaining_lifetime_sec: float) -> int | None:
    remaining_whole_seconds = int(remaining_lifetime_sec)
    if remaining_whole_seconds <= 0:
        return None
    return min(E2B_SANDBOX_TIMEOUT_SEC, remaining_whole_seconds)


def _is_missing_default_template_tag(exc: Exception) -> bool:
    """Identify the E2B response for an alias with no launchable default build."""

    message = str(exc)
    return message.startswith("404:") and _MISSING_DEFAULT_TEMPLATE_TAG in message


class NexAUE2BEnvironment(harbor_e2b.E2BEnvironment):
    """Use a bounded renewable lease without exceeding the E2B plan limit."""

    SUPPORTS_DETACHED_AGENT_EXEC = True

    def __init__(
        self,
        *args,
        max_lifetime_sec: int | str | None = None,
        e2b_start_ramp_delay_sec: int | float | str | None = None,
        e2b_template_build_attempts: int | str | None = None,
        e2b_template_build_timeout_sec: int | float | str | None = None,
        **kwargs,
    ) -> None:
        _merge_configured_no_proxy_hosts()
        self._max_lifetime_sec = _resolve_max_lifetime_sec(max_lifetime_sec)
        self._e2b_start_ramp_delay_sec = _resolve_start_ramp_delay_sec(e2b_start_ramp_delay_sec)
        self._e2b_template_build_attempts = _resolve_template_build_attempts(
            e2b_template_build_attempts
        )
        self._e2b_template_build_timeout_sec = _resolve_template_build_timeout_sec(
            e2b_template_build_timeout_sec
        )
        self._last_template_build_log_tail = ""
        self._keepalive_task: asyncio.Task[None] | None = None
        self._keepalive_deadline: float | None = None
        self._pending_agent_log_download: _PendingAgentLogDownload | None = None
        super().__init__(*args, **kwargs)

    @override
    async def start(self, force_build: bool) -> None:
        try:
            await super().start(force_build)
        except Exception as exc:
            if force_build or not _is_missing_default_template_tag(exc):
                raise

            self.logger.warning(
                "E2B template %s exists but has no launchable default build; forcing one rebuild",
                self._template_name,
            )
            await super().start(force_build=True)

    async def _read_detached_output(
        self,
        path: str,
        *,
        user: str,
    ) -> tuple[str, str | None]:
        if self._sandbox is None:
            return "", "E2B sandbox disappeared before detached output could be read"
        try:
            content = await self._sandbox.files.read(
                path,
                format="text",
                user=user,
                request_timeout=E2B_DETACHED_REQUEST_TIMEOUT_SEC,
            )
            return str(content), None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning("Failed to read detached E2B output %s: %s", path, exc)
            return "", f"Failed to read detached E2B output {path}: {exc}"

    async def _detached_exec_result(
        self,
        paths: _DetachedExecPaths,
        *,
        user: str,
        return_code: int,
        diagnostic: str | None = None,
    ) -> ExecResult:
        stdout, stdout_error = await self._read_detached_output(paths.stdout, user=user)
        stderr, stderr_error = await self._read_detached_output(paths.stderr, user=user)
        details = [item for item in (diagnostic, stdout_error, stderr_error) if item]
        if details:
            suffix = "\n".join(details)
            stderr = f"{stderr.rstrip()}\n{suffix}".lstrip()
        return ExecResult(stdout=stdout, stderr=stderr, return_code=return_code)

    async def _poll_detached_command(
        self,
        *,
        pid: int,
        paths: _DetachedExecPaths,
        user: str,
        on_running: Callable[[], Awaitable[None]] | None = None,
    ) -> ExecResult:
        status_command = _detached_status_command(pid, paths)
        consecutive_errors = 0

        while True:
            await asyncio.sleep(E2B_DETACHED_POLL_INTERVAL_SEC)
            sandbox = self._sandbox
            if sandbox is None:
                return await self._detached_exec_result(
                    paths,
                    user=user,
                    return_code=-1,
                    diagnostic="E2B sandbox disappeared while detached Agent was running",
                )

            try:
                status_result = await sandbox.commands.run(
                    status_command,
                    timeout=E2B_DETACHED_REQUEST_TIMEOUT_SEC,
                    request_timeout=E2B_DETACHED_REQUEST_TIMEOUT_SEC,
                    user=user,
                )
                status = _parse_detached_status(status_result.stdout)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= E2B_DETACHED_MAX_CONSECUTIVE_POLL_ERRORS:
                    raise RuntimeError(
                        "Failed to poll detached E2B Agent command "
                        f"{consecutive_errors} consecutive times"
                    ) from exc
                self.logger.warning(
                    "Failed to poll detached E2B Agent command (attempt %s/%s); "
                    "the remote process remains running: %s",
                    consecutive_errors,
                    E2B_DETACHED_MAX_CONSECUTIVE_POLL_ERRORS,
                    exc,
                )
                continue

            consecutive_errors = 0
            if status == "RUNNING":
                if on_running is not None:
                    await on_running()
                continue
            if status == "CRASHED":
                return await self._detached_exec_result(
                    paths,
                    user=user,
                    return_code=-1,
                    diagnostic=(
                        f"Detached E2B Agent process {pid} disappeared without writing "
                        f"an exit status; state directory: {paths.directory}"
                    ),
                )
            if not status.startswith("DONE:"):
                return await self._detached_exec_result(
                    paths,
                    user=user,
                    return_code=-1,
                    diagnostic=f"Unrecognized detached E2B Agent status: {status!r}",
                )

            raw_exit_code = status.removeprefix("DONE:").strip()
            try:
                return_code = int(raw_exit_code)
            except ValueError:
                return await self._detached_exec_result(
                    paths,
                    user=user,
                    return_code=-1,
                    diagnostic=f"Invalid detached E2B Agent exit status: {raw_exit_code!r}",
                )
            return await self._detached_exec_result(
                paths,
                user=user,
                return_code=return_code,
            )

    async def _exec_detached(
        self,
        command: str,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_sec: int | None,
        user: str | int | None,
    ) -> ExecResult:
        resolved_user = self._resolve_user(user)
        user_string = str(resolved_user) if resolved_user is not None else "root"
        status_user = self._detached_status_user(user_string)
        merged_env = self._merge_env(env)
        paths = _detached_exec_paths()
        wrapper = _detached_command_wrapper(command, paths)
        handle = await self._dispatch_detached_command(
            wrapper,
            cwd=effective_exec_cwd(cwd, self.task_env_config.workdir, self._workdir),
            env=merged_env,
            timeout_sec=timeout_sec,
            user=user_string,
        )
        pid = int(handle.pid)
        self.logger.info(
            "Started detached E2B Agent command as process %s; polling every %s seconds",
            pid,
            E2B_DETACHED_POLL_INTERVAL_SEC,
        )
        try:
            try:
                await handle.disconnect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.warning(
                    "Failed to close the initial E2B Agent event stream; switching to "
                    "short polling anyway: %s",
                    exc,
                )
            result = await self._poll_detached_command(
                pid=pid,
                paths=paths,
                user=status_user,
            )
            self.logger.info(
                "Detached E2B Agent command %s finished with exit code %s",
                pid,
                result.return_code,
            )
            return result
        except asyncio.CancelledError:
            await self._cancel_detached_command(pid=pid, paths=paths, user=user_string)
            raise

    def _detached_status_user(self, command_user: str) -> str:
        """Return the user used to read and poll detached command state."""

        return command_user

    async def _dispatch_detached_command(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_sec: int | None,
        user: str,
    ):
        """Dispatch a detached command in the provider's execution target.

        The direct E2B adapter targets the sandbox itself. Compose-capable
        subclasses override this hook so the wrapped command runs in the main
        service while the returned handle remains on the DinD host.
        """

        return await self._dispatch_command(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def _cancel_detached_command(
        self,
        *,
        pid: int,
        paths: _DetachedExecPaths,
        user: str,
    ) -> None:
        """Best-effort cancellation for a detached command."""

        del paths, user
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            await sandbox.commands.kill(
                pid,
                request_timeout=E2B_DETACHED_REQUEST_TIMEOUT_SEC,
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to kill detached E2B Agent command %s after cancellation: %s",
                pid,
                exc,
            )

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        sanitized_env = dict(env) if env is not None else None
        detached = False
        if sanitized_env is not None:
            detached = sanitized_env.pop(NEXAU_DETACHED_EXEC_ENV, None) == "1"
        if not detached:
            return await super().exec(
                command,
                cwd=cwd,
                env=sanitized_env,
                timeout_sec=timeout_sec,
                user=user,
            )
        return await self._exec_detached(
            command,
            cwd=cwd,
            env=sanitized_env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def _keepalive_loop(self) -> None:
        while True:
            sandbox = self._sandbox
            if sandbox is None:
                return

            loop = asyncio.get_running_loop()
            if self._keepalive_deadline is None:
                return
            remaining_lifetime_sec = self._keepalive_deadline - loop.time()
            lease_timeout_sec = _bounded_lease_timeout(remaining_lifetime_sec)
            if lease_timeout_sec is None:
                self.logger.warning(
                    "E2B sandbox %s reached its configured maximum lifetime of %s "
                    "seconds; lease renewal stopped",
                    sandbox.sandbox_id,
                    self._max_lifetime_sec,
                )
                return

            try:
                await sandbox.set_timeout(lease_timeout_sec)
                self.logger.debug(
                    "Refreshed E2B sandbox %s timeout to %s seconds (maximum lifetime %s seconds)",
                    sandbox.sandbox_id,
                    lease_timeout_sec,
                    self._max_lifetime_sec,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.warning(
                    "Failed to refresh E2B sandbox timeout; will retry in %s seconds: %s",
                    E2B_KEEPALIVE_INTERVAL_SEC,
                    exc,
                )

            await asyncio.sleep(min(E2B_KEEPALIVE_INTERVAL_SEC, remaining_lifetime_sec))

    def _start_keepalive(self) -> None:
        keepalive_task = getattr(self, "_keepalive_task", None)
        if keepalive_task is not None and not keepalive_task.done():
            return

        self._keepalive_deadline = asyncio.get_running_loop().time() + self._max_lifetime_sec
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(),
            name=f"e2b-keepalive-{self.session_id}",
        )

    @override
    async def _create_template(self) -> None:
        attempts = getattr(
            self,
            "_e2b_template_build_attempts",
            E2B_DEFAULT_TEMPLATE_BUILD_ATTEMPTS,
        )
        for attempt in range(1, attempts + 1):
            try:
                await self._create_template_once()
                return
            except Exception as exc:
                if attempt >= attempts:
                    raise
                delay_sec = min(10.0, float(2 ** (attempt - 1)))
                self.logger.warning(
                    "E2B template build attempt %s/%s failed for %s: %s; retrying in %.1fs",
                    attempt,
                    attempts,
                    self._template_name,
                    exc,
                    delay_sec,
                )
                await asyncio.sleep(delay_sec)

    async def _create_template_once(self) -> None:
        if self.task_env_config.docker_image:
            template = harbor_e2b.Template().from_image(
                image=self.task_env_config.docker_image,
            )
        else:
            dockerfile = self._environment_definition_path.read_text(encoding="utf-8")
            dockerfile, rewrite_count = _rewrite_e2b_incompatible_uv_copy(dockerfile)
            if rewrite_count:
                self.logger.info(
                    "Applied %s E2B compatibility rewrite(s) for Astral uv",
                    rewrite_count,
                )

            template = harbor_e2b.Template(
                file_context_path=str(self.environment_dir),
            ).from_dockerfile(
                dockerfile_content_or_path=dockerfile,
            )

        build_kwargs = {
            "template": template,
            "alias": self._template_name,
        }
        if self._effective_cpus is not None:
            build_kwargs["cpu_count"] = self._effective_cpus
        if self._effective_memory_mb is not None:
            build_kwargs["memory_mb"] = self._effective_memory_mb

        build_log_entries: deque[str] = deque(maxlen=E2B_TEMPLATE_BUILD_LOG_MAX_ENTRIES)

        def capture_build_log(entry: object) -> None:
            level = str(getattr(entry, "level", "info"))
            message = str(getattr(entry, "message", entry)).strip()
            build_log_entries.append(f"[{level}] {message}")
            self._last_template_build_log_tail = "\n".join(build_log_entries)[
                -E2B_TEMPLATE_BUILD_LOG_TAIL_CHARS:
            ]

        build_kwargs["on_build_logs"] = capture_build_log
        self._last_template_build_log_tail = ""
        try:
            build_timeout_sec = getattr(self, "_e2b_template_build_timeout_sec", None)
            if build_timeout_sec is None:
                await harbor_e2b.AsyncTemplate.build(**build_kwargs)
            else:
                build_timeout = asyncio.timeout(build_timeout_sec)
                try:
                    async with build_timeout:
                        await harbor_e2b.AsyncTemplate.build(**build_kwargs)
                except TimeoutError as exc:
                    if not build_timeout.expired():
                        raise
                    raise E2BTemplateBuildTimeoutError(
                        f"E2B template build exceeded {build_timeout_sec:g}s"
                    ) from exc
        except Exception as exc:
            try:
                setattr(exc, "e2b_build_log_tail", self._last_template_build_log_tail)
            except Exception:
                pass
            raise

    async def _stop_keepalive(self) -> None:
        keepalive_task = getattr(self, "_keepalive_task", None)
        self._keepalive_task = None
        if keepalive_task is None:
            return

        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.logger.warning("E2B keepalive task stopped with an error: %s", exc)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    @override
    async def _create_sandbox(self) -> None:
        ramp_wait_sec = _reserve_sandbox_start_delay(self._e2b_start_ramp_delay_sec)
        if ramp_wait_sec > 0:
            self.logger.info(
                "Staggering E2B sandbox creation by %.1f seconds",
                ramp_wait_sec,
            )
            await asyncio.sleep(ramp_wait_sec)
        metadata = {
            "environment_name": self.environment_name,
            "session_id": self.session_id,
        }
        initial_lease_timeout_sec = min(
            E2B_SANDBOX_TIMEOUT_SEC,
            self._max_lifetime_sec,
        )

        self._sandbox = await harbor_e2b.AsyncSandbox.create(
            template=self._template_name,
            metadata=metadata,
            envs=self._startup_env(),
            timeout=initial_lease_timeout_sec,
            allow_internet_access=(self.network_policy.network_mode != NetworkMode.NO_NETWORK),
            network=self._sandbox_create_network_options(),
        )
        self._start_keepalive()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _stream_trace_file(
        self,
        source_path: str,
        target_path: Path,
    ) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = target_path.with_name(f".{target_path.name}.{uuid4().hex}.part")
        try:
            stream = await self._sandbox.files.read(
                source_path,
                format="stream",
                user="root",
                gzip=True,
                stream_idle_timeout=300,
            )
            async with stream:
                with temporary.open("wb") as output:
                    async for chunk in stream:
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            temporary.replace(target_path)
        finally:
            temporary.unlink(missing_ok=True)

    async def _download_priority_trace_assets(
        self,
        source_dir: str,
        target: Path,
        assets: PriorityTraceAssets,
    ) -> None:
        source = source_dir.rstrip("/")
        for name in assets.files:
            await self._stream_trace_file(f"{source}/{name}", target / name)
        for name in assets.directories:
            await super().download_dir(f"{source}/{name}", target / name)

    @staticmethod
    def _missing_priority_trace_assets(
        target: Path,
        assets: PriorityTraceAssets,
    ) -> list[str]:
        missing = [name for name in assets.files if not (target / name).is_file()]
        missing.extend(
            name for name in assets.directories if not (target / name).is_dir()
        )
        return missing

    async def _download_agent_logs(
        self,
        source_dir: str,
        target: Path,
    ) -> None:
        target.mkdir(parents=True, exist_ok=True)
        try:
            probe = await super().exec(
                priority_trace_probe_command(source_dir),
                timeout_sec=30,
                user="root",
            )
        except Exception as probe_error:
            self.logger.warning(
                "Could not determine which remote Agent trace assets exist; "
                "falling back to a full log-directory transfer: %s",
                probe_error,
            )
            try:
                await super().download_dir(source_dir, target)
            except Exception as transfer_error:
                raise _AgentLogDownloadError(
                    "Failed to download Agent logs after the trace existence probe failed",
                    trace_expected=True,
                ) from transfer_error
            return

        if probe.return_code != 0:
            self.logger.warning(
                "Remote Agent trace probe failed with exit code %s; falling back "
                "to a full log-directory transfer: %s",
                probe.return_code,
                probe.stderr or probe.stdout or "no output",
            )
            try:
                await super().download_dir(source_dir, target)
            except Exception as transfer_error:
                raise _AgentLogDownloadError(
                    "Failed to download Agent logs after a failed trace probe",
                    trace_expected=True,
                ) from transfer_error
            return

        assets = parse_priority_trace_probe(probe.stdout or "")
        if not assets.present:
            try:
                await super().download_dir(source_dir, target)
            except Exception as transfer_error:
                raise _AgentLogDownloadError(
                    "Failed to download Agent logs; no trace assets were present when checked",
                    trace_expected=False,
                ) from transfer_error
            return

        try:
            await self._download_priority_trace_assets(source_dir, target, assets)
        except Exception as priority_error:
            self.logger.warning(
                "Priority Agent trace download failed; retrying it inside the full "
                "log-directory transfer: %s",
                priority_error,
            )
            try:
                await super().download_dir(source_dir, target)
            except Exception as transfer_error:
                raise _AgentLogDownloadError(
                    "Both priority trace download and full Agent-log download failed",
                    trace_expected=True,
                ) from transfer_error
            missing = self._missing_priority_trace_assets(target, assets)
            if missing:
                raise _AgentLogDownloadError(
                    "The full Agent-log transfer completed without expected trace assets: "
                    + ", ".join(missing),
                    trace_expected=True,
                )
            return

        try:
            await super().download_dir(source_dir, target)
        except Exception:
            self.logger.exception(
                "The critical Agent trace sources were recovered, but the remaining "
                "Agent logs could not be downloaded"
            )

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if PurePosixPath(source_dir) != REMOTE_LOG_DIR:
            await super().download_dir(source_dir, target_dir)
            return

        target = Path(target_dir)
        try:
            await self._download_agent_logs(source_dir, target)
        except _AgentLogDownloadError as exc:
            self._pending_agent_log_download = _PendingAgentLogDownload(
                source_dir,
                target,
                trace_expected=exc.trace_expected,
            )
            self.logger.exception(
                "Failed to recover E2B Agent logs; a final retry will run before cleanup"
            )
            raise
        self._pending_agent_log_download = None

    async def _retry_pending_agent_log_download(self) -> bool:
        pending = self._pending_agent_log_download
        if pending is None:
            return True
        self.logger.warning(
            "Retrying Agent-log recovery before stopping E2B sandbox %s",
            getattr(self._sandbox, "sandbox_id", "unknown"),
        )
        try:
            await self._download_agent_logs(pending.source_dir, pending.target_dir)
        except _AgentLogDownloadError as exc:
            self._pending_agent_log_download = _PendingAgentLogDownload(
                pending.source_dir,
                pending.target_dir,
                trace_expected=pending.trace_expected or exc.trace_expected,
            )
            self.logger.exception("Agent-log recovery retry failed")
            return False
        except Exception:
            self.logger.exception("Agent-log recovery retry failed")
            return False
        self._pending_agent_log_download = None
        return True

    async def _preserve_sandbox_for_trace_recovery(self) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        sandbox_id = sandbox.sandbox_id
        deadline = self._keepalive_deadline
        remaining_lifetime_sec = (
            deadline - asyncio.get_running_loop().time() if deadline is not None else None
        )
        grace_sec = E2B_AGENT_TRACE_RECOVERY_GRACE_SEC
        if remaining_lifetime_sec is not None:
            grace_sec = min(grace_sec, max(1, round(remaining_lifetime_sec)))

        await self._stop_keepalive()
        try:
            await sandbox.set_timeout(grace_sec)
        except Exception as exc:
            self.logger.error(
                "Could not set a recovery grace period on E2B sandbox %s: %s",
                sandbox_id,
                exc,
            )
        self.logger.error(
            "E2B sandbox %s was not deleted because its Agent trace could not be "
            "recovered; it remains available for up to %s seconds for manual recovery",
            sandbox_id,
            grace_sec,
        )
        self._pending_agent_log_download = None
        self._sandbox = None

    @override
    async def stop(self, delete: bool) -> None:
        pending = getattr(self, "_pending_agent_log_download", None)
        if pending is not None and self._sandbox is not None:
            recovered = await self._retry_pending_agent_log_download()
            pending = self._pending_agent_log_download
            if not recovered and pending is not None and pending.trace_expected:
                await self._preserve_sandbox_for_trace_recovery()
                return
        try:
            await self._stop_keepalive()
        finally:
            await super().stop(delete)
