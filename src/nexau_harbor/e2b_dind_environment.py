"""Compose-capable Harbor environment backed by the project E2B DinD template."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import override
from uuid import uuid4

from harbor.constants import MAIN_SERVICE_NAME
from harbor.environments import e2b as harbor_e2b
from harbor.environments.base import ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.compose_service_ops import (
    ComposeServiceOpsMixin,
    ComposeServiceTransport,
)
from harbor.environments.definition import (
    effective_exec_cwd,
    require_agent_environment_definition,
    should_use_prebuilt_docker_image,
)
from harbor.environments.dind_compose import DinDComposeOps
from harbor.environments.docker import (
    COMPOSE_BUILD_PATH,
    COMPOSE_PREBUILT_PATH,
    ENV_COMPOSE_NAME,
    RESOURCES_COMPOSE_NAME,
    self_bind_mount,
    write_mounts_compose_file,
    write_resources_compose_file,
)
from harbor.environments.docker.compose_env import (
    ComposeInfraEnvVars,
    legacy_log_mount_env_vars,
    merge_compose_env,
)
from harbor.environments.docker.docker import _sanitize_docker_image_name
from harbor.environments.tar_transfer import (
    extract_dir_from_file,
    pack_dir_to_file,
    remote_pack_command,
    remote_unpack_command,
)
from harbor.models.task.config import NetworkMode
from harbor.models.trial.config import ResourceMode, ServiceVolumeConfig
from harbor.utils.env import resolve_env_vars
from tenacity import retry, stop_after_attempt, wait_exponential

from nexau_harbor.constants import NEXAU_DETACHED_EXEC_ENV, REMOTE_LOG_DIR
from nexau_harbor.e2b_dind_template import (
    DEFAULT_CPU_COUNT,
    DEFAULT_MEMORY_MB,
    TEMPLATE_NAME,
    TEMPLATE_VERSION,
)
from nexau_harbor.e2b_environment import (
    E2B_DETACHED_POLL_INTERVAL_SEC,
    E2B_DETACHED_REQUEST_TIMEOUT_SEC,
    E2B_SANDBOX_TIMEOUT_SEC,
    NexAUE2BEnvironment,
    _detached_command_wrapper,
    _detached_exec_paths,
    _DetachedExecPaths,
    _reserve_sandbox_start_delay,
)
from nexau_harbor.trace_artifacts import (
    AgentLogDownloadError,
    PendingAgentLogDownload,
    PriorityTraceAssets,
    parse_priority_trace_probe,
    priority_trace_probe_command,
)

_AgentLogDownloadError = AgentLogDownloadError
_PendingAgentLogDownload = PendingAgentLogDownload

E2B_DIND_TEMPLATE_ENV = "HARBOR_NEXAU_E2B_DIND_TEMPLATE"
E2B_DIND_DEFAULT_TEMPLATE = f"{TEMPLATE_NAME}:{TEMPLATE_VERSION}"
E2B_DIND_DEFAULT_STORAGE_CAPACITY_MB = 41_062

_COMPOSE_DIR = "/harbor/compose"
_ENVIRONMENT_DIR = "/harbor/environment"
_MOUNTS_COMPOSE_NAME = "docker-compose-mounts.json"
_CLI_ENV_NAME = "compose-cli.env"
_DOCKER_READY_TIMEOUT_SEC = 120
_MAIN_READY_TIMEOUT_SEC = 120
_DEFAULT_COMPOSE_UP_TIMEOUT_SEC = 600
_HOST_TRANSFER_TIMEOUT_SEC = 300
_HOST_TRANSFER_MAX_ATTEMPTS = 5
_AGENT_TRACE_DOWNLOAD_CONCURRENCY = 4
_AGENT_TRACE_RECOVERY_GRACE_SEC = 15 * 60
_HOST_DETACHED_STATE_ROOT = PurePosixPath("/tmp/harbor-nexau-exec")
_COMPOSE_BUILD_MAX_ATTEMPTS = 3
_COMPOSE_BUILD_RETRY_BASE_DELAY_SEC = 5

_agent_trace_download_slots = asyncio.Semaphore(_AGENT_TRACE_DOWNLOAD_CONCURRENCY)
_TRANSIENT_DOCKER_BUILD_ERROR = re.compile(
    r"(?:unexpected eof|short read|connection reset|tls handshake timeout|"
    r"peer closed connection without sending tls close_notify)",
    flags=re.IGNORECASE,
)


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


def _is_transient_docker_build_error(result: ExecResult) -> bool:
    output = f"{result.stderr}\n{result.stdout}"
    return _TRANSIENT_DOCKER_BUILD_ERROR.search(output) is not None


def _remote_pack_command_with_exclusions(
    source_dir: str,
    archive_path: str,
    exclude_relative_paths: tuple[str, ...] = (),
) -> str:
    exclude_flags = " ".join(
        f"--exclude={shlex.quote(f'./{relative_path}')}" for relative_path in exclude_relative_paths
    )
    parts = (
        "tar -czf",
        shlex.quote(archive_path),
        exclude_flags,
        "-C",
        shlex.quote(source_dir),
        ".",
    )
    return " ".join(part for part in parts if part)


def _compose_project_name(session_id: str) -> str:
    """Return a stable Docker Compose project name with a bounded length."""

    slug = re.sub(r"[^a-z0-9_-]+", "-", session_id.lower())
    slug = re.sub(r"-+", "-", slug).strip("-_")
    if not slug or not slug[0].isalnum():
        slug = f"p-{slug}"
    if len(slug) <= 63:
        return slug
    digest = hashlib.sha256(slug.encode()).hexdigest()[:8]
    return f"{slug[:54].rstrip('-_')}-{digest}"


def _compose_env_file_content(env: dict[str, str]) -> str:
    """Serialize Compose interpolation values without expanding ``$`` tokens."""

    lines: list[str] = []
    for name, value in sorted(env.items()):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(f"Invalid Docker Compose environment variable name: {name!r}")
        # Compose treats single-quoted dotenv values literally. It supports
        # escaped backslashes/single quotes and multiline single-quoted values.
        quoted_value = value.replace("\\", "\\\\").replace("'", "\\'")
        lines.append(f"{name}='{quoted_value}'")
    return "\n".join(lines) + ("\n" if lines else "")


class _E2BDinDCompose(DinDComposeOps):
    """Run Harbor's shared Compose transport inside an E2B DinD sandbox."""

    _SELF_BIND_LOG_DIRS = True
    _CP_FILE_TIMEOUT_SEC = 120
    _CP_DIR_TIMEOUT_SEC = 300

    def __init__(self, env: NexAUE2BDinDEnvironment) -> None:
        self._env = env
        self._use_prebuilt = False
        self._resolved_task_env = (
            resolve_env_vars(env.task_env_config.env) if env.task_env_config.env else {}
        )

    @override
    async def _host_exec(
        self,
        command: str,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        return await self._env._exec_on_host(command, timeout_sec=timeout_sec)

    @override
    async def _stage_file_to_host(self, source_path: Path | str, host_path: str) -> None:
        await self._env._upload_file_to_host(source_path, host_path)

    @override
    async def _fetch_file_from_host(
        self,
        host_path: str,
        target_path: Path | str,
    ) -> None:
        await self._env._download_file_from_host(host_path, target_path)

    @override
    async def _stage_dir_to_host(self, source_dir: Path | str, host_dir: str) -> None:
        remote_archive = f"/tmp/harbor-upload-{uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as temp_dir:
            local_archive = Path(temp_dir) / "payload.tar.gz"
            pack_dir_to_file(source_dir, local_archive)
            try:
                await self._env._upload_file_to_host(local_archive, remote_archive)
                result = await self._host_exec(
                    remote_unpack_command(remote_archive, host_dir),
                    timeout_sec=_HOST_TRANSFER_TIMEOUT_SEC,
                )
                if result.return_code != 0:
                    raise RuntimeError(
                        "Failed to unpack directory on E2B DinD host: "
                        f"{result.stderr or result.stdout or 'no output'}"
                    )
            finally:
                await self._env._remove_host_path_best_effort(remote_archive)

    @override
    async def _fetch_dir_from_host(
        self,
        host_dir: str,
        target_dir: Path | str,
        *,
        exclude_relative_paths: tuple[str, ...] = (),
        detached_pack: bool = False,
    ) -> None:
        remote_archive = f"/tmp/harbor-download-{uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as temp_dir:
            local_archive = Path(temp_dir) / "payload.tar.gz"
            try:
                pack_command = (
                    _remote_pack_command_with_exclusions(
                        host_dir,
                        remote_archive,
                        exclude_relative_paths,
                    )
                    if exclude_relative_paths
                    else remote_pack_command(host_dir, remote_archive)
                )
                result = (
                    await self._env._exec_on_host_detached(pack_command)
                    if detached_pack
                    else await self._host_exec(
                        pack_command,
                        timeout_sec=_HOST_TRANSFER_TIMEOUT_SEC,
                    )
                )
                if result.return_code != 0:
                    raise RuntimeError(
                        "Failed to pack directory on E2B DinD host: "
                        f"{result.stderr or result.stdout or 'no output'}"
                    )
                await self._env._download_file_from_host(remote_archive, local_archive)
                extract_dir_from_file(local_archive, target_dir)
            finally:
                await self._env._remove_host_path_best_effort(remote_archive)

    async def download_agent_logs(
        self,
        source_dir: str,
        target_dir: Path | str,
    ) -> None:
        """Recover native trace sources before transferring secondary Agent logs."""

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        try:
            trace_probe = await self._host_exec(
                priority_trace_probe_command(source_dir),
                timeout_sec=30,
            )
        except Exception as probe_error:
            self._env.logger.warning(
                "Could not determine which remote Agent trace assets exist; "
                "falling back to a full log-directory transfer: %s",
                probe_error,
            )
            try:
                await self._fetch_dir_from_host(
                    source_dir,
                    target,
                    detached_pack=True,
                )
            except Exception as transfer_error:
                raise _AgentLogDownloadError(
                    "Failed to download Agent logs after the trace existence probe failed",
                    trace_expected=True,
                ) from transfer_error
            return

        if trace_probe.return_code not in (0, 1):
            self._env.logger.warning(
                "Remote Agent trace probe failed with exit code %s; falling back "
                "to a full log-directory transfer: %s",
                trace_probe.return_code,
                trace_probe.stderr or trace_probe.stdout or "no output",
            )
            try:
                await self._fetch_dir_from_host(
                    source_dir,
                    target,
                    detached_pack=True,
                )
            except Exception as transfer_error:
                raise _AgentLogDownloadError(
                    "Failed to download Agent logs after a failed trace probe",
                    trace_expected=True,
                ) from transfer_error
            return

        assets = parse_priority_trace_probe(trace_probe.stdout or "")
        if not assets.present:
            try:
                await self._fetch_dir_from_host(source_dir, target)
            except Exception as transfer_error:
                raise _AgentLogDownloadError(
                    "Failed to download Agent logs; no trace file was present when checked",
                    trace_expected=False,
                ) from transfer_error
            return

        try:
            await self._fetch_priority_trace_assets(source_dir, target, assets)
        except Exception as priority_error:
            self._env.logger.warning(
                "Priority Agent trace download failed; retrying it inside the full "
                "log-directory archive: %s",
                priority_error,
            )
            try:
                await self._fetch_dir_from_host(
                    source_dir,
                    target,
                    detached_pack=True,
                )
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
            await self._fetch_dir_from_host(
                source_dir,
                target,
                exclude_relative_paths=assets.exclusions,
            )
        except Exception:
            self._env.logger.exception(
                "The critical Agent trace sources were recovered, but the remaining "
                "Agent logs could not be downloaded"
            )

    async def _fetch_priority_trace_assets(
        self,
        source_dir: str,
        target: Path,
        assets: PriorityTraceAssets,
    ) -> None:
        source = source_dir.rstrip("/")
        for name in assets.files:
            async with _agent_trace_download_slots:
                await self._fetch_file_from_host(f"{source}/{name}", target / name)

        for name in assets.directories:
            destination = target / name
            temporary = target / f".{name}.{uuid4().hex}.part"
            try:
                async with _agent_trace_download_slots:
                    await self._fetch_dir_from_host(
                        f"{source}/{name}",
                        temporary,
                        detached_pack=True,
                    )
                if destination.exists():
                    shutil.rmtree(destination)
                temporary.replace(destination)
            finally:
                shutil.rmtree(temporary, ignore_errors=True)

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

    def _compose_referenced_env_vars(self) -> dict[str, str]:
        compose_paths = [
            self._env.environment_dir / "docker-compose.yaml",
            *self._env.extra_docker_compose_paths,
        ]
        content = "\n".join(
            path.read_text(encoding="utf-8") for path in compose_paths if path.exists()
        )
        matches = re.findall(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}|\$([A-Za-z_][A-Za-z0-9_]*)\b",
            content,
        )
        return {
            name: value
            for first, second in matches
            if (name := first or second)
            if (value := os.environ.get(name)) is not None
        }

    def _resolve_volumes(self) -> list[ServiceVolumeConfig]:
        return [
            self_bind_mount(mount) if mount.get("type") == "bind" else mount
            for mount in self._env._mounts
        ]

    def _infra_env_vars(self) -> dict[str, str]:
        env_vars = ComposeInfraEnvVars(
            main_image_name=_sanitize_docker_image_name(f"hb__{self._env.environment_name}"),
            context_dir=_ENVIRONMENT_DIR,
            prebuilt_image_name=(
                self._env.task_env_config.docker_image if self._use_prebuilt else None
            ),
            cpus=self._env._effective_cpus,
            memory=(f"{memory_mb}M" if (memory_mb := self._env._effective_memory_mb) else None),
        ).to_env_dict()
        env_vars.update(legacy_log_mount_env_vars(self._resolve_volumes(), host_value="target"))
        return env_vars

    def _compose_env_vars(self) -> dict[str, str]:
        user_env = self._compose_referenced_env_vars()
        user_env.update(self._resolved_task_env)
        user_env.update(self._env._persistent_env)
        return merge_compose_env(
            user_env=user_env,
            infra_env=self._infra_env_vars(),
            logger=self._env.logger,
            collision_label="Referenced/task/persistent env vars",
        )

    def _extra_compose_target_paths(self) -> list[str]:
        return [
            f"{_COMPOSE_DIR}/docker-compose-extra-{index}.yaml"
            for index, _ in enumerate(self._env.extra_docker_compose_paths)
        ]

    def _compose_file_flags(self) -> list[str]:
        build_or_prebuilt = (
            COMPOSE_PREBUILT_PATH.name if self._use_prebuilt else COMPOSE_BUILD_PATH.name
        )
        files = [
            f"{_COMPOSE_DIR}/{RESOURCES_COMPOSE_NAME}",
            f"{_COMPOSE_DIR}/{build_or_prebuilt}",
            f"{_COMPOSE_DIR}/{_MOUNTS_COMPOSE_NAME}",
        ]
        if (self._env.environment_dir / "docker-compose.yaml").exists():
            files.append(f"{_ENVIRONMENT_DIR}/docker-compose.yaml")
        files.extend(self._extra_compose_target_paths())
        files.append(f"{_COMPOSE_DIR}/{ENV_COMPOSE_NAME}")

        flags: list[str] = []
        for path in files:
            flags.extend(("-f", path))
        return flags

    @property
    def _project_name(self) -> str:
        return _compose_project_name(self._env.session_id)

    def _compose_cmd(self, subcommand: list[str]) -> str:
        env_file_args: list[str] = []
        if (self._env.environment_dir / ".env").is_file():
            env_file_args.extend(("--env-file", f"{_ENVIRONMENT_DIR}/.env"))
        env_file_args.extend(("--env-file", f"{_COMPOSE_DIR}/{_CLI_ENV_NAME}"))
        return shlex.join(
            [
                "docker",
                "compose",
                "-p",
                self._project_name,
                "--project-directory",
                _ENVIRONMENT_DIR,
                *env_file_args,
                *self._compose_file_flags(),
                *subcommand,
            ]
        )

    @override
    async def _compose_exec(
        self,
        subcommand: list[str],
        timeout_sec: int | None = None,
    ) -> ExecResult:
        compose_env = self._compose_env_vars()
        return await self._env._exec_on_host(
            self._compose_cmd(subcommand),
            env=compose_env,
            timeout_sec=timeout_sec,
        )

    async def _compose_build_exec(self, subcommand: list[str], *, phase: str) -> ExecResult:
        """Run a long Compose build phase detached and retry recoverable network failures."""

        compose_env = self._compose_env_vars()
        command = self._compose_cmd(subcommand)
        for attempt in range(1, _COMPOSE_BUILD_MAX_ATTEMPTS + 1):
            result = await self._env._exec_on_host_detached(command, env=compose_env)
            if result.return_code == 0:
                return result
            if attempt >= _COMPOSE_BUILD_MAX_ATTEMPTS or not _is_transient_docker_build_error(
                result
            ):
                return result

            delay_sec = _COMPOSE_BUILD_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
            self._env.logger.warning(
                "docker compose %s hit a transient network error; retrying in %ss "
                "(attempt %s/%s): %s",
                phase,
                delay_sec,
                attempt + 1,
                _COMPOSE_BUILD_MAX_ATTEMPTS,
                (result.stderr or result.stdout or "no output")[-500:],
            )
            await asyncio.sleep(delay_sec)
        raise AssertionError("unreachable")

    @staticmethod
    def _service_exec_subcommand(
        command: str,
        *,
        service: str,
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | int | None,
    ) -> list[str]:
        parts = ["exec", "-T"]
        if cwd:
            parts.extend(("-w", cwd))
        for name, value in (env or {}).items():
            parts.extend(("-e", f"{name}={value}"))
        if user is not None:
            parts.extend(("-u", str(user)))
        parts.append(service)
        if service == MAIN_SERVICE_NAME:
            parts.extend(("bash", "-lc", command))
        else:
            parts.extend(("sh", "-c", command))
        return parts

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        *,
        service: str | None = None,
    ) -> ExecResult:
        return await self._compose_exec(
            self._service_exec_subcommand(
                command,
                service=service or MAIN_SERVICE_NAME,
                cwd=cwd,
                env=env,
                user=user,
            ),
            timeout_sec=timeout_sec,
        )

    async def _stage_resources_compose_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / RESOURCES_COMPOSE_NAME
            write_resources_compose_file(
                local_path,
                cpu_request=self._env._resource_request_value("cpu", auto_mode=ResourceMode.LIMIT),
                cpu_limit=self._env._resource_limit_value("cpu", auto_mode=ResourceMode.LIMIT),
                memory_request_mb=self._env._resource_request_value(
                    "memory", auto_mode=ResourceMode.LIMIT
                ),
                memory_limit_mb=self._env._resource_limit_value(
                    "memory", auto_mode=ResourceMode.LIMIT
                ),
            )
            await self._stage_file_to_host(
                local_path,
                f"{_COMPOSE_DIR}/{RESOURCES_COMPOSE_NAME}",
            )

    async def _stage_mounts_compose_file(
        self,
        volumes: list[ServiceVolumeConfig],
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / _MOUNTS_COMPOSE_NAME
            write_mounts_compose_file(local_path, volumes)
            await self._stage_file_to_host(
                local_path,
                f"{_COMPOSE_DIR}/{_MOUNTS_COMPOSE_NAME}",
            )

    async def _stage_extra_compose_files(self) -> None:
        for source, target in zip(
            self._env.extra_docker_compose_paths,
            self._extra_compose_target_paths(),
            strict=True,
        ):
            await self._stage_file_to_host(source, target)

    async def _stage_cli_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / _CLI_ENV_NAME
            local_path.write_text(
                _compose_env_file_content(self._compose_env_vars()),
                encoding="utf-8",
            )
            target_path = f"{_COMPOSE_DIR}/{_CLI_ENV_NAME}"
            await self._stage_file_to_host(local_path, target_path)
            result = await self._host_exec(
                f"chmod 600 {shlex.quote(target_path)}",
                timeout_sec=10,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    "Failed to protect Docker Compose interpolation env file: "
                    f"{result.stderr or result.stdout or 'no output'}"
                )

    async def _wait_for_main_container(self) -> None:
        last_output = ""
        for _ in range(_MAIN_READY_TIMEOUT_SEC // 2):
            result = await self._compose_exec(
                ["exec", "-T", MAIN_SERVICE_NAME, "true"],
                timeout_sec=10,
            )
            if result.return_code == 0:
                return
            last_output = result.stderr or result.stdout or ""
            await asyncio.sleep(2)
        raise RuntimeError(
            f"Compose main service did not become ready within {_MAIN_READY_TIMEOUT_SEC}s: "
            f"{last_output[-500:]}"
        )

    async def start(self, force_build: bool) -> None:
        self._use_prebuilt = should_use_prebuilt_docker_image(
            self._env.environment_dir,
            docker_image=self._env.task_env_config.docker_image,
            force_build=force_build,
        )

        ready_command = (
            "if command -v harbor-nexau-dind-ready >/dev/null 2>&1; then "
            "harbor-nexau-dind-ready; else "
            f"timeout {_DOCKER_READY_TIMEOUT_SEC} sh -c "
            "'until docker info >/dev/null 2>&1; do sleep 1; done' "
            "&& docker compose version >/dev/null; fi"
        )
        ready = await self._host_exec(
            ready_command,
            timeout_sec=_DOCKER_READY_TIMEOUT_SEC + 10,
        )
        if ready.return_code != 0:
            raise RuntimeError(
                "Docker daemon in E2B DinD sandbox did not become ready: "
                f"{ready.stderr or ready.stdout or 'no output'}"
            )

        mkdir_result = await self._host_exec(
            f"mkdir -p {_COMPOSE_DIR} {_ENVIRONMENT_DIR}",
            timeout_sec=10,
        )
        if mkdir_result.return_code != 0:
            raise RuntimeError(
                "Failed to create Harbor directories on E2B DinD host: "
                f"{mkdir_result.stderr or mkdir_result.stdout or 'no output'}"
            )

        for path in (COMPOSE_BUILD_PATH, COMPOSE_PREBUILT_PATH):
            await self._stage_file_to_host(path, f"{_COMPOSE_DIR}/{path.name}")
        await self._stage_resources_compose_file()
        await self._stage_env_compose_file(_COMPOSE_DIR)
        await self._stage_dir_to_host(self._env.environment_dir, _ENVIRONMENT_DIR)
        await self._stage_extra_compose_files()
        # The pinned swem-dind base wraps /usr/bin/docker with ``sudo``, which
        # scrubs command-scoped variables before Compose interpolation. An
        # explicit protected env file preserves them across that wrapper.
        await self._stage_cli_env_file()

        volumes = self._resolve_volumes()
        await self._stage_mounts_compose_file(volumes)
        bind_sources = [volume["source"] for volume in volumes if volume.get("type") == "bind"]
        if bind_sources:
            quoted = " ".join(shlex.quote(source) for source in bind_sources)
            result = await self._host_exec(
                f"mkdir -p {quoted} && chmod 777 {quoted}",
                timeout_sec=30,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    "Failed to prepare Compose bind mounts on E2B DinD host: "
                    f"{result.stderr or result.stdout or 'no output'}"
                )

        self._env._compose_files_staged = True
        configured = await self._compose_exec(["config", "--quiet"], timeout_sec=30)
        if configured.return_code != 0:
            raise RuntimeError(
                "docker compose config failed: "
                f"{configured.stderr or configured.stdout or 'no output'}"
            )

        pulled = await self._compose_build_exec(
            ["pull", "--ignore-buildable"],
            phase="pull",
        )
        if pulled.return_code != 0:
            raise RuntimeError(
                f"docker compose pull failed: {pulled.stderr or pulled.stdout or 'no output'}"
            )

        built = await self._compose_build_exec(
            ["build"],
            phase="build",
        )
        if built.return_code != 0:
            raise RuntimeError(
                f"docker compose build failed: {built.stderr or built.stdout or 'no output'}"
            )

        # The sandbox starts public so Docker can pull/build. Runtime policy is
        # enforced at the outer E2B boundary, which keeps service-to-service
        # Docker networking intact and supports later phase-policy changes.
        if self._env.network_policy.network_mode != NetworkMode.PUBLIC:
            await self._env._apply_network_policy(self._env.network_policy)

        started = await self._compose_exec(
            ["up", "-d", "--pull", "never"],
            timeout_sec=self._env._compose_up_timeout_sec,
        )
        if started.return_code != 0:
            raise RuntimeError(
                f"docker compose up failed: {started.stderr or started.stdout or 'no output'}"
            )
        await self._wait_for_main_container()


class NexAUE2BDinDEnvironment(ComposeServiceOpsMixin, NexAUE2BEnvironment):
    """Run every Harbor task as a Compose project in ``harbor-nexau-dind``."""

    SUPPORTS_DETACHED_AGENT_EXEC = True

    def __init__(
        self,
        *args,
        dind_template: str | None = None,
        dind_cpu_capacity: int | str | None = None,
        dind_memory_capacity_mb: int | str | None = None,
        dind_storage_capacity_mb: int | str | None = None,
        compose_up_timeout_sec: int | str = _DEFAULT_COMPOSE_UP_TIMEOUT_SEC,
        e2b_start_ramp_delay_sec: int | float | str | None = None,
        **kwargs,
    ) -> None:
        template = dind_template or os.environ.get(
            E2B_DIND_TEMPLATE_ENV,
            E2B_DIND_DEFAULT_TEMPLATE,
        )
        if not template or any(char.isspace() for char in template):
            raise ValueError("dind_template must be a non-empty E2B template reference")
        capacity_values = {
            "dind_cpu_capacity": dind_cpu_capacity,
            "dind_memory_capacity_mb": dind_memory_capacity_mb,
            "dind_storage_capacity_mb": dind_storage_capacity_mb,
        }
        missing_capacities = [name for name, value in capacity_values.items() if value is None]
        if template != E2B_DIND_DEFAULT_TEMPLATE and missing_capacities:
            names = ", ".join(missing_capacities)
            raise ValueError(
                "A custom dind_template requires explicit capacity metadata; "
                f"set {names} to the actual template capacity values"
            )

        self._dind_template = template
        self._dind_cpu_capacity = _positive_int(
            dind_cpu_capacity if dind_cpu_capacity is not None else DEFAULT_CPU_COUNT,
            name="dind_cpu_capacity",
        )
        self._dind_memory_capacity_mb = _positive_int(
            (dind_memory_capacity_mb if dind_memory_capacity_mb is not None else DEFAULT_MEMORY_MB),
            name="dind_memory_capacity_mb",
        )
        self._dind_storage_capacity_mb = _positive_int(
            (
                dind_storage_capacity_mb
                if dind_storage_capacity_mb is not None
                else E2B_DIND_DEFAULT_STORAGE_CAPACITY_MB
            ),
            name="dind_storage_capacity_mb",
        )
        self._compose_up_timeout_sec = _positive_int(
            compose_up_timeout_sec,
            name="compose_up_timeout_sec",
        )
        self._compose_files_staged = False
        self._pending_agent_log_download: _PendingAgentLogDownload | None = None

        super().__init__(
            *args,
            e2b_start_ramp_delay_sec=e2b_start_ramp_delay_sec,
            **kwargs,
        )
        self._template_name = self._dind_template
        self._workdir = None
        self._compose = _E2BDinDCompose(self)
        self._validate_dind_capacity()

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(
            cpu_limit=True,
            cpu_request=True,
            memory_limit=True,
            memory_request=True,
        )

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            disable_internet=True,
            network_allowlist=True,
            network_allowlist_hostnames=True,
            network_allowlist_wildcard_hostnames=True,
            network_allowlist_ipv4_addresses=True,
            network_allowlist_ipv6_addresses=False,
            network_allowlist_ipv4_cidrs=False,
            network_allowlist_ipv6_cidrs=False,
            dynamic_network_policy=True,
            docker_compose=True,
        )

    @property
    @override
    def _uses_compose(self) -> bool:
        return True

    @override
    def _validate_definition(self) -> None:
        require_agent_environment_definition(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
            extra_docker_compose_paths=self.extra_docker_compose_paths,
        )

    def _validate_dind_capacity(self) -> None:
        requirements = (
            ("CPU", self._effective_cpus, self._dind_cpu_capacity, "override_cpus"),
            (
                "memory",
                self._effective_memory_mb,
                self._dind_memory_capacity_mb,
                "override_memory_mb",
            ),
            (
                "storage",
                self._effective_storage_mb,
                self._dind_storage_capacity_mb,
                "override_storage_mb",
            ),
        )
        for label, requested, capacity, override_name in requirements:
            if requested is not None and requested > capacity:
                raise ValueError(
                    f"Task requests {requested} {label} units, but E2B DinD template "
                    f"{self._dind_template!r} is configured for at most {capacity}. "
                    f"Use a larger template and matching dind_*_capacity setting, or "
                    f"set {override_name} explicitly if reduced resources are acceptable."
                )

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
        initial_lease_timeout_sec = min(
            E2B_SANDBOX_TIMEOUT_SEC,
            self._max_lifetime_sec,
        )
        self._sandbox = await harbor_e2b.AsyncSandbox.create(
            template=self._dind_template,
            metadata={
                "environment_name": self.environment_name,
                "session_id": self.session_id,
            },
            envs=self._startup_env(),
            timeout=initial_lease_timeout_sec,
            allow_internet_access=True,
            network=None,
        )
        self._start_keepalive()

    async def _exec_on_host_detached(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run a long DinD-host command in the background and poll it briefly."""

        paths = _detached_exec_paths(_HOST_DETACHED_STATE_ROOT)
        wrapper = _detached_command_wrapper(command, paths)
        handle = await self._dispatch_command(
            wrapper,
            cwd="/",
            env=env,
            timeout_sec=None,
            user="root",
        )
        pid = int(handle.pid)
        self.logger.info(
            "Started detached E2B DinD host command as process %s; polling every %s seconds",
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
                    "Failed to close the initial E2B DinD command stream; switching to "
                    "short polling anyway: %s",
                    exc,
                )
            return await self._poll_detached_command(
                pid=pid,
                paths=paths,
                user="root",
            )
        except asyncio.CancelledError:
            await NexAUE2BEnvironment._cancel_detached_command(
                self,
                pid=pid,
                paths=paths,
                user="root",
            )
            raise

    async def _exec_on_host(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        return await harbor_e2b.E2BEnvironment.exec(
            self,
            command,
            cwd="/",
            env=env,
            timeout_sec=timeout_sec,
            user="root",
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _upload_file_to_host(
        self,
        source_path: Path | str,
        target_path: str,
    ) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        with Path(source_path).open("rb") as source:
            await self._sandbox.files.write(target_path, source, user="root")

    async def _stream_file_from_host(
        self,
        source_path: str,
        target_path: Path | str,
    ) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = target.with_name(f".{target.name}.{uuid4().hex}.part")
        try:
            stream = await self._sandbox.files.read(
                source_path,
                format="stream",
                user="root",
                gzip=True,
                stream_idle_timeout=_HOST_TRANSFER_TIMEOUT_SEC,
            )
            async with stream:
                with temporary_path.open("wb") as output:
                    async for chunk in stream:
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)

    @retry(
        stop=stop_after_attempt(_HOST_TRANSFER_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _download_file_from_host(
        self,
        source_path: str,
        target_path: Path | str,
    ) -> None:
        await self._stream_file_from_host(source_path, target_path)

    async def _remove_host_path_best_effort(self, path: str) -> None:
        try:
            result = await self._exec_on_host(
                f"rm -f {shlex.quote(path)}",
                timeout_sec=10,
            )
            if result.return_code != 0:
                self.logger.warning(
                    "Failed to remove temporary E2B DinD host path %s: %s",
                    path,
                    result.stderr or result.stdout or "no output",
                )
        except Exception as exc:
            self.logger.warning(
                "Failed to remove temporary E2B DinD host path %s: %s",
                path,
                exc,
            )

    def _compose_service_transport(
        self,
        service: str | None,
    ) -> ComposeServiceTransport:
        del service
        return self._compose

    @override
    async def start(self, force_build: bool) -> None:
        if self._sandbox is not None:
            await self.stop(delete=True)
        await self._create_sandbox()
        await self._compose.start(force_build)
        await self._upload_environment_dir_after_start()

    async def _retry_pending_agent_log_download(self) -> bool:
        pending = self._pending_agent_log_download
        if pending is None:
            return True
        self.logger.warning(
            "Retrying Agent-log recovery before stopping E2B sandbox %s",
            getattr(self._sandbox, "sandbox_id", "unknown"),
        )
        try:
            await self._compose.download_agent_logs(
                pending.source_dir,
                pending.target_dir,
            )
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
        deadline = getattr(self, "_keepalive_deadline", None)
        remaining_lifetime_sec = (
            deadline - asyncio.get_running_loop().time() if deadline is not None else None
        )
        grace_sec = _AGENT_TRACE_RECOVERY_GRACE_SEC
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
        if detached:
            return await self._exec_detached(
                command,
                cwd=cwd,
                env=sanitized_env,
                timeout_sec=timeout_sec,
                user=user,
            )
        return await self._compose.exec(
            command,
            cwd=effective_exec_cwd(cwd, self.task_env_config.workdir, self._workdir),
            env=self._merge_env(sanitized_env),
            timeout_sec=timeout_sec,
            user=self._resolve_user(user),
        )

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._compose.upload_file(source_path, target_path)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._compose.upload_dir(source_dir, target_dir)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._compose.download_file(source_path, target_path)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if PurePosixPath(source_dir) != REMOTE_LOG_DIR:
            await self._compose.download_dir(source_dir, target_dir)
            return

        target = Path(target_dir)
        try:
            await self._compose.download_agent_logs(source_dir, target)
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
        except Exception:
            self._pending_agent_log_download = _PendingAgentLogDownload(
                source_dir,
                target,
                trace_expected=True,
            )
            self.logger.exception(
                "Failed to recover E2B Agent logs; a final retry will run before cleanup"
            )
            raise
        self._pending_agent_log_download = None

    @override
    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return await self._compose.is_dir(path, user=self._resolve_user(user))

    @override
    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        return await self._compose.is_file(path, user=self._resolve_user(user))

    @override
    async def _dispatch_detached_command(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_sec: int | None,
        user: str,
    ):
        subcommand = self._compose._service_exec_subcommand(
            command,
            service=MAIN_SERVICE_NAME,
            cwd=cwd,
            env=env,
            user=user,
        )
        compose_env = self._compose._compose_env_vars()
        return await self._dispatch_command(
            self._compose._compose_cmd(subcommand),
            cwd="/",
            env=compose_env,
            timeout_sec=timeout_sec,
            user="root",
        )

    @override
    def _detached_status_user(self, command_user: str) -> str:
        # State is self-bound from the main container into the DinD host. The
        # main-container user may not exist on that outer host.
        del command_user
        return "root"

    @override
    async def _cancel_detached_command(
        self,
        *,
        pid: int,
        paths: _DetachedExecPaths,
        user: str,
    ) -> None:
        if self._sandbox is not None:
            try:
                raw_inner_pid = await self._sandbox.files.read(
                    paths.pid,
                    format="text",
                    user="root",
                    request_timeout=E2B_DETACHED_REQUEST_TIMEOUT_SEC,
                )
                inner_pid = int(str(raw_inner_pid).strip())
                await self._compose.exec(
                    f"kill -TERM {inner_pid}",
                    timeout_sec=10,
                    user="root",
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to stop inner detached Agent process after cancellation: %s",
                    exc,
                )
        await super()._cancel_detached_command(pid=pid, paths=paths, user=user)

    @override
    async def stop(self, delete: bool) -> None:
        pending = self._pending_agent_log_download
        if pending is not None and self._sandbox is not None:
            recovered = await self._retry_pending_agent_log_download()
            pending = self._pending_agent_log_download
            if not recovered and pending is not None and pending.trace_expected:
                await self._preserve_sandbox_for_trace_recovery()
                return

        if self._sandbox is not None and self._compose_files_staged:
            try:
                result = await self._compose._compose_exec(
                    ["down", "-v", "--remove-orphans"],
                    timeout_sec=120,
                )
                if result.return_code != 0:
                    self.logger.warning(
                        "docker compose down failed during E2B cleanup: %s",
                        result.stderr or result.stdout or "no output",
                    )
            except Exception as exc:
                self.logger.warning("docker compose cleanup failed: %s", exc)
        self._compose_files_staged = False
        await super().stop(delete)
