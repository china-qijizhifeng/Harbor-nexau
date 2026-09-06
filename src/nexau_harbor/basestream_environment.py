"""Harbor environment backed by isolated Basestream Volcano GPU Jobs.

The Harbor controller is expected to run on ``basestream-cpu`` (or another
host with the same GPFS, NorthJob, internal Harbor, and kubectl access).  Each
environment instance owns one image build context and one short-lived GPU Job.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import ssl
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, override

from harbor.environments.base import BaseEnvironment, ExecResult, SandboxBuildFailedError
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.definition import effective_exec_cwd
from harbor.models.task.config import NetworkMode
from harbor.models.trial.paths import TrialPaths

_TERMINAL_PHASES = {"Aborted", "Completed", "Failed", "Succeeded", "Terminated"}
_SUCCESS_PHASES = {"Completed", "Succeeded"}
_SAFE_LABEL = re.compile(r"[^a-z0-9-]+")
_SAFE_USER = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_.-]*|[0-9]+)(?::[0-9]+)?")
_ALLOWED_ENTRYPOINTS = {(), ("/opt/nvidia/nvidia_entrypoint.sh",)}
_DEFAULT_B300_NODES = tuple(f"node-b300-{index:03d}" for index in range(1, 9))
_BUILD_LOCKS: dict[str, asyncio.Lock] = {}
_IO_EXECUTOR = ThreadPoolExecutor(max_workers=64, thread_name_prefix="harbor-basestream")
_SKOPEO_STAGED_MIRROR_SCRIPT = """\
set -eu
source_image=$1
destination_image=$2
expected_digest=$3
parallel_copies=$4
image_dir=/mirror/image
complete_marker=/mirror/download.complete

download_complete=0
if [ -f "$complete_marker" ] && [ -f "$image_dir/manifest.json" ]; then
  recorded_digest=$(cat "$complete_marker")
  actual_digest="sha256:$(sha256sum "$image_dir/manifest.json" | awk '{print $1}')"
  if [ "$recorded_digest" = "$expected_digest" ] && [ "$actual_digest" = "$expected_digest" ]; then
    download_complete=1
    echo "Reusing verified local dir transport: $expected_digest"
  fi
fi

if [ "$download_complete" -ne 1 ]; then
  rm -f "$complete_marker"
  mkdir -p "$image_dir"
  download_attempt=1
  while [ "$download_attempt" -le 3 ]; do
    echo "Downloading source image to local dir transport (attempt $download_attempt/3)"
    if /usr/bin/skopeo copy \
      --preserve-digests \
      --retry-times=10 \
      --image-parallel-copies="$parallel_copies" \
      --src-tls-verify=true \
      "$source_image" "dir:$image_dir"; then
      download_complete=1
      break
    fi
    sleep $((download_attempt * 5))
    download_attempt=$((download_attempt + 1))
  done
  if [ "$download_complete" -ne 1 ]; then
    echo "Source image download did not complete" >&2
    exit 1
  fi
  actual_digest="sha256:$(sha256sum "$image_dir/manifest.json" | awk '{print $1}')"
  if [ "$actual_digest" != "$expected_digest" ]; then
    echo "Downloaded manifest digest mismatch: $actual_digest" >&2
    exit 1
  fi
  printf '%s\n' "$actual_digest" > "$complete_marker"
fi

push_complete=0
push_attempt=1
while [ "$push_attempt" -le 3 ]; do
  echo "Pushing verified local dir transport to Harbor (attempt $push_attempt/3)"
  if /usr/bin/skopeo copy \
    --preserve-digests \
    --retry-times=10 \
    --image-parallel-copies="$parallel_copies" \
    --dest-precompute-digests \
    --dest-tls-verify=false \
    --dest-authfile=/auth/config.json \
    "dir:$image_dir" "$destination_image"; then
    push_complete=1
    break
  fi
  sleep $((push_attempt * 5))
  push_attempt=$((push_attempt + 1))
done
if [ "$push_complete" -ne 1 ]; then
  echo "Internal Harbor push did not complete" >&2
  exit 1
fi
"""


class BasestreamError(RuntimeError):
    """Basestream control-plane or runtime contract failure."""


async def _offload[T](function: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run blocking control-plane I/O without owning the event loop's executor.

    Python 3.13's event-loop executor shutdown can hang in some restricted
    controller environments.  A process-owned executor also lets independent
    Harbor trials share a bounded pool instead of blocking the event loop.
    """

    future = _IO_EXECUTOR.submit(function, *args, **kwargs)
    try:
        # Polling deliberately avoids depending on call_soon_threadsafe wakeups.
        # Python 3.13 can lose that wakeup under the restricted controller's
        # selector sandbox, leaving an already-completed wrapped future asleep.
        while not future.done():
            await asyncio.sleep(0.01)
        return future.result()
    except asyncio.CancelledError:
        future.cancel()
        raise


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _as_int(value: int | str, *, name: str, minimum: int = 1) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _as_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _csv(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    return tuple(item.strip() for item in values if item.strip())


def _slug(value: str, *, limit: int = 42) -> str:
    normalized = _SAFE_LABEL.sub("-", value.lower()).strip("-") or "environment"
    return normalized[:limit].rstrip("-")


def _atomic_json(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(mode)
    temporary.replace(path)


def _tail(value: str, limit: int = 8000) -> str:
    return value if len(value) <= limit else value[-limit:]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class BasestreamSettings:
    namespace: str
    queue: str
    priority_class: str
    pvc_name: str
    pvc_host_root: Path
    control_root: Path
    api_host: str
    credentials_path: Path
    registry_repository: str
    registry_auth_path: Path
    registry_auth_subpath: str
    registry_tls_verify: bool
    harbor_host_ip: str | None
    kaniko_image: str
    kaniko_cache_repository: str
    skopeo_image: str
    skopeo_parallel_copies: int
    docker_hub_mirror: str
    egress_proxy_url: str | None
    egress_state_path: Path
    excluded_gpu_nodes: tuple[str, ...]
    kubectl: str
    build_timeout_sec: int
    start_timeout_sec: int
    runtime_lifetime_sec: int
    poll_interval_sec: int
    build_cpu: str
    build_memory: str
    build_storage: str

    @property
    def control_subpath(self) -> str:
        return self.control_root.relative_to(self.pvc_host_root).as_posix()

    def pvc_subpath(self, path: Path) -> str:
        resolved = path.resolve()
        root = self.pvc_host_root.resolve()
        if not _is_relative_to(resolved, root):
            raise BasestreamError(f"path is outside the Basestream PVC root: {resolved}")
        return resolved.relative_to(root).as_posix()


@dataclass(frozen=True)
class _RegistryReference:
    registry: str
    api_registry: str
    repository: str
    digest: str


def _parse_registry_reference(image_ref: str) -> _RegistryReference:
    if "@" not in image_ref:
        raise BasestreamError(f"runtime image is not immutable: {image_ref}")
    name, digest = image_ref.rsplit("@", 1)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise BasestreamError(f"runtime image has an invalid digest: {image_ref}")
    components = name.split("/")
    first = components[0]
    if "." in first or ":" in first or first == "localhost":
        registry = first.lower()
        repository_parts = components[1:]
    else:
        registry = "docker.io"
        repository_parts = components
    if not repository_parts:
        raise BasestreamError(f"invalid image reference: {image_ref}")
    if len(repository_parts) == 1:
        repository_parts.insert(0, "library")
    last = repository_parts[-1]
    if ":" in last:
        last, tag = last.rsplit(":", 1)
        if not last or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
            raise BasestreamError(f"invalid image tag in reference: {image_ref}")
        repository_parts[-1] = last
    repository = "/".join(repository_parts)
    if not re.fullmatch(
        r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+", repository
    ):
        raise BasestreamError(f"invalid image repository: {image_ref}")
    api_registry = (
        "registry-1.docker.io" if registry in {"docker.io", "index.docker.io"} else registry
    )
    return _RegistryReference(
        registry=registry,
        api_registry=api_registry,
        repository=repository,
        digest=digest,
    )


class _NorthJobClient:
    def __init__(self, settings: BasestreamSettings):
        self.settings = settings
        self._token: str | None = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if authenticated:
            if self._token is None:
                self.authenticate()
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(
            f"http://{self.settings.api_host}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise BasestreamError(
                f"NorthJob {method} {path} returned HTTP {exc.code}: {_tail(body)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise BasestreamError(f"NorthJob {method} {path} failed: {exc.reason}") from exc
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise BasestreamError(f"NorthJob {method} {path} returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise BasestreamError(f"NorthJob {method} {path} returned a non-object payload")
        return result

    def authenticate(self) -> None:
        try:
            credentials = json.loads(self.settings.credentials_path.read_text(encoding="utf-8"))
            username = credentials["username"]
            password = credentials["password"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise BasestreamError(
                f"cannot read NorthJob credentials: {self.settings.credentials_path}"
            ) from exc
        response = self._request(
            "POST",
            "/api/auth/get_token",
            {"username": username, "password": password, "expireTime": 3600},
            authenticated=False,
        )
        token = response.get("data")
        if not isinstance(token, str) or not token:
            raise BasestreamError("NorthJob authentication returned no token")
        self._token = token

    def submit(self, manifest: dict[str, Any]) -> None:
        response = self._request(
            "POST",
            "/api/task/create_job_by_yaml",
            {"code": json.dumps(manifest, separators=(",", ":"))},
        )
        if response.get("code") != 200:
            raise BasestreamError(
                "NorthJob submission failed: "
                + json.dumps(
                    {key: response.get(key) for key in ("code", "message", "err_msg")},
                    ensure_ascii=False,
                )
            )

    def delete(self, job_name: str) -> None:
        response = self._request(
            "DELETE",
            "/api/delete_jobs",
            {"name": job_name, "namespace": self.settings.namespace},
        )
        if response.get("success") is not True:
            raise BasestreamError(f"NorthJob deletion failed for {job_name}")


class BasestreamEnvironment(BaseEnvironment):
    """Single-container Linux GPU environment implemented with Volcano Jobs."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: Any,
        namespace: str | None = None,
        queue: str | None = None,
        priority_class: str | None = None,
        pvc_name: str | None = None,
        pvc_host_root: str | Path | None = None,
        control_root: str | Path | None = None,
        api_host: str | None = None,
        credentials_path: str | Path | None = None,
        registry_repository: str | None = None,
        registry_auth_path: str | Path | None = None,
        registry_auth_subpath: str | None = None,
        registry_tls_verify: bool | str | None = None,
        harbor_host_ip: str | None = None,
        kaniko_image: str | None = None,
        kaniko_cache_repository: str | None = None,
        skopeo_image: str | None = None,
        skopeo_parallel_copies: int | str | None = None,
        docker_hub_mirror: str | None = None,
        egress_proxy_url: str | None = None,
        egress_state_path: str | Path | None = None,
        excluded_gpu_nodes: str | tuple[str, ...] | list[str] | None = None,
        kubectl: str | None = None,
        build_timeout_sec: int | str | None = None,
        start_timeout_sec: int | str | None = None,
        runtime_lifetime_sec: int | str | None = None,
        poll_interval_sec: int | str | None = None,
        build_cpu: str | None = None,
        build_memory: str | None = None,
        build_storage: str | None = None,
        **kwargs: Any,
    ) -> None:
        host_root = Path(
            pvc_host_root or _env("HARBOR_BASESTREAM_PVC_HOST_ROOT", "/gpfs/users/i-zhangguoqiang")
        ).expanduser()
        provider_root = Path(
            control_root
            or _env(
                "HARBOR_BASESTREAM_CONTROL_ROOT",
                "/gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau/.basestream",
            )
        ).expanduser()
        configured_exclusions = excluded_gpu_nodes
        if configured_exclusions is None:
            configured_exclusions = _env(
                "HARBOR_BASESTREAM_EXCLUDED_GPU_NODES", ",".join(_DEFAULT_B300_NODES)
            )
        verify_tls = (
            _as_bool(registry_tls_verify)
            if registry_tls_verify is not None
            else _as_bool(_env("HARBOR_BASESTREAM_REGISTRY_TLS_VERIFY", "false"))
        )
        self.settings = BasestreamSettings(
            namespace=namespace or _env("HARBOR_BASESTREAM_NAMESPACE", "bg-agentic-coding"),
            queue=queue or _env("HARBOR_BASESTREAM_QUEUE", "bg-agentic-coding"),
            priority_class=priority_class
            or _env("HARBOR_BASESTREAM_PRIORITY_CLASS", "higher-priority-job"),
            pvc_name=pvc_name or _env("HARBOR_BASESTREAM_PVC", "i-zhangguoqiang"),
            pvc_host_root=host_root,
            control_root=provider_root,
            api_host=api_host or _env("HARBOR_BASESTREAM_API_HOST", "10.1.2.7:31000"),
            credentials_path=Path(
                credentials_path
                or _env(
                    "HARBOR_BASESTREAM_NORTHJOB_CONFIG",
                    "/root/.config/northjob/userinfo.conf",
                )
            ).expanduser(),
            registry_repository=registry_repository
            or _env(
                "HARBOR_BASESTREAM_REGISTRY_REPOSITORY",
                "harbor.local.clusters/zhangguoqiang/harbor-tb4",
            ),
            registry_auth_path=Path(
                registry_auth_path
                or _env(
                    "HARBOR_BASESTREAM_REGISTRY_AUTH_PATH",
                    "/gpfs/users/i-zhangguoqiang/info/.posttrainbench-secrets/harbor-kaniko-config.json",
                )
            ).expanduser(),
            registry_auth_subpath=registry_auth_subpath
            or _env(
                "HARBOR_BASESTREAM_REGISTRY_AUTH_SUBPATH",
                "info/.posttrainbench-secrets/harbor-kaniko-config.json",
            ),
            registry_tls_verify=verify_tls,
            harbor_host_ip=(
                harbor_host_ip
                if harbor_host_ip is not None
                else _env("HARBOR_BASESTREAM_HARBOR_IP", "10.51.6.8")
            )
            or None,
            kaniko_image=kaniko_image
            or _env(
                "HARBOR_BASESTREAM_KANIKO_IMAGE",
                "m.daocloud.io/gcr.io/kaniko-project/executor@"
                "sha256:c3109d5926a997b100c4343944e06c6b30a6804b2f9abe0994d3de6ef92b028e",
            ),
            kaniko_cache_repository=kaniko_cache_repository
            or _env(
                "HARBOR_BASESTREAM_KANIKO_CACHE_REPOSITORY",
                "harbor.local.clusters/zhangguoqiang/harbor-tb4-cache",
            ),
            skopeo_image=skopeo_image
            or _env(
                "HARBOR_BASESTREAM_SKOPEO_IMAGE",
                "m.daocloud.io/quay.io/skopeo/stable@"
                "sha256:47853bb9fb24202af9110531ebd6e43c5f97701254ca290596640290d17942f4",
            ),
            skopeo_parallel_copies=_as_int(
                skopeo_parallel_copies or _env("HARBOR_BASESTREAM_SKOPEO_PARALLEL_COPIES", "1"),
                name="skopeo_parallel_copies",
            ),
            docker_hub_mirror=(
                docker_hub_mirror
                if docker_hub_mirror is not None
                else _env("HARBOR_BASESTREAM_DOCKER_HUB_MIRROR", "")
            ).rstrip("/"),
            egress_proxy_url=egress_proxy_url
            or os.environ.get("HARBOR_BASESTREAM_EGRESS_PROXY_URL"),
            egress_state_path=Path(
                egress_state_path
                or _env(
                    "HARBOR_BASESTREAM_EGRESS_STATE_PATH",
                    "/gpfs/users/i-zhangguoqiang/terminal-code/NexMLTask/"
                    "gpu_manager/deploy/basestream/.state.json",
                )
            ).expanduser(),
            excluded_gpu_nodes=_csv(configured_exclusions),
            kubectl=kubectl or _env("HARBOR_BASESTREAM_KUBECTL", "kubectl"),
            build_timeout_sec=_as_int(
                build_timeout_sec or _env("HARBOR_BASESTREAM_BUILD_TIMEOUT_SEC", "7200"),
                name="build_timeout_sec",
            ),
            start_timeout_sec=_as_int(
                start_timeout_sec or _env("HARBOR_BASESTREAM_START_TIMEOUT_SEC", "7200"),
                name="start_timeout_sec",
            ),
            runtime_lifetime_sec=_as_int(
                runtime_lifetime_sec or _env("HARBOR_BASESTREAM_RUNTIME_LIFETIME_SEC", "43200"),
                name="runtime_lifetime_sec",
            ),
            poll_interval_sec=_as_int(
                poll_interval_sec or _env("HARBOR_BASESTREAM_POLL_INTERVAL_SEC", "3"),
                name="poll_interval_sec",
            ),
            build_cpu=build_cpu or _env("HARBOR_BASESTREAM_BUILD_CPU", "8"),
            build_memory=build_memory or _env("HARBOR_BASESTREAM_BUILD_MEMORY", "32Gi"),
            build_storage=build_storage or _env("HARBOR_BASESTREAM_BUILD_STORAGE", "100Gi"),
        )
        self._validate_settings()
        self._image_ref: str | None = None
        self._image_user = "root"
        self._image_entrypoint: tuple[str, ...] = ()
        self._image_workdir: str | None = None
        self._job_alias: str | None = None
        self._runtime_job_name: str | None = None
        self._pod_name: str | None = None
        session_digest = hashlib.sha256(session_id.encode()).hexdigest()[:10]
        session_name = f"{_slug(session_id, limit=38)}-{session_digest}"
        self._session_dir = self.settings.control_root / "sessions" / session_name
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

    def _validate_settings(self) -> None:
        host_root = self.settings.pvc_host_root.resolve()
        control_root = self.settings.control_root.resolve()
        if not _is_relative_to(control_root, host_root) or control_root == host_root:
            raise ValueError("control_root must be a child of pvc_host_root")
        if not self.settings.registry_repository or "@" in self.settings.registry_repository:
            raise ValueError("registry_repository must be a taggable repository prefix")
        if "@sha256:" not in self.settings.skopeo_image:
            raise ValueError("skopeo_image must be pinned by sha256 digest")
        if self.settings.docker_hub_mirror and (
            "://" in self.settings.docker_hub_mirror
            or not re.fullmatch(
                r"[A-Za-z0-9.-]+(?::[0-9]+)?(?:/[A-Za-z0-9._-]+)*",
                self.settings.docker_hub_mirror,
            )
        ):
            raise ValueError("docker_hub_mirror must be an image path without a URL scheme")
        if not re.fullmatch(r"[A-Za-z0-9.-]+(?::[0-9]+)?", self.settings.api_host):
            raise ValueError("api_host must be host:port without a URL scheme")
        for value, label in (
            (self.settings.namespace, "namespace"),
            (self.settings.queue, "queue"),
            (self.settings.priority_class, "priority_class"),
            (self.settings.pvc_name, "pvc_name"),
        ):
            if not value or any(character.isspace() for character in value):
                raise ValueError(f"{label} must be a non-empty single value")

    @staticmethod
    @override
    def type() -> str:
        return "basestream"

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
        return EnvironmentCapabilities(gpus=True, disable_internet=True, mounted=True)

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @override
    def _validate_definition(self) -> None:
        if self.extra_docker_compose_paths or any(
            (self.environment_dir / name).exists()
            for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
        ):
            raise ValueError("BasestreamEnvironment supports only single-container tasks")
        if not self.task_env_config.docker_image and not self._dockerfile_path.is_file():
            raise FileNotFoundError(
                f"GPU environment Dockerfile not found: {self._dockerfile_path}"
            )

    @classmethod
    @override
    def preflight(cls) -> None:
        kubectl = _env("HARBOR_BASESTREAM_KUBECTL", "kubectl")
        if shutil.which(kubectl) is None:
            raise SystemExit(
                "BasestreamEnvironment requires kubectl and must run on basestream-cpu"
            )

    def _ensure_host_layout(self) -> None:
        root = self.settings.pvc_host_root.resolve()
        if not root.is_dir():
            raise BasestreamError(f"Basestream PVC host root is missing: {root}")
        control = self.settings.control_root.resolve()
        control.mkdir(mode=0o700, parents=True, exist_ok=True)
        control.chmod(0o700)
        self._session_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._session_dir.chmod(0o700)
        if not self.settings.credentials_path.is_file():
            raise BasestreamError(
                f"NorthJob credentials are missing: {self.settings.credentials_path}"
            )
        if not self.settings.registry_auth_path.is_file():
            raise BasestreamError(
                f"Harbor registry auth is missing: {self.settings.registry_auth_path}"
            )

    def _kubectl_command(self, *args: str) -> list[str]:
        return [self.settings.kubectl, *args, "-n", self.settings.namespace]

    def _run_text(
        self,
        args: list[str],
        *,
        timeout: float | None = None,
        check: bool = True,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                args,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                env={**os.environ, "NO_COLOR": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            raise BasestreamError(f"command timed out: {shlex.join(args)}") from exc
        if check and completed.returncode != 0:
            detail = _tail((completed.stderr or completed.stdout).strip())
            raise BasestreamError(f"command failed ({shlex.join(args)}): {detail}")
        return completed

    def _kubectl_json(self, resource: str, *args: str) -> dict[str, Any]:
        completed = self._run_text(
            self._kubectl_command("get", resource, *args, "-o", "json"),
            timeout=60,
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BasestreamError(f"kubectl returned invalid JSON for {resource}") from exc
        if not isinstance(payload, dict):
            raise BasestreamError(f"kubectl returned a non-object for {resource}")
        return payload

    def _find_job(self, alias: str) -> dict[str, Any] | None:
        jobs = self._kubectl_json("vcjob").get("items") or []
        matches = [
            job
            for job in jobs
            if isinstance(job, dict)
            and job.get("metadata", {}).get("annotations", {}).get("JobAlias") == alias
        ]
        matches.sort(
            key=lambda job: job.get("metadata", {}).get("creationTimestamp", ""),
            reverse=True,
        )
        return matches[0] if matches else None

    def _pods_for_job(self, job_name: str) -> list[dict[str, Any]]:
        pods = (
            self._kubectl_json("pods", "-l", f"volcano.sh/job-name={job_name}").get("items") or []
        )
        valid = [pod for pod in pods if isinstance(pod, dict)]
        valid.sort(
            key=lambda pod: pod.get("metadata", {}).get("creationTimestamp", ""),
            reverse=True,
        )
        return valid

    @staticmethod
    def _extract_job_name(job: dict[str, Any]) -> str:
        return str(job.get("metadata", {}).get("name") or "")

    @staticmethod
    def _job_phase(job: dict[str, Any] | None) -> str:
        if job is None:
            return "Missing"
        return str(job.get("status", {}).get("state", {}).get("phase") or "Pending")

    def _job_logs(self, pod_name: str, container_name: str) -> str:
        if not pod_name:
            return ""
        completed = self._run_text(
            self._kubectl_command("logs", pod_name, "-c", container_name),
            timeout=120,
            check=False,
        )
        return (completed.stdout or "") + (completed.stderr or "")

    def _wait_for_job(
        self,
        alias: str,
        *,
        timeout_sec: int,
        require_terminal: bool,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        deadline = time.monotonic() + timeout_sec
        job: dict[str, Any] | None = None
        pod: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            job = self._find_job(alias)
            phase = self._job_phase(job)
            if job is not None:
                pods = self._pods_for_job(self._extract_job_name(job))
                if pods:
                    pod = pods[0]
            if require_terminal and phase in _TERMINAL_PHASES:
                return job, pod
            if not require_terminal:
                if pod is not None:
                    pod_phase = str(pod.get("status", {}).get("phase") or "")
                    if pod_phase == "Running":
                        return job, pod
                if phase in _TERMINAL_PHASES:
                    logs = self._job_logs(
                        str((pod or {}).get("metadata", {}).get("name") or ""), "main"
                    )
                    raise BasestreamError(
                        f"runtime Job {alias} became {phase} before it was ready: {_tail(logs)}"
                    )
            time.sleep(self.settings.poll_interval_sec)
        condition = ""
        if pod is not None:
            condition = json.dumps(pod.get("status", {}).get("conditions") or [])
        raise BasestreamError(
            f"timed out waiting for Basestream Job {alias}; "
            f"phase={self._job_phase(job)} conditions={_tail(condition)}"
        )

    def _delete_job(self, job_name: str) -> None:
        if not job_name:
            return
        _NorthJobClient(self.settings).delete(job_name)

    def _registry_auth(self, registry: str) -> str:
        try:
            payload = json.loads(self.settings.registry_auth_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BasestreamError("cannot read Harbor registry auth") from exc
        auths = payload.get("auths") if isinstance(payload, dict) else None
        if not isinstance(auths, dict):
            raise BasestreamError("Harbor registry auth has no auths object")
        entry = None
        for candidate in (registry, f"https://{registry}", f"http://{registry}"):
            if isinstance(auths.get(candidate), dict):
                entry = auths[candidate]
                break
        if entry is None:
            raise BasestreamError(f"Harbor registry auth has no entry for {registry}")
        auth = entry.get("auth")
        if isinstance(auth, str) and auth:
            return auth
        username = entry.get("username")
        password = entry.get("password")
        if not isinstance(username, str) or not isinstance(password, str):
            raise BasestreamError(f"Harbor registry auth entry is incomplete for {registry}")
        return base64.b64encode(f"{username}:{password}".encode()).decode()

    def _registry_request(
        self,
        url: str,
        *,
        auth: str,
        accept: str | None = None,
    ) -> tuple[bytes, Any]:
        headers = {"Authorization": f"Basic {auth}"}
        if accept:
            headers["Accept"] = accept
        request = urllib.request.Request(url, headers=headers)
        context = (
            ssl.create_default_context()
            if self.settings.registry_tls_verify
            else ssl._create_unverified_context()
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=context),
        )
        try:
            with opener.open(request, timeout=30) as response:
                return response.read(), response.headers
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise BasestreamError(f"registry request failed for {url}: {exc}") from exc

    @staticmethod
    def _docker_hub_auth_parameters(challenge: str) -> dict[str, str]:
        scheme, separator, raw_parameters = challenge.partition(" ")
        if not separator or scheme.lower() != "bearer":
            raise BasestreamError("Docker Hub returned an unsupported auth challenge")
        try:
            parameters = urllib.request.parse_keqv_list(
                urllib.request.parse_http_list(raw_parameters)
            )
        except (TypeError, ValueError) as exc:
            raise BasestreamError("Docker Hub returned an invalid auth challenge") from exc
        realm = parameters.get("realm")
        parsed_realm = urllib.parse.urlsplit(realm or "")
        if (
            parsed_realm.scheme != "https"
            or parsed_realm.hostname != "auth.docker.io"
            or parsed_realm.username is not None
            or parsed_realm.password is not None
        ):
            raise BasestreamError(f"Docker Hub returned an unsafe token realm: {realm!r}")
        return {str(key): str(value) for key, value in parameters.items()}

    def _verified_https_request(
        self,
        request: urllib.request.Request,
        *,
        timeout: int = 30,
    ) -> tuple[bytes, Any]:
        proxy = self._egress_proxy_url()
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.read(), response.headers
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise BasestreamError(
                f"public registry request failed for {request.full_url}: {exc}"
            ) from exc

    def _docker_hub_request(
        self,
        reference: _RegistryReference,
        path: str,
        *,
        accept: str | None = None,
    ) -> tuple[bytes, Any]:
        url = f"https://{reference.api_registry}/v2/{reference.repository}/{path}"
        headers = {"Accept": accept} if accept else {}
        anonymous_request = urllib.request.Request(url, headers=headers)
        try:
            return self._verified_https_request(anonymous_request)
        except BasestreamError as wrapped:
            cause = wrapped.__cause__
            if not isinstance(cause, urllib.error.HTTPError) or cause.code != 401:
                raise
            challenge = cause.headers.get("WWW-Authenticate", "")

        parameters = self._docker_hub_auth_parameters(challenge)
        query = {
            key: value for key, value in parameters.items() if key in {"service", "scope"} and value
        }
        expected_scope = f"repository:{reference.repository}:pull"
        if query.get("scope") not in {None, expected_scope}:
            raise BasestreamError(
                f"Docker Hub returned an unexpected token scope: {query.get('scope')!r}"
            )
        query["scope"] = expected_scope
        realm = parameters["realm"]
        separator = "&" if urllib.parse.urlsplit(realm).query else "?"
        token_request = urllib.request.Request(f"{realm}{separator}{urllib.parse.urlencode(query)}")
        token_bytes, _ = self._verified_https_request(token_request)
        try:
            token_payload = json.loads(token_bytes)
            token = token_payload.get("token") or token_payload.get("access_token")
        except (json.JSONDecodeError, AttributeError) as exc:
            raise BasestreamError("Docker Hub token response is invalid") from exc
        if not isinstance(token, str) or not token:
            raise BasestreamError("Docker Hub token response contains no token")
        authenticated_request = urllib.request.Request(
            url,
            headers={**headers, "Authorization": f"Bearer {token}"},
        )
        return self._verified_https_request(authenticated_request)

    def _image_metadata(self, image_ref: str) -> dict[str, Any]:
        reference = _parse_registry_reference(image_ref)
        internal_registry = self.settings.registry_repository.partition("/")[0].lower()
        accept = (
            "application/vnd.oci.image.manifest.v1+json,"
            "application/vnd.docker.distribution.manifest.v2+json"
        )
        if reference.registry in {"docker.io", "index.docker.io"}:

            def request(path: str, accept: str | None = None) -> tuple[bytes, Any]:
                return self._docker_hub_request(reference, path, accept=accept)

        elif reference.registry == internal_registry:
            auth = self._registry_auth(reference.registry)
            base = f"https://{reference.api_registry}/v2/{reference.repository}"

            def request(path: str, accept: str | None = None) -> tuple[bytes, Any]:
                return self._registry_request(f"{base}/{path}", auth=auth, accept=accept)

        else:
            raise BasestreamError(f"runtime image registry is not approved: {reference.registry}")
        manifest_bytes, headers = request(f"manifests/{reference.digest}", accept=accept)
        calculated = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        header_digest = headers.get("Docker-Content-Digest")
        if calculated != reference.digest or header_digest != reference.digest:
            raise BasestreamError(
                f"registry manifest digest mismatch: expected={reference.digest} "
                f"header={header_digest} calculated={calculated}"
            )
        try:
            manifest = json.loads(manifest_bytes)
            config_digest = manifest["config"]["digest"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise BasestreamError("registry manifest has no image config") from exc
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", config_digest):
            raise BasestreamError("registry manifest has an invalid image config digest")
        config_bytes, _ = request(f"blobs/{config_digest}")
        calculated_config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
        if calculated_config_digest != config_digest:
            raise BasestreamError(
                f"registry image config digest mismatch: expected={config_digest} "
                f"calculated={calculated_config_digest}"
            )
        try:
            image_config = json.loads(config_bytes)
            config = image_config.get("config") or {}
        except json.JSONDecodeError as exc:
            raise BasestreamError("registry image config is invalid") from exc
        entrypoint = tuple(config.get("Entrypoint") or ())
        if entrypoint not in _ALLOWED_ENTRYPOINTS:
            raise BasestreamError(
                f"image ENTRYPOINT cannot safely receive Harbor args: {entrypoint!r}"
            )
        user = str(config.get("User") or "root")
        if user not in {"", "0", "0:0", "root"} and not _SAFE_USER.fullmatch(user):
            raise BasestreamError(f"image declares an unsafe USER: {user!r}")
        workdir = str(config.get("WorkingDir") or "") or None
        if workdir is not None and not PurePosixPath(workdir).is_absolute():
            raise BasestreamError(f"image declares a non-absolute WorkingDir: {workdir!r}")
        return {
            "config_digest": config_digest,
            "entrypoint": list(entrypoint),
            "image": image_ref,
            "user": user or "root",
            "workdir": workdir,
        }

    def _validate_build_context(self) -> None:
        source = self.environment_dir.resolve()
        for path in source.rglob("*"):
            if path.is_symlink() and not _is_relative_to(path.resolve(), source):
                raise BasestreamError(f"build context symlink escapes its root: {path}")

    def _stage_build_context(self, image_dir: Path) -> Path:
        context_dir = image_dir / "context"
        marker = image_dir / "context.json"
        expected = {"environment_id": self.environment_id}
        if context_dir.is_dir() and marker.is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")) == expected:
                    return context_dir
            except json.JSONDecodeError:
                pass
        self._validate_build_context()
        temporary = image_dir / f"context-{uuid.uuid4().hex}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(self.environment_dir, temporary, symlinks=True)
        if context_dir.exists():
            shutil.rmtree(context_dir)
        temporary.replace(context_dir)
        _atomic_json(marker, expected)
        return context_dir

    def _build_job_manifest(
        self,
        *,
        alias: str,
        context_dir: Path,
        output_dir: Path,
        destination: str,
        timeout_sec: int,
    ) -> dict[str, Any]:
        registry = self.settings.registry_repository.partition("/")[0]
        build_environment = self._proxy_environment()
        container = {
            "name": "kaniko",
            "image": self.settings.kaniko_image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["/kaniko/executor"],
            "args": [
                "--context=dir:///workspace",
                "--dockerfile=/workspace/Dockerfile",
                f"--destination={destination}",
                "--digest-file=/output/image.digest",
                "--cache=true",
                f"--cache-repo={self.settings.kaniko_cache_repository}",
                "--cache-run-layers=true",
                "--cache-copy-layers=true",
                "--cache-ttl=336h",
                "--compressed-caching=false",
                "--snapshot-mode=redo",
                "--image-download-retry=5",
                "--push-retry=5",
                f"--skip-tls-verify-registry={registry}",
                "--verbosity=info",
            ],
            "env": [
                {"name": key, "value": value} for key, value in sorted(build_environment.items())
            ],
            "resources": {
                "limits": {
                    "cpu": self.settings.build_cpu,
                    "memory": self.settings.build_memory,
                    "ephemeral-storage": self.settings.build_storage,
                },
                "requests": {
                    "cpu": self.settings.build_cpu,
                    "memory": self.settings.build_memory,
                    "ephemeral-storage": self.settings.build_storage,
                },
            },
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "runAsGroup": 0,
                "runAsUser": 0,
            },
            "volumeMounts": [
                {
                    "mountPath": "/workspace",
                    "name": "user-pvc",
                    "readOnly": True,
                    "subPath": self.settings.pvc_subpath(context_dir),
                },
                {
                    "mountPath": "/output",
                    "name": "user-pvc",
                    "subPath": self.settings.pvc_subpath(output_dir),
                },
                {
                    "mountPath": "/kaniko/.docker/config.json",
                    "name": "user-pvc",
                    "readOnly": True,
                    "subPath": self.settings.registry_auth_subpath,
                },
                {"mountPath": "/tmp", "name": "build-tmp"},
            ],
        }
        pod_spec: dict[str, Any] = {
            "automountServiceAccountToken": False,
            "containers": [container],
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 30,
            "volumes": [
                {
                    "name": "user-pvc",
                    "persistentVolumeClaim": {"claimName": self.settings.pvc_name},
                },
                {"name": "build-tmp", "emptyDir": {}},
            ],
        }
        if self.settings.harbor_host_ip:
            pod_spec["hostAliases"] = [
                {
                    "hostnames": [registry],
                    "ip": self.settings.harbor_host_ip,
                }
            ]
        return {
            "apiVersion": "batch.volcano.sh/v1alpha1",
            "kind": "Job",
            "metadata": {
                "name": alias,
                "namespace": self.settings.namespace,
                "annotations": {
                    "JobAlias": alias,
                    "scheduling.volcano.sh/queue-name": self.settings.queue,
                },
                "labels": {
                    "app": "harbor-basestream-image-build",
                    "reclaimableByVolcano": "true",
                },
            },
            "spec": {
                "activeDeadlineSeconds": timeout_sec,
                "maxRetry": 1,
                "minAvailable": 1,
                "plugins": {"env": []},
                "policies": [{"action": "RestartJob", "events": ["PodEvicted"]}],
                "priorityClassName": self.settings.priority_class,
                "queue": self.settings.queue,
                "schedulerName": "volcano",
                "tasks": [
                    {
                        "minAvailable": 1,
                        "name": "builder",
                        "replicas": 1,
                        "template": {
                            "metadata": {"labels": {"app": "harbor-basestream-image-build"}},
                            "spec": pod_spec,
                        },
                    }
                ],
                "ttlSecondsAfterFinished": 600,
            },
        }

    def _mirror_copy_source(self, source_ref: str) -> str:
        reference = _parse_registry_reference(source_ref)
        if reference.registry not in {"docker.io", "index.docker.io"}:
            raise BasestreamError(
                f"Skopeo mirroring currently supports Docker Hub only: {source_ref}"
            )
        prefix = self.settings.docker_hub_mirror or "docker.io"
        return f"{prefix}/{reference.repository}@{reference.digest}"

    def _mirror_job_manifest(
        self,
        *,
        alias: str,
        source_ref: str,
        destination: str,
        staging_dir: Path,
        timeout_sec: int,
    ) -> dict[str, Any]:
        registry = self.settings.registry_repository.partition("/")[0]
        source_digest = _parse_registry_reference(source_ref).digest
        container = {
            "name": "skopeo",
            "image": self.settings.skopeo_image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["/bin/sh", "-ceu"],
            "args": [
                _SKOPEO_STAGED_MIRROR_SCRIPT,
                "--",
                f"docker://{self._mirror_copy_source(source_ref)}",
                f"docker://{destination}",
                source_digest,
                str(self.settings.skopeo_parallel_copies),
            ],
            "env": [
                {"name": key, "value": value}
                for key, value in sorted(
                    {
                        **self._proxy_environment(),
                        "HOME": "/tmp",
                        "TMPDIR": "/tmp",
                    }.items()
                )
            ],
            "resources": {
                "limits": {
                    "cpu": self.settings.build_cpu,
                    "memory": self.settings.build_memory,
                    "ephemeral-storage": self.settings.build_storage,
                },
                "requests": {
                    "cpu": self.settings.build_cpu,
                    "memory": self.settings.build_memory,
                    "ephemeral-storage": self.settings.build_storage,
                },
            },
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "runAsGroup": 0,
                "runAsUser": 0,
            },
            "volumeMounts": [
                {
                    "mountPath": "/auth/config.json",
                    "name": "user-pvc",
                    "readOnly": True,
                    "subPath": self.settings.registry_auth_subpath,
                },
                {
                    "mountPath": "/mirror",
                    "name": "user-pvc",
                    "subPath": self.settings.pvc_subpath(staging_dir),
                },
                {"mountPath": "/tmp", "name": "mirror-tmp"},
                {"mountPath": "/var/tmp", "name": "mirror-var"},
                {"mountPath": "/var/lib/containers", "name": "mirror-storage"},
            ],
        }
        pod_spec: dict[str, Any] = {
            "automountServiceAccountToken": False,
            "containers": [container],
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 30,
            "volumes": [
                {
                    "name": "user-pvc",
                    "persistentVolumeClaim": {"claimName": self.settings.pvc_name},
                },
                {"name": "mirror-tmp", "emptyDir": {}},
                {"name": "mirror-var", "emptyDir": {}},
                {"name": "mirror-storage", "emptyDir": {}},
            ],
        }
        if self.settings.harbor_host_ip:
            pod_spec["hostAliases"] = [
                {
                    "hostnames": [registry],
                    "ip": self.settings.harbor_host_ip,
                }
            ]
        return {
            "apiVersion": "batch.volcano.sh/v1alpha1",
            "kind": "Job",
            "metadata": {
                "name": alias,
                "namespace": self.settings.namespace,
                "annotations": {
                    "JobAlias": alias,
                    "scheduling.volcano.sh/queue-name": self.settings.queue,
                },
                "labels": {
                    "app": "harbor-basestream-image-mirror",
                    "reclaimableByVolcano": "true",
                },
            },
            "spec": {
                "activeDeadlineSeconds": timeout_sec,
                "maxRetry": 1,
                "minAvailable": 1,
                "plugins": {"env": []},
                "policies": [{"action": "RestartJob", "events": ["PodEvicted"]}],
                "priorityClassName": self.settings.priority_class,
                "queue": self.settings.queue,
                "schedulerName": "volcano",
                "tasks": [
                    {
                        "minAvailable": 1,
                        "name": "mirror",
                        "replicas": 1,
                        "template": {
                            "metadata": {"labels": {"app": "harbor-basestream-image-mirror"}},
                            "spec": pod_spec,
                        },
                    }
                ],
                "ttlSecondsAfterFinished": 600,
            },
        }

    def _run_kaniko_build(
        self,
        *,
        image_dir: Path,
        context_dir: Path,
        destination: str,
        timeout_sec: int,
    ) -> tuple[str, dict[str, Any]]:
        attempt_dir = image_dir / "builds" / uuid.uuid4().hex
        attempt_dir.mkdir(mode=0o700, parents=True)
        image_name = _slug(self.environment_name, limit=50)
        alias = _slug(
            f"hb-build-{image_name}-{uuid.uuid4().hex[:8]}",
            limit=63,
        )
        manifest = self._build_job_manifest(
            alias=alias,
            context_dir=context_dir,
            output_dir=attempt_dir,
            destination=destination,
            timeout_sec=timeout_sec,
        )
        _atomic_json(attempt_dir / "job-manifest.json", manifest)
        client = _NorthJobClient(self.settings)
        job_name = ""
        try:
            client.submit(manifest)
            job, pod = self._wait_for_job(
                alias,
                timeout_sec=timeout_sec + 180,
                require_terminal=True,
            )
            job_name = self._extract_job_name(job)
            pod_name = str((pod or {}).get("metadata", {}).get("name") or "")
            logs = self._job_logs(pod_name, "kaniko")
            (attempt_dir / "job.log").write_text(logs, encoding="utf-8")
            phase = self._job_phase(job)
            digest_file = attempt_dir / "image.digest"
            digest = (
                digest_file.read_text(encoding="utf-8").strip() if digest_file.is_file() else ""
            )
            if phase not in _SUCCESS_PHASES or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise SandboxBuildFailedError(
                    f"Basestream Kaniko build failed: phase={phase} logs={_tail(logs)}"
                )
            immutable = f"{destination.rsplit(':', 1)[0]}@{digest}"
            metadata = self._image_metadata(immutable)
            receipt = {
                "schema_version": "harbor-basestream-image-v1",
                "environment_id": self.environment_id,
                "immutable_ref": immutable,
                "requested_image": destination,
                "digest": digest,
                "job_name": job_name,
                "metadata": metadata,
                "source": "kaniko",
                "success": True,
            }
            _atomic_json(attempt_dir / "receipt.json", receipt)
            _atomic_json(image_dir / "receipt.json", receipt)
            return immutable, metadata
        except BaseException as exc:
            with contextlib.suppress(Exception):
                _atomic_json(
                    attempt_dir / "receipt.json",
                    {
                        "schema_version": "harbor-basestream-image-v1",
                        "environment_id": self.environment_id,
                        "requested_image": destination,
                        "job_name": job_name or None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "source": "kaniko",
                        "success": False,
                    },
                )
            raise
        finally:
            if not job_name:
                job = self._find_job(alias)
                job_name = self._extract_job_name(job) if job else ""
            if job_name:
                try:
                    _NorthJobClient(self.settings).delete(job_name)
                except Exception as exc:  # noqa: BLE001 - preserve primary build result
                    (attempt_dir / "cleanup-error.log").write_text(
                        f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                    )

    def _build_image(self, image_dir: Path) -> tuple[str, dict[str, Any]]:
        context_dir = self._stage_build_context(image_dir)
        image_name = _slug(self.environment_name, limit=50)
        destination = (
            f"{self.settings.registry_repository.rstrip('/')}/{image_name}:"
            f"env-{self.environment_id[:24]}"
        )
        task_timeout = int(getattr(self.task_env_config, "build_timeout_sec", 0) or 0)
        timeout_sec = max(self.settings.build_timeout_sec, task_timeout)
        return self._run_kaniko_build(
            image_dir=image_dir,
            context_dir=context_dir,
            destination=destination,
            timeout_sec=timeout_sec,
        )

    def _mirror_prebuilt_image(
        self,
        image_dir: Path,
        source_ref: str,
        source_metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        reference = _parse_registry_reference(source_ref)
        image_name = _slug(self.environment_name, limit=50)
        destination = (
            f"{self.settings.registry_repository.rstrip('/')}/{image_name}:"
            f"mirror-{reference.digest.removeprefix('sha256:')[:24]}"
        )
        task_timeout = int(getattr(self.task_env_config, "build_timeout_sec", 0) or 0)
        timeout_sec = max(self.settings.build_timeout_sec, task_timeout)
        staging_dir = image_dir / "mirror-staging"
        if staging_dir.is_symlink():
            raise BasestreamError(f"mirror staging directory cannot be a symlink: {staging_dir}")
        staging_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging_dir.chmod(0o700)
        attempt_dir = image_dir / "mirrors" / uuid.uuid4().hex
        attempt_dir.mkdir(mode=0o700, parents=True)
        alias = _slug(
            f"hb-mirror-{image_name}-{uuid.uuid4().hex[:8]}",
            limit=63,
        )
        manifest = self._mirror_job_manifest(
            alias=alias,
            source_ref=source_ref,
            destination=destination,
            staging_dir=staging_dir,
            timeout_sec=timeout_sec,
        )
        _atomic_json(attempt_dir / "job-manifest.json", manifest)
        receipt_context = {
            "source": "prebuilt-mirror",
            "source_ref": source_ref,
            "copy_source_ref": self._mirror_copy_source(source_ref),
            "source_digest": reference.digest,
            "source_metadata": source_metadata,
            "mirror_tool_image": self.settings.skopeo_image,
            "mirror_mode": "staged-dir-v1",
        }
        client = _NorthJobClient(self.settings)
        job_name = ""
        try:
            client.submit(manifest)
            job, pod = self._wait_for_job(
                alias,
                timeout_sec=timeout_sec + 180,
                require_terminal=True,
            )
            job_name = self._extract_job_name(job)
            pod_name = str((pod or {}).get("metadata", {}).get("name") or "")
            logs = self._job_logs(pod_name, "skopeo")
            (attempt_dir / "job.log").write_text(logs, encoding="utf-8")
            phase = self._job_phase(job)
            if phase not in _SUCCESS_PHASES:
                raise SandboxBuildFailedError(
                    f"Basestream Skopeo mirror failed: phase={phase} logs={_tail(logs)}"
                )
            immutable = f"{destination.rsplit(':', 1)[0]}@{reference.digest}"
            metadata = self._image_metadata(immutable)
            mismatches = {
                key: {"expected": source_metadata.get(key), "actual": metadata.get(key)}
                for key in ("config_digest", "entrypoint", "user", "workdir")
                if metadata.get(key) != source_metadata.get(key)
            }
            if mismatches:
                raise BasestreamError(
                    "Skopeo mirror changed image metadata: "
                    + json.dumps(mismatches, sort_keys=True)
                )
            receipt = {
                "schema_version": "harbor-basestream-image-v1",
                "environment_id": self.environment_id,
                "immutable_ref": immutable,
                "requested_image": destination,
                "digest": reference.digest,
                "job_name": job_name,
                "metadata": metadata,
                "success": True,
                **receipt_context,
            }
            _atomic_json(attempt_dir / "receipt.json", receipt)
            _atomic_json(image_dir / "receipt.json", receipt)
            try:
                shutil.rmtree(staging_dir)
            except OSError as exc:
                (attempt_dir / "staging-cleanup-error.log").write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
            return immutable, metadata
        except BaseException as exc:
            with contextlib.suppress(Exception):
                _atomic_json(
                    attempt_dir / "receipt.json",
                    {
                        "schema_version": "harbor-basestream-image-v1",
                        "environment_id": self.environment_id,
                        "requested_image": destination,
                        "job_name": job_name or None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "success": False,
                        **receipt_context,
                    },
                )
            raise
        finally:
            if not job_name:
                job = self._find_job(alias)
                job_name = self._extract_job_name(job) if job else ""
            if job_name:
                try:
                    _NorthJobClient(self.settings).delete(job_name)
                except Exception as exc:  # noqa: BLE001 - preserve primary mirror result
                    (attempt_dir / "cleanup-error.log").write_text(
                        f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                    )

    def _reuse_image_receipt(
        self,
        receipt_path: Path,
        *,
        required_source: str | None = None,
        source_ref: str | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        if not receipt_path.is_file() or receipt_path.is_symlink():
            return None
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != "harbor-basestream-image-v1"
            or receipt.get("environment_id") != self.environment_id
            or receipt.get("success") is not True
            or (required_source is not None and receipt.get("source") != required_source)
            or (source_ref is not None and receipt.get("source_ref") != source_ref)
        ):
            return None
        image_ref = receipt.get("immutable_ref")
        if not isinstance(image_ref, str):
            return None
        metadata = self._image_metadata(image_ref)
        return image_ref, metadata

    def _ensure_image_sync(self, force_build: bool) -> tuple[str, dict[str, Any]]:
        configured = self.task_env_config.docker_image
        image_dir = self.settings.control_root / "images" / self.environment_id
        image_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if configured:
            image_ref = str(configured)
            reference = _parse_registry_reference(image_ref)
            with _file_lock(image_dir / ".lock"):
                if reference.registry in {"docker.io", "index.docker.io"}:
                    if not force_build:
                        reused = self._reuse_image_receipt(
                            image_dir / "receipt.json",
                            required_source="prebuilt-mirror",
                            source_ref=image_ref,
                        )
                        if reused is not None:
                            return reused
                    metadata = self._image_metadata(image_ref)
                    return self._mirror_prebuilt_image(image_dir, image_ref, metadata)
                metadata = self._image_metadata(image_ref)
                receipt = {
                    "schema_version": "harbor-basestream-image-v1",
                    "environment_id": self.environment_id,
                    "immutable_ref": image_ref,
                    "requested_image": image_ref,
                    "digest": _parse_registry_reference(image_ref).digest,
                    "job_name": None,
                    "metadata": metadata,
                    "source": "prebuilt",
                    "success": True,
                }
                _atomic_json(image_dir / "receipt.json", receipt)
            return image_ref, metadata
        with _file_lock(image_dir / ".lock"):
            if not force_build:
                reused = self._reuse_image_receipt(image_dir / "receipt.json")
                if reused is not None:
                    return reused
            return self._build_image(image_dir)

    async def _ensure_image(self, force_build: bool) -> tuple[str, dict[str, Any]]:
        lock = _BUILD_LOCKS.setdefault(self.environment_id, asyncio.Lock())
        async with lock:
            return await _offload(self._ensure_image_sync, force_build)

    def _egress_proxy_url(self) -> str:
        if self.settings.egress_proxy_url:
            value = self.settings.egress_proxy_url
        else:
            try:
                state = json.loads(self.settings.egress_state_path.read_text(encoding="utf-8"))
                value = state.get("egress_proxy_url")
            except (OSError, json.JSONDecodeError) as exc:
                raise BasestreamError(
                    "public Basestream environments require an existing CPU egress proxy"
                ) from exc
        if not isinstance(value, str):
            raise BasestreamError("Basestream egress proxy URL is missing")
        parsed = urllib.parse.urlsplit(value)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or not parsed.hostname.endswith(f".{self.settings.namespace}.svc.cluster.local")
            or parsed.port != 7890
            or parsed.path not in {"", "/"}
        ):
            raise BasestreamError(f"unsafe Basestream egress proxy URL: {value!r}")
        return value.rstrip("/")

    def _proxy_environment(self) -> dict[str, str]:
        proxy = self._egress_proxy_url()
        registry = self.settings.registry_repository.partition("/")[0]
        registry_host = registry.rsplit(":", 1)[0]
        no_proxy = (
            "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,"
            f"192.168.0.0/16,.svc,.cluster.local,{registry_host}"
        )
        return {
            **{name: proxy for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")},
            "no_proxy": no_proxy,
            "NO_PROXY": no_proxy,
        }

    def _network_environment(self) -> dict[str, str]:
        names = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
        if self.network_policy.network_mode == NetworkMode.NO_NETWORK:
            return {name: "" for name in names}
        if self.network_policy.network_mode != NetworkMode.PUBLIC:
            raise BasestreamError("BasestreamEnvironment does not support network allowlists")
        return self._proxy_environment()

    def _runtime_mounts(self) -> list[dict[str, Any]]:
        mounts: list[dict[str, Any]] = [
            {
                "mountPath": "/harbor/session",
                "name": "user-pvc",
                "subPath": self.settings.pvc_subpath(self._session_dir),
            }
        ]
        targets = {"/harbor/session"}
        for index, mount in enumerate(self._mounts):
            if mount.get("type") != "bind":
                raise BasestreamError("BasestreamEnvironment supports only bind mounts")
            source_value = mount.get("source")
            target_value = mount.get("target")
            if not source_value or not target_value:
                raise BasestreamError("Basestream bind mounts require source and target")
            source = Path(str(source_value)).resolve()
            pvc_root = self.settings.pvc_host_root.resolve()
            if source == pvc_root or not _is_relative_to(source, pvc_root):
                raise BasestreamError(f"Basestream mount source is outside the trial PVC: {source}")
            source.mkdir(mode=0o777, parents=True, exist_ok=True)
            source.chmod(0o777)
            target = str(PurePosixPath(str(target_value)))
            if not target.startswith("/") or target in targets:
                raise BasestreamError(f"invalid or duplicate Basestream mount target: {target}")
            targets.add(target)
            mounts.append(
                {
                    "mountPath": target,
                    "name": "user-pvc",
                    "readOnly": bool(mount.get("read_only")),
                    "subPath": self.settings.pvc_subpath(source),
                }
            )
            self.logger.debug("Prepared Basestream mount %s at index %s", target, index)
        return mounts

    def _runtime_manifest(self, *, alias: str, image_ref: str) -> dict[str, Any]:
        cpu = str(self._effective_cpus or 1)
        memory_mb = int(self._effective_memory_mb or 2048)
        storage_mb = int(self._effective_storage_mb or 10240)
        gpu_count = int(self._effective_gpus)
        if gpu_count < 1:
            raise BasestreamError("BasestreamEnvironment requires at least one GPU")
        gpu_types = tuple(self.task_env_config.gpu_types or ("H100",))
        if not gpu_types or any(value.upper() != "H100" for value in gpu_types):
            raise BasestreamError(
                f"BasestreamEnvironment currently supports only H100, got {gpu_types!r}"
            )
        resources = {
            "limits": {
                "cpu": cpu,
                "memory": f"{memory_mb}Mi",
                "ephemeral-storage": f"{storage_mb}Mi",
                "nvidia.com/gpu": str(gpu_count),
            },
            "requests": {
                "cpu": cpu,
                "memory": f"{memory_mb}Mi",
                "ephemeral-storage": f"{storage_mb}Mi",
                "nvidia.com/gpu": str(gpu_count),
            },
        }
        environment = {**self._startup_env(), **self._network_environment()}
        environment.update(
            {
                "HARBOR_BASESTREAM_SESSION": self.session_id,
            }
        )
        container: dict[str, Any] = {
            "name": "main",
            "image": image_ref,
            "imagePullPolicy": "IfNotPresent",
            # Intentionally omit `command`: Kubernetes then retains image ENTRYPOINT.
            "args": [
                "/bin/bash",
                "--norc",
                "-lc",
                (
                    "set -euo pipefail\n"
                    "touch /tmp/.harbor-basestream-ready\n"
                    "trap 'exit 0' TERM INT\n"
                    "while true; do sleep 3600 & wait $!; done\n"
                ),
            ],
            "env": [{"name": key, "value": value} for key, value in sorted(environment.items())],
            "readinessProbe": {
                "exec": {"command": ["test", "-f", "/tmp/.harbor-basestream-ready"]},
                "failureThreshold": 60,
                "periodSeconds": 2,
                "timeoutSeconds": 1,
            },
            "resources": resources,
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "runAsGroup": 0,
                "runAsUser": 0,
            },
            "volumeMounts": [
                *self._runtime_mounts(),
                {"mountPath": "/tmp", "name": "runtime-tmp"},
                {"mountPath": "/dev/shm", "name": "dshm"},
            ],
        }
        if self.task_env_config.workdir:
            container["workingDir"] = self.task_env_config.workdir
        node_expressions: list[dict[str, Any]] = []
        if self.settings.excluded_gpu_nodes:
            node_expressions.append(
                {
                    "key": "kubernetes.io/hostname",
                    "operator": "NotIn",
                    "values": list(self.settings.excluded_gpu_nodes),
                }
            )
        pod_spec: dict[str, Any] = {
            "automountServiceAccountToken": False,
            "containers": [container],
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 60,
            "volumes": [
                {
                    "name": "user-pvc",
                    "persistentVolumeClaim": {"claimName": self.settings.pvc_name},
                },
                {"name": "runtime-tmp", "emptyDir": {}},
                {
                    "name": "dshm",
                    "emptyDir": {
                        "medium": "Memory",
                        "sizeLimit": f"{max(1024, min(memory_mb // 2, 65536))}Mi",
                    },
                },
            ],
        }
        if node_expressions:
            pod_spec["affinity"] = {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [{"matchExpressions": node_expressions}]
                    }
                }
            }
        return {
            "apiVersion": "batch.volcano.sh/v1alpha1",
            "kind": "Job",
            "metadata": {
                "name": alias,
                "namespace": self.settings.namespace,
                "annotations": {
                    "JobAlias": alias,
                    "harbor/session-id": self.session_id,
                    "scheduling.volcano.sh/queue-name": self.settings.queue,
                },
                "labels": {
                    "app": "harbor-basestream-runtime",
                    "harbor-session": _slug(self.session_id, limit=48),
                    "reclaimableByVolcano": "true",
                },
            },
            "spec": {
                "activeDeadlineSeconds": self.settings.runtime_lifetime_sec,
                "maxRetry": 0,
                "minAvailable": 1,
                "plugins": {"env": []},
                "priorityClassName": self.settings.priority_class,
                "queue": self.settings.queue,
                "schedulerName": "volcano",
                "tasks": [
                    {
                        "minAvailable": 1,
                        "name": "environment",
                        "replicas": 1,
                        "template": {
                            "metadata": {
                                "labels": {
                                    "app": "harbor-basestream-runtime",
                                    "harbor-session": _slug(self.session_id, limit=48),
                                }
                            },
                            "spec": pod_spec,
                        },
                    }
                ],
                "ttlSecondsAfterFinished": 604800,
            },
        }

    def _effective_image_user(self, user: str | int | None) -> str:
        requested = self._resolve_user(user)
        if requested is None:
            requested = self._image_user
        value = str(requested or "root")
        if not _SAFE_USER.fullmatch(value):
            raise BasestreamError(f"unsafe container user: {value!r}")
        return value.split(":", 1)[0]

    def _kubectl_exec_command(self, remote: list[str], *, stdin: bool = False) -> list[str]:
        if not self._pod_name:
            raise BasestreamError("Basestream runtime Pod is not available")
        args = [self.settings.kubectl, "exec"]
        if stdin:
            args.append("-i")
        return [
            *args,
            "-n",
            self.settings.namespace,
            self._pod_name,
            "-c",
            "main",
            "--",
            *remote,
        ]

    async def _validate_runtime(self) -> dict[str, Any]:
        validation: dict[str, Any] = {}
        token = await self.exec(
            "test ! -e /var/run/secrets/kubernetes.io/serviceaccount/token",
            user="root",
            timeout_sec=30,
        )
        if token.return_code != 0:
            raise BasestreamError("Kubernetes service-account token is visible in runtime")
        validation["service_account_token_absent"] = True
        gpu = await self.exec(
            "nvidia-smi --query-gpu=name --format=csv,noheader",
            user="root",
            timeout_sec=60,
        )
        names = [line.strip() for line in (gpu.stdout or "").splitlines() if line.strip()]
        if gpu.return_code != 0 or len(names) != self._effective_gpus:
            raise BasestreamError(
                f"allocated GPU count mismatch: expected={self._effective_gpus} names={names!r}"
            )
        if any("H100" not in name for name in names):
            raise BasestreamError(f"Basestream scheduled a non-H100 GPU: {names!r}")
        validation["gpu_names"] = names
        storage_mb = int(self._effective_storage_mb or 0)
        if storage_mb:
            disk = await self.exec("df -Pm / | tail -n 1", user="root", timeout_sec=30)
            try:
                available_mb = int((disk.stdout or "").split()[3])
            except (IndexError, ValueError) as exc:
                raise BasestreamError(
                    f"cannot parse runtime disk capacity: {disk.stdout!r}"
                ) from exc
            if disk.return_code != 0 or available_mb < storage_mb:
                raise BasestreamError(
                    f"runtime disk is below task request: requested={storage_mb}MB "
                    f"available={available_mb}MB"
                )
            validation["available_storage_mb"] = available_mb
        image_user = self._effective_image_user(None)
        if image_user not in {"0", "root"}:
            user_check = await self.exec(
                f"command -v su >/dev/null && getent passwd {shlex.quote(image_user)} >/dev/null",
                user="root",
                timeout_sec=30,
            )
            if user_check.return_code != 0:
                raise BasestreamError(
                    f"image USER {image_user!r} cannot be selected with su/getent"
                )
        validation["effective_image_user"] = image_user
        return validation

    @override
    async def start(self, force_build: bool) -> None:
        self._ensure_host_layout()
        image_ref, metadata = await self._ensure_image(force_build)
        self._image_ref = image_ref
        self._image_user = str(metadata.get("user") or "root")
        self._image_entrypoint = tuple(metadata.get("entrypoint") or ())
        self._image_workdir = str(metadata["workdir"]) if metadata.get("workdir") else None
        alias = _slug(f"hb-env-{self.environment_name}-{uuid.uuid4().hex[:8]}", limit=63)
        self._job_alias = alias
        manifest = self._runtime_manifest(alias=alias, image_ref=image_ref)
        _atomic_json(self._session_dir / "job-manifest.json", manifest)
        try:
            await _offload(_NorthJobClient(self.settings).submit, manifest)
            job, pod = await _offload(
                self._wait_for_job,
                alias,
                timeout_sec=self.settings.start_timeout_sec,
                require_terminal=False,
            )
            self._runtime_job_name = self._extract_job_name(job)
            self._pod_name = str((pod or {}).get("metadata", {}).get("name") or "")
            validation = await self._validate_runtime()
            _atomic_json(
                self._session_dir / "runtime-receipt.json",
                {
                    "schema_version": "harbor-basestream-runtime-v1",
                    "success": True,
                    "session_id": self.session_id,
                    "job_alias": alias,
                    "job_name": self._runtime_job_name,
                    "pod_name": self._pod_name,
                    "node_name": str((pod or {}).get("spec", {}).get("nodeName") or ""),
                    "image": image_ref,
                    "validation": validation,
                    "resources": manifest["spec"]["tasks"][0]["template"]["spec"]["containers"][0][
                        "resources"
                    ],
                    "validated_at_unix": time.time(),
                },
            )
        except BaseException as exc:
            if not self._runtime_job_name:
                job = await _offload(self._find_job, alias)
                self._runtime_job_name = self._extract_job_name(job) if job else None
            failed_job_name = self._runtime_job_name
            failed_pod_name = self._pod_name
            cleanup_error: str | None = None
            if failed_job_name:
                try:
                    await _offload(self._delete_job, failed_job_name)
                except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                    cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                else:
                    self._runtime_job_name = None
                    self._pod_name = None
            with contextlib.suppress(Exception):
                _atomic_json(
                    self._session_dir / "runtime-receipt.json",
                    {
                        "schema_version": "harbor-basestream-runtime-v1",
                        "success": False,
                        "session_id": self.session_id,
                        "job_alias": alias,
                        "job_name": failed_job_name,
                        "pod_name": failed_pod_name,
                        "image": image_ref,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "cleanup_error": cleanup_error,
                        "failed_at_unix": time.time(),
                    },
                )
            raise

    @override
    async def stop(self, delete: bool) -> None:
        job_name = self._runtime_job_name
        if not job_name:
            return
        if delete:
            await _offload(self._delete_job, job_name)
        elif self._pod_name:
            await _offload(
                self._run_text,
                self._kubectl_exec_command(["/bin/kill", "-TERM", "1"]),
                timeout=30,
                check=False,
            )
        _atomic_json(
            self._session_dir / "stop-receipt.json",
            {
                "schema_version": "harbor-basestream-stop-v1",
                "job_name": job_name,
                "delete": delete,
                "stopped_at_unix": time.time(),
            },
        )
        self._runtime_job_name = None
        self._pod_name = None

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._pod_name:
            return ExecResult(
                stdout=None,
                stderr="Basestream runtime is not started",
                return_code=1,
            )
        target_user = self._effective_image_user(user)
        effective_cwd = effective_exec_cwd(
            cwd,
            self.task_env_config.workdir,
            self._image_workdir,
        )
        merged_env = self._merge_env(env) or {}
        merged_env.update(self._network_environment())
        shell_parts = [
            *(f"export {key}={shlex.quote(value)}" for key, value in sorted(merged_env.items())),
            f"cd {shlex.quote(str(effective_cwd))}" if effective_cwd else "",
            f"exec /bin/bash --norc -lc {shlex.quote(command)}",
        ]
        shell = "\n".join(part for part in shell_parts if part)
        if target_user not in {"0", "root"}:
            if target_user.isdigit():
                resolve = f"target_user=$(getent passwd {target_user} | cut -d: -f1)"
                target = '"${target_user}"'
            else:
                resolve = ""
                target = shlex.quote(target_user)
            shell = "\n".join(
                part
                for part in (
                    resolve,
                    f"test -n {target}",
                    f"exec su {target} -s /bin/bash -c {shlex.quote(shell)}",
                )
                if part
            )
        remote = ["/bin/bash", "--norc", "-lc", shell]
        if timeout_sec is not None:
            remote = [
                "timeout",
                "--signal=TERM",
                "--kill-after=30s",
                f"{max(1, math.ceil(timeout_sec))}s",
                *remote,
            ]
        try:
            completed = await _offload(
                self._run_text,
                self._kubectl_exec_command(remote),
                timeout=(timeout_sec + 60 if timeout_sec is not None else None),
                check=False,
            )
        except BasestreamError as exc:
            return ExecResult(stdout=None, stderr=str(exc), return_code=1)
        callback = self._output_callback()
        if callback is not None:
            if completed.stdout:
                await callback(completed.stdout, "stdout")
            if completed.stderr:
                await callback(completed.stderr, "stderr")
        return ExecResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
        )

    def _stream_to_pod(self, remote: list[str], source: BinaryIO) -> None:
        command = self._kubectl_exec_command(remote, stdin=True)
        completed = subprocess.run(
            command,
            stdin=source,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = _tail(completed.stderr.decode("utf-8", errors="replace"))
            raise BasestreamError(f"upload to Basestream Pod failed: {detail}")

    def _stream_from_pod(self, remote: list[str], destination: BinaryIO) -> None:
        command = self._kubectl_exec_command(remote)
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=destination,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            detail = _tail(completed.stderr.decode("utf-8", errors="replace"))
            raise BasestreamError(f"download from Basestream Pod failed: {detail}")

    @staticmethod
    def _write_safe_tar(source_dir: Path, destination: Path) -> None:
        with tarfile.open(destination, "w") as archive:
            for path in sorted(source_dir.rglob("*")):
                if path.is_symlink():
                    raise BasestreamError(f"refusing to upload symlink: {path}")
                if not path.is_dir() and not path.is_file():
                    raise BasestreamError(f"refusing to upload special file: {path}")
                archive.add(
                    path,
                    arcname=path.relative_to(source_dir).as_posix(),
                    recursive=False,
                )

    @staticmethod
    def _extract_safe_tar(archive_path: Path, target_dir: Path) -> None:
        with tarfile.open(archive_path, "r") as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise BasestreamError(f"refusing unsafe artifact archive member: {member.name}")
            archive.extractall(target_dir, filter="data")

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"upload source is not a regular file: {source}")
        parent = str(PurePosixPath(target_path).parent)
        mkdir = await self.exec(
            f"mkdir -p {shlex.quote(parent)}",
            user="root",
            timeout_sec=60,
        )
        if mkdir.return_code != 0:
            raise BasestreamError(f"failed to create upload target: {mkdir.stderr}")
        remote = [
            "/bin/bash",
            "--norc",
            "-lc",
            f"cat > {shlex.quote(target_path)} && chmod {source.stat().st_mode & 0o777:o} "
            f"{shlex.quote(target_path)}",
        ]
        with source.open("rb") as handle:
            await _offload(self._stream_to_pod, remote, handle)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if source.is_symlink() or not source.is_dir():
            raise FileNotFoundError(f"upload source is not a directory: {source}")
        mkdir = await self.exec(
            f"mkdir -p {shlex.quote(target_dir)}",
            user="root",
            timeout_sec=60,
        )
        if mkdir.return_code != 0:
            raise BasestreamError(f"failed to create upload directory: {mkdir.stderr}")
        self._session_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        archive_path = self._session_dir / f"upload-{uuid.uuid4().hex}.tar"
        try:
            await _offload(self._write_safe_tar, source, archive_path)
            remote = [
                "tar",
                "-xf",
                "-",
                "-C",
                target_dir,
            ]
            with archive_path.open("rb") as handle:
                await _offload(self._stream_to_pod, remote, handle)
        finally:
            archive_path.unlink(missing_ok=True)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            remote = [
                "/bin/bash",
                "--norc",
                "-lc",
                f"test -f {shlex.quote(source_path)} && cat {shlex.quote(source_path)}",
            ]
            with temporary.open("wb") as handle:
                await _offload(self._stream_from_pod, remote, handle)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        descriptor, archive_name = tempfile.mkstemp(
            prefix=".harbor-basestream-download-",
            suffix=".tar",
            dir=target.parent,
        )
        os.close(descriptor)
        archive_path = Path(archive_name)
        try:
            remote = [
                "/bin/bash",
                "--norc",
                "-lc",
                f"test -d {shlex.quote(source_dir)} && tar -C {shlex.quote(source_dir)} -cf - .",
            ]
            with archive_path.open("wb") as handle:
                await _offload(self._stream_from_pod, remote, handle)
            await _offload(self._extract_safe_tar, archive_path, target)
        finally:
            archive_path.unlink(missing_ok=True)

    @override
    async def prepare_logs_for_host(self) -> None:
        for mount in self._mounts:
            source = mount.get("source")
            if source and Path(str(source)).is_dir() and not mount.get("read_only"):
                Path(str(source)).chmod(0o777)
