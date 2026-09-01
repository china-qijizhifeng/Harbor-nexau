"""Build and verify the project-owned E2B Docker-in-Docker template."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from dotenv import load_dotenv

TEMPLATE_CONTEXT = Path(__file__).resolve().parent / "e2b_dind"

TEMPLATE_NAME = "harbor-nexau-dind"
TEMPLATE_VERSION = "v1"
BASE_TEMPLATE_ID = "qusopjo1y7f5s6po0hyl"
BASE_TEMPLATE_BUILD_ID = "1ac1ebce-6cc3-43cf-8c30-1483468bbf38"
DOCKER_VERSION = "29.2.1"
COMPOSE_VERSION = "5.0.2"
CONTAINERD_VERSION = "2.2.1"
BUILDX_VERSION = "0.31.1"
STANDALONE_UBUNTU_VARIANT = "25.04"
STANDALONE_SOURCE_ID = f"ubuntu:{STANDALONE_UBUNTU_VARIANT}"
STANDALONE_SOURCE_BUILD = f"docker-ce-{DOCKER_VERSION}-compose-{COMPOSE_VERSION}-pinned"
REGISTRY_MIRRORS = ("https://docker.1ms.run",)
SMOKE_IMAGE = "registry-1.docker.io/library/alpine:3.20"

# These defaults preserve the original 8-CPU/32-GiB published profile. Larger
# profiles are built explicitly and loaded from their packaged manifests; GPU
# resources remain unsupported by the E2B DinD adapter.
DEFAULT_CPU_COUNT = 8
DEFAULT_MEMORY_MB = 32_768
DEFAULT_SANDBOX_TIMEOUT_SEC = 900

_API_URL_ENV = "E2B_API_URL"
_DOMAIN_ENV = "E2B_DOMAIN"
_API_KEY_ENV = "E2B_API_KEY"
_NO_PROXY_HOSTS_ENV = "HARBOR_NEXAU_NO_PROXY_HOSTS"


@dataclass(frozen=True)
class E2BConnection:
    """Non-secret connection settings plus the API key used by the SDK."""

    api_key: str
    api_url: str | None = None
    domain: str | None = None

    def sdk_options(self) -> dict[str, str]:
        options = {"api_key": self.api_key}
        if self.api_url is not None:
            options["api_url"] = self.api_url
        if self.domain is not None:
            options["domain"] = self.domain
        return options


@dataclass(frozen=True)
class DindBuildResult:
    template_id: str
    build_id: str
    name: str
    tags: tuple[str, ...]
    base_template_id: str
    base_build_id: str
    cpu_count: int
    memory_mb: int


@dataclass(frozen=True)
class DindSmokeResult:
    template: str
    sandbox_id: str
    base_build_id: str
    docker_version: str
    compose_version: str
    cpu_count: int
    memory_mb: int
    root_disk_mb: int
    available_disk_mb: int


@dataclass(frozen=True)
class PublishedDindTemplate:
    """Capacity metadata for a project-published DinD template."""

    reference: str
    cpu_count: int
    memory_mb: int
    storage_mb: int


def load_published_dind_templates(
    context: Path = TEMPLATE_CONTEXT,
) -> dict[str, PublishedDindTemplate]:
    """Load packaged, smoke-verified template manifests by full reference."""

    profiles: dict[str, PublishedDindTemplate] = {}
    for manifest_path in sorted(context.glob("manifest*.json")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        reference = str(data.get("reference", "")).strip()
        resources = data.get("resources", {})
        if not reference:
            raise ValueError(f"Published DinD manifest {manifest_path} has no reference")
        profile = PublishedDindTemplate(
            reference=reference,
            cpu_count=int(resources["cpu_count"]),
            memory_mb=int(resources["memory_mb"]),
            storage_mb=int(resources["observed_root_disk_mb"]),
        )
        if min(profile.cpu_count, profile.memory_mb, profile.storage_mb) <= 0:
            raise ValueError(f"Published DinD manifest {manifest_path} has invalid resources")
        if reference in profiles:
            raise ValueError(f"Duplicate published DinD template reference {reference!r}")
        profiles[reference] = profile
    return profiles


PUBLISHED_DIND_TEMPLATES = load_published_dind_templates()


def _clean_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _merge_no_proxy_hosts(hosts: list[str]) -> None:
    configured = [
        value.strip()
        for value in os.environ.get(_NO_PROXY_HOSTS_ENV, "").split(",")
        if value.strip()
    ]
    configured.extend(host for host in hosts if host)

    for env_name in ("NO_PROXY", "no_proxy"):
        existing = [
            value.strip() for value in os.environ.get(env_name, "").split(",") if value.strip()
        ]
        merged = list(existing)
        for host in configured:
            if host not in merged:
                merged.append(host)
        if merged:
            os.environ[env_name] = ",".join(merged)


def load_e2b_connection(env_file: Path | str | None = None) -> E2BConnection:
    """Load E2B settings without overriding already-exported variables."""

    resolved_env_file = Path(env_file) if env_file is not None else Path(".env")
    load_dotenv(resolved_env_file, override=False)

    api_key = _clean_env(_API_KEY_ENV)
    api_url = _clean_env(_API_URL_ENV)
    domain = _clean_env(_DOMAIN_ENV)

    if api_key is None:
        raise ValueError(f"{_API_KEY_ENV} is required")
    if (api_url is None) != (domain is None):
        raise ValueError(
            f"{_API_URL_ENV} and {_DOMAIN_ENV} must either both be set or both be empty"
        )

    direct_hosts: list[str] = []
    if api_url is not None:
        api_hostname = urlparse(api_url).hostname
        if api_hostname:
            direct_hosts.append(api_hostname)
    if domain is not None:
        direct_hosts.extend((domain, f".{domain}"))
    _merge_no_proxy_hosts(direct_hosts)

    return E2BConnection(api_key=api_key, api_url=api_url, domain=domain)


def _load_e2b_sdk() -> tuple[Any, Any]:
    try:
        from e2b import Sandbox, Template
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise RuntimeError(
            "E2B SDK is not installed; run `uv sync --locked --extra cloud`"
        ) from exc
    return Template, Sandbox


def _tag_build_id(template_api: Any, template: str, tag: str, **options: str) -> str:
    tags = template_api.get_tags(template, **options)
    matches = [item.build_id for item in tags if item.tag == tag]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {tag!r} tag for E2B template {template!r}, found {len(matches)}"
        )
    return str(matches[0])


def validate_base_template(
    template_api: Any,
    connection: E2BConnection,
    *,
    allow_base_drift: bool = False,
    trust_pinned_base: bool = False,
) -> str:
    """Return the base build ID, rejecting an unexpected replacement.

    ``trust_pinned_base`` is for deployments that may consume a public base but
    are not allowed to inspect its tags. It explicitly accepts that the caller
    cannot verify the current default build identity; the derived-template smoke
    still validates the runtime software and provenance contract before use.
    """

    if trust_pinned_base:
        return BASE_TEMPLATE_BUILD_ID

    current_build_id = _tag_build_id(
        template_api,
        BASE_TEMPLATE_ID,
        "default",
        **connection.sdk_options(),
    )
    if current_build_id != BASE_TEMPLATE_BUILD_ID and not allow_base_drift:
        raise RuntimeError(
            "The pinned DinD base template changed: "
            f"expected {BASE_TEMPLATE_BUILD_ID}, got {current_build_id}. "
            "Inspect the new base before using --allow-base-drift."
        )
    return current_build_id


def _provenance(
    base_build_id: str,
    name: str,
    version: str,
    *,
    base_template_id: str = BASE_TEMPLATE_ID,
    source_image: str | None = None,
) -> dict[str, str]:
    result = {
        "name": name,
        "version": version,
        "base_template_id": base_template_id,
        "base_build_id": base_build_id,
        "docker_version": DOCKER_VERSION,
        "compose_version": COMPOSE_VERSION,
    }
    if source_image is not None:
        result["source_image"] = source_image
    return result


def _standalone_docker_install_command(ubuntu_variant: str) -> str:
    if ubuntu_variant != STANDALONE_UBUNTU_VARIANT:
        raise ValueError(
            f"Unsupported standalone Ubuntu variant {ubuntu_variant!r}; "
            f"expected {STANDALONE_UBUNTU_VARIANT!r}"
        )

    package_suffix = f"1~ubuntu.{ubuntu_variant}~plucky"
    return "\n".join(
        (
            "set -eux",
            "export DEBIAN_FRONTEND=noninteractive",
            "apt-get update",
            "apt-get install -y --no-install-recommends ca-certificates curl python3",
            "install -m 0755 -d /etc/apt/keyrings",
            "curl -fsSL https://download.docker.com/linux/ubuntu/gpg "
            "-o /etc/apt/keyrings/docker.asc",
            "chmod a+r /etc/apt/keyrings/docker.asc",
            ". /etc/os-release",
            "printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] "
            "https://download.docker.com/linux/ubuntu %s stable\\n' "
            '"$(dpkg --print-architecture)" "${UBUNTU_CODENAME:-$VERSION_CODENAME}" '
            "> /etc/apt/sources.list.d/docker.list",
            "apt-get update",
            "apt-get install -y --no-install-recommends "
            f"docker-ce='5:{DOCKER_VERSION}-{package_suffix}' "
            f"docker-ce-cli='5:{DOCKER_VERSION}-{package_suffix}' "
            f"containerd.io='{CONTAINERD_VERSION}-{package_suffix}' "
            f"docker-buildx-plugin='{BUILDX_VERSION}-{package_suffix}' "
            f"docker-compose-plugin='{COMPOSE_VERSION}-{package_suffix}'",
            "apt-mark hold docker-ce docker-ce-cli containerd.io "
            "docker-buildx-plugin docker-compose-plugin",
            "systemctl enable containerd.service docker.service docker.socket",
            "rm -rf /var/lib/apt/lists/*",
        )
    )


def _template_builder(
    template_api: Any,
    base_build_id: str,
    name: str,
    version: str,
    *,
    base_template_id: str = BASE_TEMPLATE_ID,
    standalone_ubuntu: str | None = None,
) -> Any:
    provenance = json.dumps(
        _provenance(
            base_build_id,
            name,
            version,
            base_template_id=base_template_id,
            source_image=(f"ubuntu:{standalone_ubuntu}" if standalone_ubuntu else None),
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    write_provenance = f"printf '%s\\n' {shlex.quote(provenance)} > /etc/harbor-nexau-dind.json"
    verify_installation = " && ".join(
        (
            "test -x /usr/bin/dockerd",
            "test -x /usr/bin/docker",
            f"docker --version | grep -F {shlex.quote(DOCKER_VERSION)}",
            f"docker compose version --short | grep -Fx {shlex.quote(COMPOSE_VERSION)}",
            *(
                ("systemctl is-enabled docker.service", "systemctl is-enabled docker.socket")
                if standalone_ubuntu
                else ()
            ),
            "test -s /etc/docker/daemon.json",
            "test -x /usr/local/bin/harbor-nexau-dind-ready",
        )
    )

    template = template_api(file_context_path=TEMPLATE_CONTEXT)
    if standalone_ubuntu is not None:
        builder = template.from_ubuntu_image(standalone_ubuntu).run_cmd(
            _standalone_docker_install_command(standalone_ubuntu),
            user="root",
        )
    else:
        builder = template.from_template(base_template_id)

    return (
        builder.copy("daemon.json", "/etc/docker/daemon.json", user="root", mode=0o644)
        .copy(
            "harbor-nexau-dind-ready",
            "/usr/local/bin/harbor-nexau-dind-ready",
            user="root",
            mode=0o755,
        )
        .run_cmd(write_provenance, user="root")
        .run_cmd(verify_installation, user="root")
    )


def _build_log(entry: Any) -> None:
    level = getattr(entry, "level", "info")
    message = getattr(entry, "message", str(entry))
    if message:
        print(f"[E2B build:{level}] {message}", flush=True)


def build_dind_template(
    connection: E2BConnection,
    *,
    name: str = TEMPLATE_NAME,
    version: str = TEMPLATE_VERSION,
    cpu_count: int = DEFAULT_CPU_COUNT,
    memory_mb: int = DEFAULT_MEMORY_MB,
    replace: bool = False,
    allow_base_drift: bool = False,
    trust_pinned_base: bool = False,
    standalone_ubuntu: str | None = None,
    skip_cache: bool = False,
) -> DindBuildResult:
    """Build a versioned DinD template and make the new build the default."""

    if cpu_count <= 0:
        raise ValueError("cpu_count must be positive")
    if memory_mb <= 0:
        raise ValueError("memory_mb must be positive")
    if not name or ":" in name:
        raise ValueError("name must be a non-empty E2B template name without a tag")
    if not version or ":" in version:
        raise ValueError("version must be a non-empty E2B tag")

    template_api, _ = _load_e2b_sdk()
    options = connection.sdk_options()
    if standalone_ubuntu is not None:
        if allow_base_drift or trust_pinned_base:
            raise ValueError(
                "Standalone Ubuntu builds do not use --allow-base-drift or --trust-pinned-base"
            )
        if standalone_ubuntu != STANDALONE_UBUNTU_VARIANT:
            raise ValueError(
                f"Unsupported standalone Ubuntu variant {standalone_ubuntu!r}; "
                f"expected {STANDALONE_UBUNTU_VARIANT!r}"
            )
        base_template_id = f"ubuntu:{standalone_ubuntu}"
        base_build_id = STANDALONE_SOURCE_BUILD
    else:
        base_template_id = BASE_TEMPLATE_ID
        base_build_id = validate_base_template(
            template_api,
            connection,
            allow_base_drift=allow_base_drift,
            trust_pinned_base=trust_pinned_base,
        )

    if template_api.alias_exists(name, **options) and not replace:
        raise RuntimeError(
            f"E2B template {name!r} already exists; pass --replace to publish a new build"
        )

    candidate_tag = f"candidate-{uuid4().hex[:12]}"
    build_info = template_api.build(
        _template_builder(
            template_api,
            base_build_id,
            name,
            version,
            base_template_id=base_template_id,
            standalone_ubuntu=standalone_ubuntu,
        ),
        name,
        tags=[candidate_tag],
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        skip_cache=skip_cache,
        on_build_logs=_build_log,
        **options,
    )

    candidate_build_id = _tag_build_id(
        template_api,
        build_info.template_id,
        candidate_tag,
        **options,
    )
    if candidate_build_id != build_info.build_id:
        raise RuntimeError(
            f"E2B tag {name}:{candidate_tag} points to {candidate_build_id}, "
            f"not the new build {build_info.build_id}"
        )

    # E2B moves requested tags before a build completes. Build under a disposable
    # candidate, then atomically move both stable tags only after start/ready pass.
    assigned = template_api.assign_tags(
        f"{name}:{candidate_tag}",
        [version, "default"],
        **options,
    )
    if str(assigned.build_id) != str(build_info.build_id):
        raise RuntimeError(
            f"E2B assigned stable tags from unexpected build {assigned.build_id}; "
            f"expected {build_info.build_id}"
        )

    for required_tag in ("default", version):
        tagged_build_id = _tag_build_id(
            template_api,
            build_info.template_id,
            required_tag,
            **options,
        )
        if tagged_build_id != build_info.build_id:
            raise RuntimeError(
                f"E2B tag {name}:{required_tag} points to {tagged_build_id}, "
                f"not the new build {build_info.build_id}"
            )

    template_api.remove_tags(name, candidate_tag, **options)

    return DindBuildResult(
        template_id=str(build_info.template_id),
        build_id=str(build_info.build_id),
        name=name,
        tags=("default", version),
        base_template_id=base_template_id,
        base_build_id=base_build_id,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
    )


def _run_checked(sandbox: Any, command: str, *, timeout: int = 120) -> str:
    result = sandbox.commands.run(
        command,
        user="root",
        timeout=timeout,
        request_timeout=timeout + 30,
    )
    if result.exit_code != 0:
        raise RuntimeError(
            f"E2B smoke command failed with exit code {result.exit_code}:\n"
            f"{command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def smoke_dind_template(
    connection: E2BConnection,
    *,
    name: str = TEMPLATE_NAME,
    version: str = TEMPLATE_VERSION,
    timeout_sec: int = DEFAULT_SANDBOX_TIMEOUT_SEC,
    expected_base_template_id: str = BASE_TEMPLATE_ID,
) -> DindSmokeResult:
    """Launch the template and exercise Docker plus lifecycle-critical Compose APIs."""

    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")

    _, sandbox_api = _load_e2b_sdk()
    template = f"{name}:{version}"
    sandbox = None
    compose_dir = "/tmp/harbor-nexau-dind-smoke"
    try:
        sandbox = sandbox_api.create(
            template,
            timeout=timeout_sec,
            metadata={
                "purpose": "harbor-nexau-dind-smoke",
                "template": template,
            },
            **connection.sdk_options(),
        )
        _run_checked(sandbox, "/usr/local/bin/harbor-nexau-dind-ready", timeout=180)

        provenance = json.loads(_run_checked(sandbox, "cat /etc/harbor-nexau-dind.json"))
        expected_provenance = {
            "name": name,
            "version": version,
            "base_template_id": expected_base_template_id,
            "docker_version": DOCKER_VERSION,
            "compose_version": COMPOSE_VERSION,
        }
        for key, expected_value in expected_provenance.items():
            if provenance.get(key) != expected_value:
                raise RuntimeError(
                    f"Template provenance field {key!r} is {provenance.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        base_build_id = str(provenance.get("base_build_id", ""))
        if not base_build_id:
            raise RuntimeError("Template provenance does not contain a base build ID")

        daemon_config = json.loads(_run_checked(sandbox, "cat /etc/docker/daemon.json"))
        if daemon_config.get("registry-mirrors") != list(REGISTRY_MIRRORS):
            raise RuntimeError(
                "Template Docker registry mirrors do not match the project config: "
                f"{daemon_config.get('registry-mirrors')!r}"
            )

        facts_json = _run_checked(
            sandbox,
            """python3 - <<'PY'
import json
import os
from pathlib import Path

meminfo = Path('/proc/meminfo').read_text().splitlines()
mem_kib = int(next(line.split()[1] for line in meminfo if line.startswith('MemTotal:')))
stat = os.statvfs('/')
facts = {
    'cpu_count': os.cpu_count(),
    'memory_mb': mem_kib // 1024,
    'root_disk_mb': stat.f_blocks * stat.f_frsize // 1024 // 1024,
    'available_disk_mb': stat.f_bavail * stat.f_frsize // 1024 // 1024,
}
print(json.dumps(facts))
PY""",
        )
        facts = json.loads(facts_json)
        docker_version = _run_checked(
            sandbox,
            "docker version --format '{{.Server.Version}}'",
        )
        compose_version = _run_checked(sandbox, "docker compose version --short")
        if docker_version != DOCKER_VERSION:
            raise RuntimeError(
                f"Expected Docker {DOCKER_VERSION}, got {docker_version or '<empty>'}"
            )
        if compose_version != COMPOSE_VERSION:
            raise RuntimeError(
                f"Expected Docker Compose {COMPOSE_VERSION}, got {compose_version or '<empty>'}"
            )

        _run_checked(sandbox, f"docker pull {SMOKE_IMAGE}", timeout=600)
        _run_checked(sandbox, f"docker tag {SMOKE_IMAGE} alpine:3.20")

        sandbox.files.make_dir(compose_dir, user="root", request_timeout=60)
        sandbox.files.write(
            f"{compose_dir}/Dockerfile",
            f"FROM {SMOKE_IMAGE}\nRUN printf 'image-built\\n' > /image-built\n",
            user="root",
            request_timeout=60,
        )
        sandbox.files.write(
            f"{compose_dir}/compose.yaml",
            """services:
  server:
    image: harbor-nexau-smoke-server:local
    build: .
    pull_policy: never
    command: ["sh", "-c", "printf 'shared-ok\\n' > /shared/message; sleep 600"]
    volumes:
      - shared:/shared
  client:
    image: registry-1.docker.io/library/alpine:3.20
    pull_policy: never
    command: ["sleep", "600"]
    depends_on:
      - server
    volumes:
      - shared:/shared
volumes:
  shared: {}
""",
            user="root",
            request_timeout=60,
        )

        compose = f"docker compose --project-directory {shlex.quote(compose_dir)}"
        _run_checked(sandbox, f"{compose} build --pull=false", timeout=600)
        _run_checked(sandbox, f"{compose} up -d --pull never", timeout=180)
        _run_checked(
            sandbox,
            f"{compose} exec -T client sh -c "
            + shlex.quote(
                'test "$(cat /shared/message)" = shared-ok && ping -c 1 server >/dev/null'
            ),
        )
        _run_checked(sandbox, "printf 'copy-ok\\n' > /tmp/harbor-nexau-host-copy")
        _run_checked(
            sandbox,
            f"{compose} cp /tmp/harbor-nexau-host-copy client:/tmp/copied",
        )
        _run_checked(
            sandbox,
            f"{compose} exec -T client sh -c " + shlex.quote('test "$(cat /tmp/copied)" = copy-ok'),
        )
        _run_checked(sandbox, f"{compose} stop server")
        _run_checked(
            sandbox,
            f"{compose} ps --status exited --services | grep -Fx server",
        )
        _run_checked(
            sandbox,
            "container_id=$(docker create --cap-add SYS_PTRACE alpine:3.20 true) && "
            "trap 'docker rm -f \"$container_id\" >/dev/null 2>&1 || true' EXIT && "
            "docker inspect --format '{{json .HostConfig.CapAdd}}' \"$container_id\" "
            "| grep -F SYS_PTRACE",
        )
        _run_checked(sandbox, f"{compose} down -v --remove-orphans", timeout=180)

        return DindSmokeResult(
            template=template,
            sandbox_id=str(sandbox.sandbox_id),
            base_build_id=base_build_id,
            docker_version=docker_version,
            compose_version=compose_version,
            cpu_count=int(facts["cpu_count"]),
            memory_mb=int(facts["memory_mb"]),
            root_disk_mb=int(facts["root_disk_mb"]),
            available_disk_mb=int(facts["available_disk_mb"]),
        )
    finally:
        if sandbox is not None:
            try:
                sandbox.commands.run(
                    f"docker compose --project-directory {shlex.quote(compose_dir)} "
                    "down -v --remove-orphans",
                    user="root",
                    timeout=120,
                    request_timeout=150,
                )
            except Exception:
                pass
            sandbox.kill()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and verify the harbor-nexau-dind E2B template.",
    )
    parser.add_argument(
        "action",
        choices=("build", "smoke", "build-and-smoke"),
        nargs="?",
        default="build-and-smoke",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--name", default=TEMPLATE_NAME)
    parser.add_argument("--version", default=TEMPLATE_VERSION)
    parser.add_argument("--cpu-count", type=_positive_int, default=DEFAULT_CPU_COUNT)
    parser.add_argument("--memory-mb", type=_positive_int, default=DEFAULT_MEMORY_MB)
    parser.add_argument(
        "--sandbox-timeout-sec",
        type=_positive_int,
        default=DEFAULT_SANDBOX_TIMEOUT_SEC,
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--allow-base-drift", action="store_true")
    parser.add_argument(
        "--standalone-ubuntu",
        choices=(STANDALONE_UBUNTU_VARIANT,),
        help=(
            "Build Docker/Compose from a pinned Ubuntu image instead of inheriting "
            "the published DinD snapshot. This allows the current project disk quota "
            "to size the new rootfs."
        ),
    )
    parser.add_argument(
        "--trust-pinned-base",
        action="store_true",
        help=(
            "Accept the packaged expected base build without querying public-base tags. "
            "This validates the runtime contract, not the inaccessible tag identity."
        ),
    )
    parser.add_argument("--skip-cache", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    connection = load_e2b_connection(args.env_file)
    summary: dict[str, Any] = {}

    if args.action in {"build", "build-and-smoke"}:
        result = build_dind_template(
            connection,
            name=args.name,
            version=args.version,
            cpu_count=args.cpu_count,
            memory_mb=args.memory_mb,
            replace=args.replace,
            allow_base_drift=args.allow_base_drift,
            trust_pinned_base=args.trust_pinned_base,
            standalone_ubuntu=args.standalone_ubuntu,
            skip_cache=args.skip_cache,
        )
        summary["build"] = asdict(result)

    if args.action in {"smoke", "build-and-smoke"}:
        result = smoke_dind_template(
            connection,
            name=args.name,
            version=args.version,
            timeout_sec=args.sandbox_timeout_sec,
            expected_base_template_id=(
                f"ubuntu:{args.standalone_ubuntu}"
                if args.standalone_ubuntu is not None
                else BASE_TEMPLATE_ID
            ),
        )
        summary["smoke"] = asdict(result)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
