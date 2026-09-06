from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from harbor.environments.base import ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ServiceVolumeConfig
from harbor.models.trial.paths import TrialPaths
from harbor.utils.import_path import import_class

from nexau_harbor.basestream_environment import (
    BasestreamEnvironment,
    BasestreamError,
    _NorthJobClient,
    _parse_registry_reference,
)


def _environment(
    tmp_path: Path,
    *,
    config: EnvironmentConfig | None = None,
    image_user: str = "root",
) -> BasestreamEnvironment:
    pvc_root = tmp_path / "pvc"
    environment_dir = pvc_root / "dataset" / "environment"
    environment_dir.mkdir(parents=True)
    (environment_dir / "Dockerfile").write_text(
        "FROM nvidia/cuda:13.2.1-devel-ubuntu22.04\nWORKDIR /app\n",
        encoding="utf-8",
    )
    trial_paths = TrialPaths(pvc_root / "jobs" / "trial")
    trial_paths.mkdir()
    credentials = tmp_path / "northjob.json"
    credentials.write_text('{"username":"u","password":"p"}\n', encoding="utf-8")
    registry_auth = pvc_root / "info" / "registry.json"
    registry_auth.parent.mkdir(parents=True)
    registry_auth.write_text(
        json.dumps({"auths": {"harbor.local.clusters": {"auth": "dTpw"}}}),
        encoding="utf-8",
    )
    mounts = [
        ServiceVolumeConfig(
            type="bind",
            source=trial_paths.agent_dir.as_posix(),
            target="/logs/agent",
        )
    ]
    environment = BasestreamEnvironment(
        environment_dir=environment_dir,
        environment_name="terminal-bench/fp8-rmsnorm-gemm",
        session_id="fp8__abc__env",
        trial_paths=trial_paths,
        task_env_config=config
        or EnvironmentConfig(
            cpus=16,
            memory_mb=32768,
            storage_mb=1024000,
            gpus=1,
            gpu_types=["H100"],
        ),
        mounts=mounts,
        pvc_host_root=pvc_root,
        control_root=pvc_root / "control",
        credentials_path=credentials,
        registry_auth_path=registry_auth,
        registry_auth_subpath="info/registry.json",
        egress_proxy_url=("http://svc-egress.bg-agentic-coding.svc.cluster.local:7890"),
    )
    environment._image_user = image_user
    environment._image_workdir = "/app"
    environment._session_dir.mkdir(mode=0o700, parents=True)
    return environment


def test_custom_environment_import_path_is_loadable() -> None:
    environment_class = import_class(
        "nexau_harbor.basestream_environment:BasestreamEnvironment",
        label="environment",
    )

    assert environment_class is BasestreamEnvironment


def test_provider_declares_gpu_mounted_and_resource_capabilities(tmp_path: Path) -> None:
    environment = _environment(tmp_path)

    assert environment.capabilities.gpus
    assert environment.capabilities.mounted
    assert environment.capabilities.disable_internet
    assert not environment.capabilities.docker_compose
    assert environment.resource_capabilities().model_dump() == {
        "cpu_limit": True,
        "cpu_request": True,
        "memory_limit": True,
        "memory_request": True,
    }


def test_runtime_manifest_preserves_entrypoint_and_requests_exact_jax_resources(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    image = "harbor.local.clusters/project/jax@sha256:" + "a" * 64

    manifest = environment._runtime_manifest(alias="jax-env", image_ref=image)

    pod_spec = manifest["spec"]["tasks"][0]["template"]["spec"]
    container = pod_spec["containers"][0]
    assert "command" not in container
    assert container["args"][:3] == ["/bin/bash", "--norc", "-lc"]
    assert container["image"] == image
    assert container["resources"]["requests"] == {
        "cpu": "16",
        "memory": "32768Mi",
        "ephemeral-storage": "1024000Mi",
        "nvidia.com/gpu": "1",
    }
    assert container["resources"]["limits"] == container["resources"]["requests"]
    assert pod_spec["automountServiceAccountToken"] is False
    expression = pod_spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"][0]
    assert expression["key"] == "kubernetes.io/hostname"
    assert expression["operator"] == "NotIn"
    assert expression["values"] == [f"node-b300-{index:03d}" for index in range(1, 9)]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["HTTPS_PROXY"] == ("http://svc-egress.bg-agentic-coding.svc.cluster.local:7890")
    assert "NVIDIA_VISIBLE_DEVICES" not in env


def test_runtime_manifest_rejects_non_h100(tmp_path: Path) -> None:
    environment = _environment(
        tmp_path,
        config=EnvironmentConfig(
            cpus=4,
            memory_mb=16384,
            storage_mb=30720,
            gpus=1,
            gpu_types=["B300"],
        ),
    )

    with pytest.raises(BasestreamError, match="only H100"):
        environment._runtime_manifest(alias="bad-gpu", image_ref="registry/x@sha256:" + "a" * 64)


def test_runtime_mount_rejects_source_outside_pvc_before_creating_it(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    outside = tmp_path / "outside" / "logs"
    environment._mounts.append(
        ServiceVolumeConfig(type="bind", source=str(outside), target="/logs/outside")
    )

    with pytest.raises(BasestreamError, match="outside the trial PVC"):
        environment._runtime_mounts()

    assert not outside.exists()


def test_build_manifest_is_cpu_only_and_mounts_exact_context(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    context = environment.settings.control_root / "images" / "hash" / "context"
    output = environment.settings.control_root / "images" / "hash" / "output"
    context.mkdir(parents=True)
    output.mkdir(parents=True)

    manifest = environment._build_job_manifest(
        alias="build-test",
        context_dir=context,
        output_dir=output,
        destination="harbor.local.clusters/project/task:hash",
        timeout_sec=600,
    )

    pod_spec = manifest["spec"]["tasks"][0]["template"]["spec"]
    container = pod_spec["containers"][0]
    assert container["command"] == ["/kaniko/executor"]
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert container["resources"]["requests"]["ephemeral-storage"] == "100Gi"
    mounts = {item["mountPath"]: item for item in container["volumeMounts"]}
    assert mounts["/workspace"]["subPath"].endswith("/context")
    assert mounts["/workspace"]["readOnly"] is True
    assert mounts["/kaniko/.docker/config.json"]["readOnly"] is True
    assert pod_spec["automountServiceAccountToken"] is False
    build_env = {item["name"]: item["value"] for item in container["env"]}
    assert build_env["HTTPS_PROXY"] == (
        "http://svc-egress.bg-agentic-coding.svc.cluster.local:7890"
    )
    assert "harbor.local.clusters" in build_env["NO_PROXY"].split(",")


def test_image_metadata_verifies_digest_entrypoint_user_and_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    config = {
        "config": {
            "Entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"],
            "User": "agent",
            "WorkingDir": "/app",
        }
    }
    config_bytes = json.dumps(config, separators=(",", ":")).encode()
    config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
    manifest_bytes = json.dumps(
        {"schemaVersion": 2, "config": {"digest": config_digest}},
        separators=(",", ":"),
    ).encode()
    digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()

    monkeypatch.setattr(environment, "_registry_auth", lambda registry: "auth")

    def request(url: str, *, auth: str, accept: str | None = None):
        del auth, accept
        if "/manifests/" in url:
            return manifest_bytes, {"Docker-Content-Digest": digest}
        return config_bytes, {}

    monkeypatch.setattr(environment, "_registry_request", request)

    metadata = environment._image_metadata(f"harbor.local.clusters/project/task@{digest}")

    assert metadata["entrypoint"] == ["/opt/nvidia/nvidia_entrypoint.sh"]
    assert metadata["user"] == "agent"
    assert metadata["workdir"] == "/app"
    assert metadata["config_digest"] == config_digest


def test_internal_registry_request_bypasses_process_proxy_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    captured_handlers: list[tuple[object, ...]] = []
    captured_request: list[urllib.request.Request] = []

    class Response:
        headers = {"Docker-Content-Digest": "sha256:" + "a" * 64}

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return b"internal"

    class Opener:
        @staticmethod
        def open(request: urllib.request.Request, timeout: int):
            assert timeout == 30
            captured_request.append(request)
            return Response()

    def build_opener(*handlers: object):
        captured_handlers.append(handlers)
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)

    body, _ = environment._registry_request(
        "https://harbor.local.clusters/v2/project/task/manifests/digest",
        auth="encoded-credentials",
    )

    assert body == b"internal"
    proxy_handler = captured_handlers[0][0]
    assert isinstance(proxy_handler, urllib.request.ProxyHandler)
    assert proxy_handler.proxies == {}
    assert captured_request[0].get_header("Authorization") == "Basic encoded-credentials"


def test_parse_official_docker_hub_reference_removes_tag_from_api_path() -> None:
    digest = "sha256:" + "a" * 64

    reference = _parse_registry_reference(
        f"harborframework/terminal-bench:fp8-environment@{digest}"
    )

    assert reference.registry == "docker.io"
    assert reference.api_registry == "registry-1.docker.io"
    assert reference.repository == "harborframework/terminal-bench"
    assert reference.digest == digest


def test_image_metadata_verifies_official_docker_hub_image_without_private_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    config = {
        "config": {
            "Entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"],
            "User": "agent",
            "WorkingDir": "/app",
        }
    }
    config_bytes = json.dumps(config, separators=(",", ":")).encode()
    config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
    manifest_bytes = json.dumps(
        {"schemaVersion": 2, "config": {"digest": config_digest}},
        separators=(",", ":"),
    ).encode()
    digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    requested_paths: list[str] = []

    def request(reference, path: str, *, accept: str | None = None):
        del accept
        assert reference.repository == "harborframework/terminal-bench"
        requested_paths.append(path)
        if path.startswith("manifests/"):
            return manifest_bytes, {"Docker-Content-Digest": digest}
        return config_bytes, {}

    monkeypatch.setattr(environment, "_docker_hub_request", request)
    monkeypatch.setattr(
        environment,
        "_registry_auth",
        lambda registry: pytest.fail(f"private auth used for public registry {registry}"),
    )

    metadata = environment._image_metadata(
        f"harborframework/terminal-bench:fp8-environment@{digest}"
    )

    assert requested_paths == [f"manifests/{digest}", f"blobs/{config_digest}"]
    assert metadata["user"] == "agent"
    assert metadata["entrypoint"] == ["/opt/nvidia/nvidia_entrypoint.sh"]


def test_docker_hub_auth_challenge_is_restricted_to_official_token_service() -> None:
    parameters = BasestreamEnvironment._docker_hub_auth_parameters(
        'Bearer realm="https://auth.docker.io/token",service="registry.docker.io",'
        'scope="repository:harborframework/terminal-bench:pull"'
    )
    assert parameters["service"] == "registry.docker.io"

    with pytest.raises(BasestreamError, match="unsafe token realm"):
        BasestreamEnvironment._docker_hub_auth_parameters(
            'Bearer realm="https://attacker.example/token",service="registry.docker.io"'
        )


def test_public_registry_requests_use_managed_egress_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    captured_handlers: list[tuple[object, ...]] = []

    class Response:
        headers = {"Docker-Content-Digest": "sha256:" + "a" * 64}

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return b"body"

    class Opener:
        @staticmethod
        def open(request: urllib.request.Request, timeout: int):
            del request, timeout
            return Response()

    def build_opener(*handlers: object):
        captured_handlers.append(handlers)
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)

    body, _ = environment._verified_https_request(
        urllib.request.Request("https://registry-1.docker.io/v2/")
    )

    assert body == b"body"
    proxy_handler = captured_handlers[0][0]
    assert isinstance(proxy_handler, urllib.request.ProxyHandler)
    assert proxy_handler.proxies == {
        "http": "http://svc-egress.bg-agentic-coding.svc.cluster.local:7890",
        "https": "http://svc-egress.bg-agentic-coding.svc.cluster.local:7890",
    }


def test_configured_prebuilt_image_writes_verified_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "sha256:" + "a" * 64
    image = f"harbor.local.clusters/project/task@{digest}"
    environment = _environment(
        tmp_path,
        config=EnvironmentConfig(
            cpus=4,
            memory_mb=16384,
            storage_mb=30720,
            gpus=1,
            gpu_types=["H100"],
            docker_image=image,
        ),
    )
    metadata = {"image": image, "entrypoint": [], "user": "root", "workdir": "/app"}
    monkeypatch.setattr(environment, "_image_metadata", lambda value: metadata)

    actual_image, actual_metadata = environment._ensure_image_sync(force_build=False)

    assert actual_image == image
    assert actual_metadata == metadata
    receipt_path = (
        environment.settings.control_root / "images" / environment.environment_id / "receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["source"] == "prebuilt"
    assert receipt["digest"] == digest
    assert receipt["success"] is True


def test_docker_hub_prebuilt_image_is_mirrored_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "sha256:" + "a" * 64
    source = f"harborframework/terminal-bench:task@{digest}"
    internal = "harbor.local.clusters/project/task@sha256:" + "b" * 64
    environment = _environment(
        tmp_path,
        config=EnvironmentConfig(
            cpus=4,
            memory_mb=16384,
            storage_mb=30720,
            gpus=1,
            gpu_types=["H100"],
            docker_image=source,
        ),
    )
    source_metadata = {
        "image": source,
        "entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"],
        "user": "agent",
        "workdir": "/app",
    }
    internal_metadata = {**source_metadata, "image": internal}
    monkeypatch.setattr(environment, "_image_metadata", lambda value: source_metadata)
    captured: list[tuple[Path, str, dict[str, object]]] = []

    def mirror(image_dir: Path, source_ref: str, metadata: dict[str, object]):
        captured.append((image_dir, source_ref, metadata))
        return internal, internal_metadata

    monkeypatch.setattr(environment, "_mirror_prebuilt_image", mirror)

    actual_image, actual_metadata = environment._ensure_image_sync(force_build=False)

    assert actual_image == internal
    assert actual_metadata == internal_metadata
    assert captured[0][1:] == (source, source_metadata)


def test_docker_hub_mirror_receipt_reuses_internal_image_without_source_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "sha256:" + "a" * 64
    source = f"harborframework/terminal-bench:task@{digest}"
    internal = f"harbor.local.clusters/project/task@{digest}"
    environment = _environment(
        tmp_path,
        config=EnvironmentConfig(
            cpus=4,
            memory_mb=16384,
            storage_mb=30720,
            gpus=1,
            gpu_types=["H100"],
            docker_image=source,
        ),
    )
    receipt_path = (
        environment.settings.control_root
        / "images"
        / environment.environment_id
        / "receipt.json"
    )
    receipt_path.parent.mkdir(mode=0o700, parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "harbor-basestream-image-v1",
                "environment_id": environment.environment_id,
                "immutable_ref": internal,
                "source": "prebuilt-mirror",
                "source_ref": source,
                "success": True,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "image": internal,
        "entrypoint": [],
        "user": "root",
        "workdir": "/app",
    }

    def internal_metadata_only(image_ref: str) -> dict[str, object]:
        assert image_ref == internal
        return metadata

    monkeypatch.setattr(environment, "_image_metadata", internal_metadata_only)

    actual_image, actual_metadata = environment._ensure_image_sync(force_build=False)

    assert actual_image == internal
    assert actual_metadata == metadata


def test_prebuilt_mirror_uses_nexmltask_skopeo_preserve_digest_contract(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    digest = "sha256:" + "a" * 64
    source = f"harborframework/terminal-bench:task@{digest}"
    destination = "harbor.local.clusters/project/task:mirror"
    staging = environment.settings.control_root / "images" / "hash" / "mirror-staging"
    staging.mkdir(parents=True)

    manifest = environment._mirror_job_manifest(
        alias="mirror-test",
        source_ref=source,
        destination=destination,
        staging_dir=staging,
        timeout_sec=600,
    )

    pod_spec = manifest["spec"]["tasks"][0]["template"]["spec"]
    container = pod_spec["containers"][0]
    assert container["image"].endswith(
        "@sha256:47853bb9fb24202af9110531ebd6e43c5f97701254ca290596640290d17942f4"
    )
    assert container["command"] == ["/bin/sh", "-ceu"]
    script = container["args"][0]
    syntax = subprocess.run(
        ["/bin/sh", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    assert '"dir:$image_dir"' in script
    assert "--dest-precompute-digests" in script
    assert "--retry-times=10" in script
    assert '--image-parallel-copies="$parallel_copies"' in script
    assert container["args"][1:] == [
        "--",
        f"docker://docker.io/harborframework/terminal-bench@{digest}",
        f"docker://{destination}",
        digest,
        "1",
    ]
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    mounts = {item["mountPath"]: item for item in container["volumeMounts"]}
    assert mounts["/auth/config.json"]["readOnly"] is True
    assert mounts["/mirror"]["subPath"].endswith("/mirror-staging")
    assert pod_spec["automountServiceAccountToken"] is False


def test_prebuilt_skopeo_mirror_preserves_source_digest_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    digest = "sha256:" + "a" * 64
    source = f"harborframework/terminal-bench:task@{digest}"
    internal = (
        "harbor.local.clusters/zhangguoqiang/harbor-tb4/terminal-bench-fp8-rmsnorm-gemm@" + digest
    )
    source_metadata = {
        "config_digest": "sha256:" + "c" * 64,
        "image": source,
        "entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"],
        "user": "agent",
        "workdir": "/app",
    }
    internal_metadata = {**source_metadata, "image": internal}
    submitted: list[dict[str, object]] = []
    deleted: list[str] = []
    monkeypatch.setattr(
        _NorthJobClient,
        "submit",
        lambda self, manifest: submitted.append(manifest),
    )
    monkeypatch.setattr(
        _NorthJobClient,
        "delete",
        lambda self, job_name: deleted.append(job_name),
    )
    monkeypatch.setattr(
        environment,
        "_wait_for_job",
        lambda alias, **kwargs: (
            {
                "metadata": {"name": "job-mirror"},
                "status": {"state": {"phase": "Completed"}},
            },
            {"metadata": {"name": "pod-mirror"}},
        ),
    )
    monkeypatch.setattr(environment, "_job_logs", lambda pod, container: "copied\n")
    monkeypatch.setattr(environment, "_image_metadata", lambda image: internal_metadata)
    image_dir = environment.settings.control_root / "images" / environment.environment_id
    image_dir.mkdir(parents=True)

    actual_image, _ = environment._mirror_prebuilt_image(
        image_dir,
        source,
        source_metadata,
    )

    assert actual_image == internal
    assert len(submitted) == 1
    assert deleted == ["job-mirror"]
    receipt = json.loads((image_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["source"] == "prebuilt-mirror"
    assert receipt["mirror_mode"] == "staged-dir-v1"
    assert receipt["source_ref"] == source
    assert receipt["copy_source_ref"] == f"docker.io/harborframework/terminal-bench@{digest}"
    assert receipt["digest"] == digest
    assert receipt["immutable_ref"] == internal
    assert receipt["metadata"] == internal_metadata
    assert not (image_dir / "mirror-staging").exists()


def test_image_metadata_rejects_application_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    config_bytes = b'{"config":{"Entrypoint":["/app/server"]}}'
    config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
    manifest_bytes = json.dumps({"config": {"digest": config_digest}}).encode()
    digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    monkeypatch.setattr(environment, "_registry_auth", lambda registry: "auth")
    monkeypatch.setattr(
        environment,
        "_registry_request",
        lambda url, **kwargs: (
            (manifest_bytes, {"Docker-Content-Digest": digest})
            if "/manifests/" in url
            else (config_bytes, {})
        ),
    )

    with pytest.raises(BasestreamError, match="cannot safely receive"):
        environment._image_metadata(f"harbor.local.clusters/project/task@{digest}")


@pytest.mark.asyncio
async def test_exec_defaults_to_image_user_and_forces_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path, image_user="agent")
    environment._pod_name = "pod-test"
    captured: list[str] = []

    def run(args: list[str], **kwargs):
        del kwargs
        captured.extend(args)
        return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(environment, "_run_text", run)

    result = await environment.exec("pwd", timeout_sec=10)

    assert result == ExecResult(stdout="ok\n", stderr="", return_code=0)
    rendered = " ".join(captured)
    assert "su agent" in rendered
    assert "cd /app" in rendered
    assert "HTTPS_PROXY" in rendered


@pytest.mark.asyncio
async def test_runtime_validation_rejects_wrong_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    environment._pod_name = "pod-test"
    monkeypatch.setattr(
        environment,
        "exec",
        AsyncMock(
            side_effect=[
                ExecResult(stdout="", stderr="", return_code=0),
                ExecResult(stdout="NVIDIA B300 SXM6 AC\n", stderr="", return_code=0),
            ]
        ),
    )

    with pytest.raises(BasestreamError, match="non-H100"):
        await environment._validate_runtime()


@pytest.mark.asyncio
async def test_runtime_validation_accepts_exact_h100_and_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    environment._pod_name = "pod-test"
    exec_mock = AsyncMock(
        side_effect=[
            ExecResult(stdout="", stderr="", return_code=0),
            ExecResult(stdout="NVIDIA H100 80GB HBM3\n", stderr="", return_code=0),
            ExecResult(
                stdout="overlay 1300000 1000 1299000 1% /\n",
                stderr="",
                return_code=0,
            ),
        ]
    )
    monkeypatch.setattr(environment, "exec", exec_mock)

    validation = await environment._validate_runtime()

    assert exec_mock.await_count == 3
    assert validation == {
        "service_account_token_absent": True,
        "gpu_names": ["NVIDIA H100 80GB HBM3"],
        "available_storage_mb": 1299000,
        "effective_image_user": "root",
    }


@pytest.mark.asyncio
async def test_start_submits_runtime_job_and_stop_deletes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    image = "harbor.local.clusters/project/task@sha256:" + "a" * 64
    monkeypatch.setattr(
        environment,
        "_ensure_image",
        AsyncMock(
            return_value=(
                image,
                {
                    "entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"],
                    "user": "agent",
                    "workdir": "/app",
                },
            )
        ),
    )
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(
        _NorthJobClient,
        "submit",
        lambda self, manifest: submitted.append(manifest),
    )
    monkeypatch.setattr(
        environment,
        "_wait_for_job",
        lambda alias, **kwargs: (
            {"metadata": {"name": "job-runtime"}, "status": {"state": {"phase": "Running"}}},
            {
                "metadata": {"name": "pod-runtime"},
                "spec": {"nodeName": "node-h100-001"},
                "status": {"phase": "Running"},
            },
        ),
    )
    validation = {
        "service_account_token_absent": True,
        "gpu_names": ["NVIDIA H100 80GB HBM3"],
        "available_storage_mb": 40000,
        "effective_image_user": "agent",
    }
    monkeypatch.setattr(
        environment,
        "_validate_runtime",
        AsyncMock(return_value=validation),
    )
    deleted: list[str] = []
    monkeypatch.setattr(environment, "_delete_job", lambda name: deleted.append(name))

    await environment.start(force_build=False)

    assert len(submitted) == 1
    container = submitted[0]["spec"]["tasks"][0]["template"]["spec"]["containers"][0]
    assert "command" not in container
    assert environment._runtime_job_name == "job-runtime"
    assert environment._pod_name == "pod-runtime"
    assert environment._image_user == "agent"
    runtime_receipt = json.loads(
        (environment._session_dir / "runtime-receipt.json").read_text(encoding="utf-8")
    )
    assert runtime_receipt["success"] is True
    assert runtime_receipt["job_name"] == "job-runtime"
    assert runtime_receipt["node_name"] == "node-h100-001"
    assert runtime_receipt["validation"] == validation

    await environment.stop(delete=True)

    assert deleted == ["job-runtime"]
    assert environment._runtime_job_name is None
    assert environment._pod_name is None
    stop_receipt = json.loads(
        (environment._session_dir / "stop-receipt.json").read_text(encoding="utf-8")
    )
    assert stop_receipt["delete"] is True


@pytest.mark.asyncio
async def test_start_cleans_up_job_when_runtime_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    image = "harbor.local.clusters/project/task@sha256:" + "a" * 64
    monkeypatch.setattr(
        environment,
        "_ensure_image",
        AsyncMock(return_value=(image, {"entrypoint": [], "user": "root", "workdir": None})),
    )
    monkeypatch.setattr(_NorthJobClient, "submit", lambda self, manifest: None)
    monkeypatch.setattr(
        environment,
        "_wait_for_job",
        lambda alias, **kwargs: (
            {"metadata": {"name": "job-invalid"}},
            {"metadata": {"name": "pod-invalid"}, "status": {"phase": "Running"}},
        ),
    )
    monkeypatch.setattr(
        environment,
        "_validate_runtime",
        AsyncMock(side_effect=BasestreamError("wrong GPU")),
    )
    deleted: list[str] = []
    monkeypatch.setattr(environment, "_delete_job", lambda name: deleted.append(name))

    with pytest.raises(BasestreamError, match="wrong GPU"):
        await environment.start(force_build=False)

    assert deleted == ["job-invalid"]
    assert environment._runtime_job_name is None
    assert environment._pod_name is None
    receipt = json.loads(
        (environment._session_dir / "runtime-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["success"] is False
    assert receipt["error_type"] == "BasestreamError"
    assert receipt["job_name"] == "job-invalid"
    assert receipt["pod_name"] == "pod-invalid"
    assert receipt["cleanup_error"] is None


@pytest.mark.asyncio
async def test_start_preserves_job_handle_when_failure_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    image = "harbor.local.clusters/project/task@sha256:" + "a" * 64
    monkeypatch.setattr(
        environment,
        "_ensure_image",
        AsyncMock(return_value=(image, {"entrypoint": [], "user": "root", "workdir": None})),
    )
    monkeypatch.setattr(_NorthJobClient, "submit", lambda self, manifest: None)
    monkeypatch.setattr(
        environment,
        "_wait_for_job",
        lambda alias, **kwargs: (
            {"metadata": {"name": "job-retry-cleanup"}},
            {"metadata": {"name": "pod-retry-cleanup"}, "status": {"phase": "Running"}},
        ),
    )
    monkeypatch.setattr(
        environment,
        "_validate_runtime",
        AsyncMock(side_effect=BasestreamError("wrong GPU")),
    )

    def fail_delete(name: str) -> None:
        del name
        raise BasestreamError("delete denied")

    monkeypatch.setattr(environment, "_delete_job", fail_delete)

    with pytest.raises(BasestreamError, match="wrong GPU"):
        await environment.start(force_build=False)

    assert environment._runtime_job_name == "job-retry-cleanup"
    assert environment._pod_name == "pod-retry-cleanup"
    receipt = json.loads(
        (environment._session_dir / "runtime-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["cleanup_error"] == "BasestreamError: delete denied"


def test_upload_tar_rejects_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    (source / "link").symlink_to(outside)

    with pytest.raises(BasestreamError, match="symlink"):
        BasestreamEnvironment._write_safe_tar(source, tmp_path / "upload.tar")


def test_download_tar_rejects_link_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "download.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("escape")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        archive.addfile(member, io.BytesIO())

    with pytest.raises(BasestreamError, match="unsafe artifact"):
        BasestreamEnvironment._extract_safe_tar(archive_path, tmp_path / "target")
