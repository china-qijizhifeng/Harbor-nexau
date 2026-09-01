from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from nexau_harbor import e2b_dind_template
from nexau_harbor.e2b_dind_template import (
    BASE_TEMPLATE_BUILD_ID,
    BASE_TEMPLATE_ID,
    BUILDX_VERSION,
    COMPOSE_VERSION,
    CONTAINERD_VERSION,
    DOCKER_VERSION,
    PUBLISHED_DIND_TEMPLATES,
    REGISTRY_MIRRORS,
    STANDALONE_SOURCE_BUILD,
    STANDALONE_SOURCE_ID,
    STANDALONE_UBUNTU_VARIANT,
    E2BConnection,
)


def _tag(tag: str, build_id: str) -> SimpleNamespace:
    return SimpleNamespace(tag=tag, build_id=build_id)


def test_load_e2b_connection_reads_a_complete_self_hosted_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "e2b.env"
    env_file.write_text(
        "E2B_API_KEY=private-key\n"
        "E2B_API_URL=https://api.e2b-hk.example\n"
        "E2B_DOMAIN=e2b-hk.example\n"
        "HARBOR_NEXAU_NO_PROXY_HOSTS=build.e2b-hk.example\n"
    )
    for name in (
        "PYTHON_DOTENV_DISABLED",
        "E2B_API_KEY",
        "E2B_API_URL",
        "E2B_DOMAIN",
        "HARBOR_NEXAU_NO_PROXY_HOSTS",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)

    connection = e2b_dind_template.load_e2b_connection(env_file)

    assert connection == E2BConnection(
        api_key="private-key",
        api_url="https://api.e2b-hk.example",
        domain="e2b-hk.example",
    )
    assert e2b_dind_template.os.environ["NO_PROXY"].split(",") == [
        "build.e2b-hk.example",
        "api.e2b-hk.example",
        "e2b-hk.example",
        ".e2b-hk.example",
    ]
    assert e2b_dind_template.os.environ["no_proxy"] == e2b_dind_template.os.environ["NO_PROXY"]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, "E2B_API_KEY is required"),
        (
            {"E2B_API_KEY": "key", "E2B_API_URL": "https://api.example"},
            "E2B_API_URL and E2B_DOMAIN must either both be set or both be empty",
        ),
    ],
)
def test_load_e2b_connection_rejects_incomplete_settings(
    values: dict[str, str],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "E2B_API_KEY",
        "E2B_API_URL",
        "E2B_DOMAIN",
        "HARBOR_NEXAU_NO_PROXY_HOSTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        e2b_dind_template.load_e2b_connection(tmp_path / "missing.env")


def test_validate_base_template_rejects_unreviewed_default_tag_drift() -> None:
    template_api = MagicMock()
    template_api.get_tags.return_value = [_tag("default", "unexpected-build")]
    connection = E2BConnection(api_key="test-key")

    with pytest.raises(RuntimeError, match="pinned DinD base template changed"):
        e2b_dind_template.validate_base_template(template_api, connection)

    assert (
        e2b_dind_template.validate_base_template(
            template_api,
            connection,
            allow_base_drift=True,
        )
        == "unexpected-build"
    )


def test_validate_base_template_can_explicitly_trust_packaged_pin() -> None:
    template_api = MagicMock()

    assert (
        e2b_dind_template.validate_base_template(
            template_api,
            E2BConnection(api_key="test-key"),
            trust_pinned_base=True,
        )
        == BASE_TEMPLATE_BUILD_ID
    )
    template_api.get_tags.assert_not_called()


def test_packaged_manifests_define_both_published_capacity_profiles() -> None:
    assert PUBLISHED_DIND_TEMPLATES["e2b-nex/harbor-nexau-dind:v1"].cpu_count == 8
    high_cpu = PUBLISHED_DIND_TEMPLATES["s-project-e89a/harbor-nexau-dind-16c:v1"]
    assert high_cpu.cpu_count == 16
    assert high_cpu.memory_mb == 16_384
    assert high_cpu.storage_mb == 55_342


def test_parser_requires_explicit_opt_in_to_trust_pinned_base() -> None:
    assert not e2b_dind_template._parser().parse_args(["build"]).trust_pinned_base
    assert (
        e2b_dind_template._parser().parse_args(["build", "--trust-pinned-base"]).trust_pinned_base
    )

    standalone = e2b_dind_template._parser().parse_args(
        ["build", "--standalone-ubuntu", STANDALONE_UBUNTU_VARIANT]
    )
    assert standalone.standalone_ubuntu == STANDALONE_UBUNTU_VARIANT


def test_template_builder_derives_from_pinned_base_and_records_provenance() -> None:
    template_api = MagicMock()
    builder = MagicMock()
    template_api.return_value.from_template.return_value = builder
    builder.copy.return_value = builder
    builder.run_cmd.return_value = builder

    result = e2b_dind_template._template_builder(
        template_api,
        BASE_TEMPLATE_BUILD_ID,
        "custom-dind",
        "v-test",
    )

    assert result is builder
    template_api.assert_called_once_with(file_context_path=e2b_dind_template.TEMPLATE_CONTEXT)
    template_api.return_value.from_template.assert_called_once_with(BASE_TEMPLATE_ID)
    assert builder.copy.call_args_list == [
        call("daemon.json", "/etc/docker/daemon.json", user="root", mode=0o644),
        call(
            "harbor-nexau-dind-ready",
            "/usr/local/bin/harbor-nexau-dind-ready",
            user="root",
            mode=0o755,
        ),
    ]
    provenance_command = builder.run_cmd.call_args_list[0].args[0]
    assert '"name":"custom-dind"' in provenance_command
    assert '"version":"v-test"' in provenance_command
    verification_command = builder.run_cmd.call_args_list[1].args[0]
    assert DOCKER_VERSION in verification_command
    assert COMPOSE_VERSION in verification_command


def test_template_builder_can_install_standalone_docker_profile() -> None:
    template_api = MagicMock()
    builder = MagicMock()
    template_api.return_value.from_ubuntu_image.return_value = builder
    builder.run_cmd.return_value = builder
    builder.copy.return_value = builder

    result = e2b_dind_template._template_builder(
        template_api,
        STANDALONE_SOURCE_BUILD,
        "standalone-dind",
        "v-test",
        base_template_id=STANDALONE_SOURCE_ID,
        standalone_ubuntu=STANDALONE_UBUNTU_VARIANT,
    )

    assert result is builder
    template_api.assert_called_once_with(file_context_path=e2b_dind_template.TEMPLATE_CONTEXT)
    template_api.return_value.from_ubuntu_image.assert_called_once_with(STANDALONE_UBUNTU_VARIANT)
    install_command = builder.run_cmd.call_args_list[0].args[0]
    assert f"docker-ce='5:{DOCKER_VERSION}-" in install_command
    assert f"containerd.io='{CONTAINERD_VERSION}-" in install_command
    assert f"docker-buildx-plugin='{BUILDX_VERSION}-" in install_command
    assert f"docker-compose-plugin='{COMPOSE_VERSION}-" in install_command
    provenance_command = builder.run_cmd.call_args_list[1].args[0]
    assert f'"base_template_id":"{STANDALONE_SOURCE_ID}"' in provenance_command
    assert f'"base_build_id":"{STANDALONE_SOURCE_BUILD}"' in provenance_command


def test_build_dind_template_publishes_default_and_version_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_id = "new-build-id"
    template_id = "new-template-id"

    class FakeTemplate:
        build_kwargs: dict[str, object] = {}

        def __init__(self, file_context_path: Path | None = None) -> None:
            self.file_context_path = file_context_path

        def from_template(self, template: str) -> FakeTemplate:
            assert template == BASE_TEMPLATE_ID
            return self

        def copy(self, *args: object, **kwargs: object) -> FakeTemplate:
            return self

        def run_cmd(self, *args: object, **kwargs: object) -> FakeTemplate:
            return self

        @staticmethod
        def get_tags(template: str, **kwargs: str) -> list[SimpleNamespace]:
            assert kwargs == {"api_key": "test-key"}
            if template == BASE_TEMPLATE_ID:
                return [_tag("default", BASE_TEMPLATE_BUILD_ID)]
            assert template == template_id
            candidate_tag = FakeTemplate.build_kwargs["tags"][0]
            return [
                _tag("default", build_id),
                _tag("v-test", build_id),
                _tag(candidate_tag, build_id),
            ]

        @staticmethod
        def alias_exists(name: str, **kwargs: str) -> bool:
            assert name == "custom-dind"
            assert kwargs == {"api_key": "test-key"}
            return False

        @staticmethod
        def assign_tags(target: str, tags: list[str], **kwargs: str) -> SimpleNamespace:
            candidate_tag = FakeTemplate.build_kwargs["tags"][0]
            assert target == f"custom-dind:{candidate_tag}"
            assert tags == ["v-test", "default"]
            assert kwargs == {"api_key": "test-key"}
            return SimpleNamespace(build_id=build_id, tags=tags)

        @staticmethod
        def remove_tags(name: str, tags: str, **kwargs: str) -> None:
            assert name == "custom-dind"
            assert tags.startswith("candidate-")
            assert kwargs == {"api_key": "test-key"}

        @classmethod
        def build(cls, builder: object, name: str, **kwargs: object) -> SimpleNamespace:
            assert isinstance(builder, cls)
            assert name == "custom-dind"
            cls.build_kwargs = kwargs
            return SimpleNamespace(template_id=template_id, build_id=build_id)

    monkeypatch.setattr(
        e2b_dind_template,
        "_load_e2b_sdk",
        lambda: (FakeTemplate, MagicMock()),
    )

    result = e2b_dind_template.build_dind_template(
        E2BConnection(api_key="test-key"),
        name="custom-dind",
        version="v-test",
        cpu_count=8,
        memory_mb=16_384,
    )

    assert result.template_id == template_id
    assert result.build_id == build_id
    assert result.tags == ("default", "v-test")
    assert len(FakeTemplate.build_kwargs["tags"]) == 1
    assert FakeTemplate.build_kwargs["tags"][0].startswith("candidate-")
    assert FakeTemplate.build_kwargs["cpu_count"] == 8
    assert FakeTemplate.build_kwargs["memory_mb"] == 16_384
    assert FakeTemplate.build_kwargs["api_key"] == "test-key"


def test_build_dind_template_requires_explicit_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_api = MagicMock()
    template_api.get_tags.return_value = [_tag("default", BASE_TEMPLATE_BUILD_ID)]
    template_api.alias_exists.return_value = True
    monkeypatch.setattr(
        e2b_dind_template,
        "_load_e2b_sdk",
        lambda: (template_api, MagicMock()),
    )

    with pytest.raises(RuntimeError, match="pass --replace"):
        e2b_dind_template.build_dind_template(E2BConnection(api_key="test-key"))

    template_api.build.assert_not_called()


def test_daemon_configuration_uses_verified_registry_mirror() -> None:
    config = json.loads((e2b_dind_template.TEMPLATE_CONTEXT / "daemon.json").read_text())

    assert config["registry-mirrors"] == list(REGISTRY_MIRRORS)
