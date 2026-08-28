from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from harbor.agents.factory import AgentFactory
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from nexau_harbor.agent import NexAUAgent
from nexau_harbor.constants import (
    NEXAU_DETACHED_EXEC_ENV,
    NEXAU_GIT_COMMIT,
    REMOTE_WORKSPACE,
)
from nexau_harbor.system_dependencies import PREPARE_DEBIAN_CA_CERTIFICATES_JAVA


class FakeEnvironment:
    def __init__(
        self,
        *,
        default_user: str | int | None = "agent",
        identity_stdout: str = "10001\n10001\n",
    ) -> None:
        self.default_user = default_user
        self.identity_stdout = identity_stdout
        self.exec_calls: list[dict[str, Any]] = []
        self.uploaded_dirs: list[tuple[Path | str, str]] = []
        self.uploaded_files: list[tuple[Path | str, str]] = []

    async def exec(self, **kwargs: Any) -> ExecResult:
        self.exec_calls.append(kwargs)
        if "id -u && id -g" in kwargs["command"]:
            return ExecResult(stdout=self.identity_stdout, stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        self.uploaded_dirs.append((source_dir, target_dir))

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploaded_files.append((source_path, target_path))


class DetachedFakeEnvironment(FakeEnvironment):
    SUPPORTS_DETACHED_AGENT_EXEC = True


def make_agent(tmp_path: Path, **kwargs: Any) -> NexAUAgent:
    extra_env = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://example.test/v1",
    }
    extra_env.update(kwargs.pop("extra_env", {}))
    return NexAUAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5",
        extra_env=extra_env,
        **kwargs,
    )


def test_agent_uses_packaged_config_and_harbor_model_connection(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    runtime_env = agent._runtime_env()

    assert agent.name() == "nexau"
    assert agent._config_path.is_file()
    assert runtime_env["LLM_MODEL"] == "gpt-5"
    assert runtime_env["LLM_KIND"] == "openai_responses"
    assert runtime_env["LLM_API_KEY"] == "test-key"
    assert runtime_env["LLM_BASE_URL"] == "https://example.test/v1"
    assert runtime_env["NEXAU_TRACE_CHECKPOINT_SEC"] == "30"
    assert runtime_env["NEXAU_LLM_RETRY_BACKOFF_MAX_SEC"] == "30"
    assert runtime_env["NEXAU_LLM_RETRY_BACKOFF_JITTER_RATIO"] == "0.25"
    assert runtime_env["GDPVAL_NEXAU_SUMMARY_MODEL"] == "gpt-5"


def test_agent_allows_explicit_api_type(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, api_type="openai_chat_completion")
    assert agent._runtime_env()["LLM_KIND"] == "openai_chat_completion"


def test_trace_checkpoint_interval_can_be_overridden(tmp_path: Path) -> None:
    agent = make_agent(
        tmp_path,
        extra_env={"NEXAU_TRACE_CHECKPOINT_SEC": "5"},
    )
    assert agent._runtime_env()["NEXAU_TRACE_CHECKPOINT_SEC"] == "5"


@pytest.mark.parametrize(
    "proxy_url",
    [
        "http://127.0.0.1:7890",
        "localhost:7890",
        "http://worker.localhost:7890",
        "socks5://[::1]:7890",
        "http://0.0.0.0:7890",
    ],
)
def test_agent_does_not_forward_process_local_proxies(
    tmp_path: Path, proxy_url: str
) -> None:
    agent = make_agent(
        tmp_path,
        extra_env={
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
            "NO_PROXY": "127.0.0.1,localhost,::1",
        },
    )

    runtime_env = agent._runtime_env()

    assert "HTTP_PROXY" not in runtime_env
    assert "HTTPS_PROXY" not in runtime_env
    assert runtime_env["NO_PROXY"] == "127.0.0.1,localhost,::1"


def test_agent_still_forwards_remotely_reachable_proxies(tmp_path: Path) -> None:
    proxy_url = "http://proxy.example.test:8080"
    agent = make_agent(
        tmp_path,
        extra_env={
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
        },
    )

    runtime_env = agent._runtime_env()

    assert runtime_env["HTTP_PROXY"] == proxy_url
    assert runtime_env["HTTPS_PROXY"] == proxy_url


def test_agent_strips_generic_harbor_provider_prefix(tmp_path: Path) -> None:
    agent = NexAUAgent(
        logs_dir=tmp_path,
        model_name="openrouter/openai/gpt-5",
        extra_env={"OPENROUTER_API_KEY": "test-key"},
    )

    runtime_env = agent._runtime_env()

    assert runtime_env["LLM_MODEL"] == "openai/gpt-5"
    assert runtime_env["LLM_API_KEY"] == "test-key"
    assert runtime_env["LLM_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert runtime_env["LLM_KIND"] == "openai_chat_completion"


def test_official_harbor_factory_loads_custom_import_path(tmp_path: Path) -> None:
    agent = AgentFactory.create_agent_from_import_path(
        "nexau_harbor.agent:NexAUAgent",
        logs_dir=tmp_path,
        model_name="openai/gpt-5",
        extra_env={
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://example.test/v1",
        },
    )
    assert isinstance(agent, NexAUAgent)


def test_agent_requires_existing_custom_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="NexAU config file not found"):
        make_agent(tmp_path, config_path=tmp_path / "missing.yaml")


@pytest.mark.asyncio
async def test_install_uploads_config_and_pinned_runner(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    environment = FakeEnvironment()

    await agent.install(environment)  # type: ignore[arg-type]

    assert environment.uploaded_dirs == [
        (agent._config_dir, REMOTE_WORKSPACE.as_posix())
    ]
    assert environment.uploaded_files[0][1] == "/nexau-workspace/runner.py"
    install_commands = "\n".join(call["command"] for call in environment.exec_calls)
    assert PREPARE_DEBIAN_CA_CERTIFICATES_JAVA in install_commands
    assert "/etc/ssl/certs/java" in environment.exec_calls[0]["command"]
    assert NEXAU_GIT_COMMIT in install_commands
    assert "chown -R 10001:10001 /nexau-workspace /opt/nexau-venv" in install_commands
    assert "uv venv --clear /opt/nexau-venv" in install_commands


@pytest.mark.asyncio
async def test_install_uses_image_default_user_identity_when_task_user_is_unset(
    tmp_path: Path,
) -> None:
    agent = make_agent(tmp_path)
    environment = FakeEnvironment(
        default_user=None,
        identity_stdout="65534\n65534\n",
    )

    await agent.install(environment)  # type: ignore[arg-type]

    identity_call = next(
        call for call in environment.exec_calls if "id -u && id -g" in call["command"]
    )
    assert identity_call["user"] is None
    install_commands = "\n".join(call["command"] for call in environment.exec_calls)
    assert "chown -R 65534:65534 /nexau-workspace /opt/nexau-venv" in install_commands


@pytest.mark.asyncio
async def test_install_rejects_unparseable_agent_identity(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    environment = FakeEnvironment(default_user=None, identity_stdout="unknown\n")

    with pytest.raises(RuntimeError, match="effective agent UID/GID"):
        await agent.install(environment)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_run_quotes_instruction_and_passes_runtime_env(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    environment = FakeEnvironment()

    await agent.run(
        "write 'quoted value' to /app/result.txt",
        environment,  # type: ignore[arg-type]
        AgentContext(),
    )

    call = environment.exec_calls[-1]
    assert "/nexau-workspace/runner.py" in call["command"]
    assert "--query" in call["command"]
    assert "quoted value" in call["command"]
    assert call["env"]["LLM_MODEL"] == "gpt-5"
    assert call["env"]["LLM_API_KEY"] == "test-key"
    assert NEXAU_DETACHED_EXEC_ENV not in call["env"]


@pytest.mark.asyncio
async def test_run_requests_detached_execution_only_from_supporting_environment(
    tmp_path: Path,
) -> None:
    agent = make_agent(tmp_path)
    environment = DetachedFakeEnvironment()

    await agent.run(
        "inspect /app/input.bin",
        environment,  # type: ignore[arg-type]
        AgentContext(),
    )

    call = environment.exec_calls[-1]
    assert call["env"][NEXAU_DETACHED_EXEC_ENV] == "1"
