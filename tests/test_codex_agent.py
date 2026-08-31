from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.factory import AgentFactory
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.utils.import_path import import_class

from nexau_harbor import codex_agent
from nexau_harbor.codex_agent import TraceSafeCodexAgent
from nexau_harbor.constants import NEXAU_DETACHED_EXEC_ENV


class FakeEnvironment:
    SUPPORTS_DETACHED_AGENT_EXEC = True

    def __init__(self, *, missing_system_dependencies: bool = False) -> None:
        self.default_user = "agent"
        self.missing_system_dependencies = missing_system_dependencies
        self.exec_calls: list[dict[str, Any]] = []
        self.uploaded_files: list[tuple[Path | str, str]] = []
        self.uploaded_payloads: dict[str, bytes] = {}

    async def exec(self, **kwargs: Any) -> ExecResult:
        self.exec_calls.append(kwargs)
        command = str(kwargs.get("command", ""))
        if self.missing_system_dependencies:
            if command.startswith("command -v curl"):
                return ExecResult(stdout="", stderr="", return_code=1)
            if "for manager in apt-get dnf yum apk" in command:
                return ExecResult(stdout="apt-get", stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploaded_files.append((source_path, target_path))
        self.uploaded_payloads[target_path] = Path(source_path).read_bytes()


def make_agent(
    tmp_path: Path,
    model_name: str = "gpt-5.5",
    **agent_kwargs: Any,
) -> TraceSafeCodexAgent:
    agent_kwargs.setdefault("upstream_model_name", "nex-agi/Nex-N2-Pro")
    return TraceSafeCodexAgent(
        logs_dir=tmp_path,
        model_name=model_name,
        version="0.148.0",
        reasoning_effort="high",
        reasoning_summary="detailed",
        extra_env={
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://example.test/v1",
        },
        **agent_kwargs,
    )


def test_trace_safe_codex_import_path_is_loadable() -> None:
    agent_class = import_class(
        "nexau_harbor.codex_agent:TraceSafeCodexAgent",
        label="agent",
    )

    assert agent_class is TraceSafeCodexAgent


def test_harbor_factory_constructs_trace_safe_codex(tmp_path: Path) -> None:
    agent = AgentFactory.create_agent_from_import_path(
        "nexau_harbor.codex_agent:TraceSafeCodexAgent",
        logs_dir=tmp_path,
        model_name="gpt-5.5",
        version="0.148.0",
        upstream_model_name="nex-agi/Nex-N2-Pro",
        extra_env={
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://example.test/v1",
        },
    )

    assert isinstance(agent, TraceSafeCodexAgent)
    assert agent._runtime_model_name() == "gpt-5.5"
    assert agent._northgate_upstream_model_name == "nex-agi/Nex-N2-Pro"


def test_codex_agent_uses_standard_function_tools_with_bridge_provider(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)

    assert agent._runtime_model_name() == "gpt-5.5"
    assert agent._build_trace_config("https://example.test/v1") == {
        "model_provider": "northgate",
        "model_context_window": 262144,
        "web_search": "disabled",
        "model_providers": {
            "northgate": {
                "name": "NorthGate via Responses bridge",
                "base_url": "http://127.0.0.1:48123/v1",
                "env_key": "OPENAI_API_KEY",
                "wire_api": "responses",
                "supports_websockets": False,
                "supports_standalone_web_search": False,
            }
        },
        "history": {"persistence": "save-all"},
        "hide_agent_reasoning": False,
    }
    assert agent.version() == "0.148.0"
    assert agent.build_cli_flags().endswith(
        "--enable unified_exec --disable code_mode --disable code_mode_only "
        "--disable multi_agent --disable multi_agent_v2"
    )
    assert agent.extra_env["NO_PROXY"] == "127.0.0.1,localhost"
    assert agent.extra_env["no_proxy"] == "127.0.0.1,localhost"
    assert not agent.SUPPORTS_ATIF
    assert not agent.SUPPORTS_LOAD_ATIF_TRAJECTORY


def test_codex_agent_preserves_namespaced_model_for_matching_provider(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, model_name="openai/nex-agi/Nex-N2-Pro")

    assert agent._runtime_model_name() == "nex-agi/Nex-N2-Pro"


async def test_codex_agent_writes_raw_streams_and_sessions_during_run(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    environment = FakeEnvironment()

    await agent.run("fix the task", environment, AgentContext())

    commands = [call["command"] for call in environment.exec_calls]
    setup_command = next(command for command in commands if '"$CODEX_HOME/sessions"' in command)
    run_call = next(call for call in environment.exec_calls if "codex exec" in call["command"])
    run_command = run_call["command"]

    assert "ln -s /logs/agent/sessions" in setup_command
    assert "ln -s /logs/agent/codex_history.jsonl" in setup_command
    assert "--model gpt-5.5" in run_command
    assert run_command.count("--enable unified_exec") == 1
    assert "--disable code_mode" in run_command
    assert "--disable code_mode_only" in run_command
    assert "--disable multi_agent" in run_command
    assert "--disable multi_agent_v2" in run_command
    assert "set -o pipefail" in run_command
    assert "2>/logs/agent/codex_stderr.log" in run_command
    assert "| tee /logs/agent/codex_events.jsonl >/dev/null" in run_command
    assert "2>&1" not in run_command
    assert 'codex_status="${PIPESTATUS[0]}"' in run_command
    assert "/logs/agent/codex_result.json" in run_command
    assert run_call["env"][NEXAU_DETACHED_EXEC_ENV] == "1"
    assert any("nohup node" in command for command in commands)
    assert any('kill "$bridge_pid"' in command for command in commands)
    bridge_path = "/tmp/northgate-codex-bridge/bridge.js"
    assert bridge_path in environment.uploaded_payloads
    assert b"customToolToFunction" in environment.uploaded_payloads[bridge_path]
    bridge_start_call = next(
        call for call in environment.exec_calls if "nohup node" in call["command"]
    )
    assert bridge_start_call["env"]["NORTHGATE_UPSTREAM_MODEL"] == "nex-agi/Nex-N2-Pro"
    syntax = subprocess.run(
        ["bash", "-n"],
        input=run_command,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


async def test_codex_agent_can_disable_optional_responses_bridge(tmp_path: Path) -> None:
    agent = make_agent(
        tmp_path,
        responses_bridge=False,
        upstream_model_name=None,
    )
    environment = FakeEnvironment()

    await agent.run("fix the task", environment, AgentContext())

    commands = [call["command"] for call in environment.exec_calls]
    run_command = next(command for command in commands if "codex exec" in command)
    assert "--disable code_mode" in run_command
    assert "--disable code_mode_only" in run_command
    assert not any("nohup node" in command for command in commands)
    assert not any('kill "$bridge_pid"' in command for command in commands)
    bridge_path = "/tmp/northgate-codex-bridge/bridge.js"
    assert bridge_path not in environment.uploaded_payloads


async def test_codex_install_uses_detached_e2b_execution(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    environment = FakeEnvironment(missing_system_dependencies=True)

    await agent.install(environment)

    repair_call = next(
        call
        for call in environment.exec_calls
        if "/etc/ssl/certs/java" in call["command"]
    )
    dependency_call = next(
        call for call in environment.exec_calls if "apt-get update" in call["command"]
    )
    install_call = next(
        call
        for call in environment.exec_calls
        if "npm install -g @openai/codex" in call["command"]
    )
    symlink_call = next(
        call for call in environment.exec_calls if "ln -sf" in call["command"]
    )
    assert environment.exec_calls.index(repair_call) < environment.exec_calls.index(
        dependency_call
    )
    assert environment.exec_calls.index(dependency_call) < environment.exec_calls.index(
        install_call
    )
    assert environment.exec_calls.index(install_call) < environment.exec_calls.index(
        symlink_call
    )
    for call in (repair_call, dependency_call, install_call, symlink_call):
        assert call["env"][NEXAU_DETACHED_EXEC_ENV] == "1"
    assert dependency_call["env"]["DEBIAN_FRONTEND"] == "noninteractive"


async def test_codex_root_execution_outside_install_stays_attached(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    environment = FakeEnvironment()

    await agent.exec_as_root(environment, "true", env={"VISIBLE": "yes"})

    assert environment.exec_calls[-1]["env"] == {"VISIBLE": "yes"}


async def test_codex_shell_pipeline_keeps_jsonl_clean_and_propagates_exit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    remote_root = tmp_path / "remote"
    remote_logs = remote_root / "logs/agent"
    remote_home = remote_root / "codex-home"
    remote_secrets = remote_root / "codex-secrets"
    monkeypatch.setattr(codex_agent, "REMOTE_LOG_DIR", PurePosixPath(remote_logs))
    monkeypatch.setattr(TraceSafeCodexAgent, "_REMOTE_CODEX_HOME", PurePosixPath(remote_home))
    monkeypatch.setattr(
        TraceSafeCodexAgent,
        "_REMOTE_CODEX_SECRETS_DIR",
        PurePosixPath(remote_secrets),
    )
    agent = make_agent(tmp_path / "host")
    environment = FakeEnvironment()

    await agent.run("fix the task", environment, AgentContext())

    run_command = next(
        call["command"] for call in environment.exec_calls if "codex exec" in call["command"]
    )
    sessions = remote_logs / "sessions"
    sessions.mkdir(parents=True)
    remote_home.mkdir(parents=True)
    (remote_home / "sessions").symlink_to(sessions, target_is_directory=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_codex = fake_bin / "codex"
    fake_codex.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"thread-shell\"}'\n"
        "printf '%s\\n' 'diagnostic only' >&2\n"
        "printf '%s\\n' '{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":3,"
        "\"cached_input_tokens\":1,\"output_tokens\":2}}'\n"
        "printf '%s\\n' '{\"type\":\"session_meta\",\"payload\":{\"id\":\"thread-shell\"}}' "
        '> "$CODEX_HOME/sessions/rollout-2026-08-19T00-00-00-thread-shell.jsonl"\n'
        'exit "${FAKE_CODEX_EXIT:-0}"\n',
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    command_env = os.environ.copy()
    command_env.update(
        {
            "CODEX_HOME": str(remote_home),
            "PATH": f"{fake_bin}:{command_env['PATH']}",
            "FAKE_CODEX_EXIT": "7",
        }
    )

    completed = subprocess.run(
        ["bash", "-c", run_command],
        env=command_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 7
    event_lines = [
        json.loads(line)
        for line in (remote_logs / "codex_events.jsonl").read_text().splitlines()
    ]
    assert [event["type"] for event in event_lines] == ["thread.started", "turn.completed"]
    assert (remote_logs / "codex_stderr.log").read_text() == "diagnostic only\n"
    result = json.loads((remote_logs / "codex_result.json").read_text())
    assert result["status"] == "failed"
    assert result["exit_code"] == 7
    assert next(sessions.glob("rollout-*.jsonl")).is_file()

    for rollout in sessions.glob("rollout-*.jsonl"):
        rollout.unlink()
    fake_codex.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"thread-killed\"}'\n"
        "printf '%s\\n' '{\"type\":\"session_meta\",\"payload\":{\"id\":\"thread-killed\"}}' "
        '> "$CODEX_HOME/sessions/rollout-2026-08-19T00-00-00-thread-killed.jsonl"\n'
        "sleep 60\n",
        encoding="utf-8",
    )
    interrupted = subprocess.Popen(
        ["bash", "-c", run_command],
        env={**command_env, "FAKE_CODEX_EXIT": "0"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        interrupted_events = remote_logs / "codex_events.jsonl"
        if (
            interrupted_events.is_file()
            and interrupted_events.read_text().strip()
            and any(sessions.glob("rollout-*.jsonl"))
        ):
            break
        time.sleep(0.01)
    else:
        interrupted.kill()
        interrupted.communicate(timeout=3)
        raise AssertionError("fake Codex did not persist its initial trace")
    os.killpg(interrupted.pid, signal.SIGKILL)
    interrupted.communicate(timeout=3)

    interrupted_agent = make_agent(remote_logs)
    interrupted_context = AgentContext()
    interrupted_agent.populate_context_post_run(interrupted_context)
    interrupted_result = json.loads((remote_logs / "codex_result.json").read_text())
    assert interrupted_result["status"] == "incomplete"
    assert interrupted_result["trace_complete"] is False
    assert interrupted_result["thread_ids"] == ["thread-killed"]
    assert next(sessions.glob("rollout-*.jsonl")).is_file()


def test_post_run_writes_manifest_and_metrics_without_trajectory(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "item-1", "type": "agent_message", "text": "finished"},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 120,
                "cached_input_tokens": 80,
                "output_tokens": 30,
                "reasoning_output_tokens": 20,
            },
        },
    ]
    (tmp_path / "codex_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (tmp_path / "codex_stderr.log").write_text("", encoding="utf-8")
    (tmp_path / "northgate-responses-bridge.log").write_text(
        "NorthGate Responses bridge listening\n",
        encoding="utf-8",
    )
    (tmp_path / "codex_history.jsonl").write_text(
        json.dumps({"session_id": "thread-1", "text": "fix the task"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "codex_result.json").write_text(
        json.dumps({"schema_version": 1, "status": "completed", "exit_code": 0}),
        encoding="utf-8",
    )
    rollout = tmp_path / "sessions" / "2026" / "08" / "19" / "rollout-test.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "thread-1"}}) + "\n",
        encoding="utf-8",
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    result = json.loads((tmp_path / "codex_result.json").read_text())
    manifest = json.loads((tmp_path / "trace_manifest.json").read_text())
    assert result["status"] == "completed"
    assert result["trace_complete"] is True
    assert result["trajectory_generated"] is False
    assert result["thread_ids"] == ["thread-1"]
    assert result["final_response"] == "finished"
    assert manifest["trace_complete"] is True
    assert manifest["artifacts"]["events"]["invalid_json_lines"] == 0
    assert manifest["artifacts"]["history"]["invalid_json_lines"] == 0
    assert manifest["artifacts"]["bridge_log"]["size_bytes"] > 0
    assert manifest["models"] == {
        "advertised": "gpt-5.5",
        "upstream": "nex-agi/Nex-N2-Pro",
    }
    assert result["model"] == "gpt-5.5"
    assert result["upstream_model"] == "nex-agi/Nex-N2-Pro"
    assert manifest["artifacts"]["rollouts"][0]["path"].endswith("rollout-test.jsonl")
    assert context.n_input_tokens == 120
    assert context.n_cache_tokens == 80
    assert context.n_output_tokens == 30
    assert not (tmp_path / "trajectory.json").exists()


def test_incomplete_stream_uses_rollout_usage_and_stays_incomplete(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    (tmp_path / "codex_events.jsonl").write_bytes(
        b'{"type":"thread.started","thread_id":"thread-2"}\n{"type":"item.started"'
    )
    (tmp_path / "codex_result.json").write_text(
        json.dumps({"schema_version": 1, "status": "running", "exit_code": None}),
        encoding="utf-8",
    )
    rollout = tmp_path / "sessions" / "2026" / "08" / "19" / "rollout-test.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 41,
                            "cached_input_tokens": 17,
                            "output_tokens": 13,
                            "reasoning_output_tokens": 8,
                            "total_tokens": 54,
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    result = json.loads((tmp_path / "codex_result.json").read_text())
    manifest = json.loads((tmp_path / "trace_manifest.json").read_text())
    assert result["status"] == "incomplete"
    assert result["trace_status"] == "running"
    assert result["trace_complete"] is False
    assert manifest["event_summary"]["invalid_json_lines"] == 1
    assert context.n_input_tokens == 41
    assert context.n_cache_tokens == 17
    assert context.n_output_tokens == 13


def test_recoverable_error_event_does_not_mark_trace_complete(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    events = [
        {"type": "thread.started", "thread_id": "thread-retry"},
        {"type": "turn.started"},
        {"type": "error", "message": "Reconnecting... 1/5 (request timed out)"},
    ]
    (tmp_path / "codex_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (tmp_path / "codex_result.json").write_text(
        json.dumps({"schema_version": 1, "status": "running", "exit_code": None}),
        encoding="utf-8",
    )
    rollout = tmp_path / "sessions/2026/08/23/rollout-retry.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "thread-retry"}})
        + "\n",
        encoding="utf-8",
    )

    agent.populate_context_post_run(AgentContext())

    result = json.loads((tmp_path / "codex_result.json").read_text())
    manifest = json.loads((tmp_path / "trace_manifest.json").read_text())
    assert result["status"] == "incomplete"
    assert result["trace_status"] == "running"
    assert result["trace_complete"] is False
    assert manifest["event_summary"]["event_type_counts"]["error"] == 1


def test_post_run_does_not_invent_artifacts_when_download_failed(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.is_empty()
    assert not (tmp_path / "codex_result.json").exists()
    assert not (tmp_path / "trace_manifest.json").exists()
