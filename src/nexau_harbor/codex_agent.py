"""Trace-preserving Codex harness for Harbor runs."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, override

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from nexau_harbor.constants import NEXAU_DETACHED_EXEC_ENV, REMOTE_LOG_DIR
from nexau_harbor.system_dependencies import PREPARE_DEBIAN_CA_CERTIFICATES_JAVA
from nexau_harbor.trace_artifacts import (
    CODEX_BRIDGE_LOG_NAME,
    CODEX_EVENTS_NAME,
    CODEX_HISTORY_NAME,
    CODEX_RESULT_NAME,
    CODEX_SESSIONS_DIR_NAME,
    CODEX_STDERR_NAME,
    CODEX_TRACE_MANIFEST_NAME,
)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_record(path: Path, root: Path) -> dict[str, Any]:
    line_count = 0
    valid_lines = 0
    invalid_lines = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            line_count += 1
            try:
                value = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid_lines += 1
                continue
            if isinstance(value, dict):
                valid_lines += 1
            else:
                invalid_lines += 1
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "line_count": line_count,
        "valid_json_lines": valid_lines,
        "invalid_json_lines": invalid_lines,
    }


def _file_record(path: Path, root: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _event_stream_summary(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    thread_ids: list[str] = []
    final_response: str | None = None
    trace_status = "missing"
    usage: dict[str, Any] | None = None
    valid_lines = 0
    invalid_lines = 0

    if not path.is_file():
        return {
            "event_type_counts": {},
            "thread_ids": [],
            "final_response": None,
            "trace_status": trace_status,
            "trace_complete": False,
            "usage": None,
            "valid_json_lines": 0,
            "invalid_json_lines": 0,
        }

    with path.open("rb") as handle:
        for raw_line in handle:
            try:
                event = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid_lines += 1
                continue
            if not isinstance(event, dict):
                invalid_lines += 1
                continue

            valid_lines += 1
            event_type = event.get("type")
            if not isinstance(event_type, str):
                event_type = "<missing>"
            counts[event_type] += 1

            if event_type == "thread.started":
                thread_id = event.get("thread_id")
                if isinstance(thread_id, str) and thread_id not in thread_ids:
                    thread_ids.append(thread_id)
                trace_status = "running"
            elif event_type == "turn.started":
                trace_status = "running"
            elif event_type == "turn.completed":
                trace_status = "completed"
                candidate_usage = event.get("usage")
                if isinstance(candidate_usage, dict):
                    usage = candidate_usage
            elif event_type == "turn.failed":
                trace_status = "failed"
            elif event_type == "item.completed":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str):
                        final_response = text

    return {
        "event_type_counts": dict(sorted(counts.items())),
        "thread_ids": thread_ids,
        "final_response": final_response,
        "trace_status": trace_status,
        "trace_complete": trace_status in {"completed", "failed"} and invalid_lines == 0,
        "usage": usage,
        "valid_json_lines": valid_lines,
        "invalid_json_lines": invalid_lines,
    }


def _rollout_usage(session_files: list[Path]) -> dict[str, int] | None:
    totals: Counter[str] = Counter()
    saw_usage = False
    for path in session_files:
        last_usage: dict[str, Any] | None = None
        with path.open("rb") as handle:
            for raw_line in handle:
                try:
                    event = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(event, dict) or event.get("type") != "event_msg":
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict) or payload.get("type") != "token_count":
                    continue
                info = payload.get("info")
                candidate = info.get("total_token_usage") if isinstance(info, dict) else None
                if isinstance(candidate, dict):
                    last_usage = candidate
        if last_usage is None:
            continue
        saw_usage = True
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        ):
            value = last_usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    return dict(totals) if saw_usage else None


class TraceSafeCodexAgent(Codex):
    """Run Codex while making every public trace event durable during execution."""

    DEFAULT_CODEX_VERSION = "0.148.0"
    DEFAULT_BRIDGE_PORT = 48123
    DEFAULT_TRANSIENT_RETRY_ATTEMPTS = 8
    DEFAULT_TRANSIENT_RETRY_BASE_DELAY_SEC = 5
    DEFAULT_TRANSIENT_RETRY_MAX_DELAY_SEC = 120
    _BRIDGE_SOURCE = Path(__file__).with_name("codex_responses_bridge.js")
    _REMOTE_BRIDGE_DIR = PurePosixPath("/tmp/northgate-codex-bridge")
    _REMOTE_BRIDGE_SCRIPT = _REMOTE_BRIDGE_DIR / "bridge.js"
    _REMOTE_BRIDGE_PID = _REMOTE_BRIDGE_DIR / "bridge.pid"
    _REMOTE_BRIDGE_LOG = REMOTE_LOG_DIR / CODEX_BRIDGE_LOG_NAME

    SUPPORTS_ATIF = False
    SUPPORTS_LOAD_ATIF_TRAJECTORY = False

    def __init__(
        self,
        *args: Any,
        unified_exec: bool = True,
        code_mode: bool = False,
        code_mode_only: bool = False,
        multi_agent: bool = False,
        responses_bridge: bool = True,
        bridge_port: int = DEFAULT_BRIDGE_PORT,
        provider_name: str = "northgate",
        upstream_model_name: str | None = None,
        context_window: int = 262144,
        transient_retry_attempts: int = DEFAULT_TRANSIENT_RETRY_ATTEMPTS,
        transient_retry_base_delay_sec: int = DEFAULT_TRANSIENT_RETRY_BASE_DELAY_SEC,
        transient_retry_max_delay_sec: int = DEFAULT_TRANSIENT_RETRY_MAX_DELAY_SEC,
        version: str | None = None,
        **kwargs: Any,
    ) -> None:
        normalized_provider = provider_name.strip()
        normalized_upstream_model = (
            upstream_model_name.strip() if upstream_model_name is not None else None
        )
        if not normalized_provider:
            raise ValueError("provider_name must not be empty")
        if upstream_model_name is not None and not normalized_upstream_model:
            raise ValueError("upstream_model_name must not be empty")
        if context_window < 1:
            raise ValueError("context_window must be positive")
        if bridge_port < 1 or bridge_port > 65535:
            raise ValueError("bridge_port must be between 1 and 65535")
        if code_mode_only and not code_mode:
            raise ValueError("code_mode_only requires code_mode")
        if responses_bridge and multi_agent:
            raise ValueError("responses_bridge does not support namespace multi-agent tools")
        if normalized_upstream_model and not responses_bridge:
            raise ValueError("upstream_model_name requires responses_bridge")
        retry_values = {
            "transient_retry_attempts": transient_retry_attempts,
            "transient_retry_base_delay_sec": transient_retry_base_delay_sec,
            "transient_retry_max_delay_sec": transient_retry_max_delay_sec,
        }
        for name, value in retry_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if transient_retry_max_delay_sec < transient_retry_base_delay_sec:
            raise ValueError(
                "transient_retry_max_delay_sec must be at least "
                "transient_retry_base_delay_sec"
            )

        extra_env = dict(kwargs.pop("extra_env", None) or {})
        for env_name in ("NO_PROXY", "no_proxy"):
            entries = [
                item.strip()
                for item in extra_env.get(env_name, "").split(",")
                if item.strip()
            ]
            for host in ("127.0.0.1", "localhost"):
                if host not in entries:
                    entries.append(host)
            extra_env[env_name] = ",".join(entries)

        self._northgate_provider_name = normalized_provider
        self._northgate_upstream_model_name = normalized_upstream_model
        self._northgate_context_window = int(context_window)
        self._northgate_unified_exec = bool(unified_exec)
        self._northgate_code_mode = bool(code_mode)
        self._northgate_code_mode_only = bool(code_mode_only)
        self._northgate_multi_agent = bool(multi_agent)
        self._northgate_responses_bridge = bool(responses_bridge)
        self._northgate_bridge_port = int(bridge_port)
        self._transient_retry_attempts = transient_retry_attempts
        self._transient_retry_base_delay_sec = transient_retry_base_delay_sec
        self._transient_retry_max_delay_sec = transient_retry_max_delay_sec
        self._northgate_detached_install = False

        kwargs.setdefault("reasoning_effort", "high")
        kwargs.setdefault("reasoning_summary", "detailed")
        kwargs.setdefault("web_search", "disabled")
        super().__init__(
            *args,
            version=version or self.DEFAULT_CODEX_VERSION,
            extra_env=extra_env,
            **kwargs,
        )

    @staticmethod
    @override
    def name() -> str:
        return "codex"

    @override
    async def exec_as_root(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        runtime_env = dict(env or {})
        if (
            getattr(environment, "SUPPORTS_DETACHED_AGENT_EXEC", False)
            and self._northgate_detached_install
        ):
            runtime_env[NEXAU_DETACHED_EXEC_ENV] = "1"
        return await super().exec_as_root(
            environment,
            command,
            env=runtime_env or None,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )

    @override
    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        runtime_env = dict(env or {})
        if getattr(environment, "SUPPORTS_DETACHED_AGENT_EXEC", False) and (
            self._northgate_detached_install or "codex exec " in command
        ):
            runtime_env[NEXAU_DETACHED_EXEC_ENV] = "1"
        return await super().exec_as_agent(
            environment,
            command,
            env=runtime_env or None,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        self._northgate_detached_install = True
        try:
            bundled_codex = getattr(environment, "preinstalled_codex_path", None)
            if bundled_codex:
                bundled_rg = getattr(environment, "preinstalled_ripgrep_path", None)
                expected_version = f"codex-cli {self.version()}"
                commands = [
                    "set -e",
                    f"test -x {shlex.quote(str(bundled_codex))}",
                    f"actual_version=$({shlex.quote(str(bundled_codex))} --version)",
                    f"if [ \"$actual_version\" != {shlex.quote(expected_version)} ]; then",
                    '  printf \'Codex bundle version mismatch: %s\\n\' "$actual_version" >&2',
                    "  exit 42",
                    "fi",
                    f"ln -sf {shlex.quote(str(bundled_codex))} /usr/local/bin/codex",
                ]
                if bundled_rg:
                    commands.extend(
                        (
                            f"test -x {shlex.quote(str(bundled_rg))}",
                            f"ln -sf {shlex.quote(str(bundled_rg))} /usr/local/bin/rg",
                        )
                    )
                commands.append("codex --version")
                await self.exec_as_root(environment, command="\n".join(commands))
                return
            await self.exec_as_root(
                environment,
                command=PREPARE_DEBIAN_CA_CERTIFICATES_JAVA,
            )
            await super().install(environment)
        finally:
            self._northgate_detached_install = False

    @override
    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags()
        feature_flags = [
            (
                "--enable unified_exec"
                if self._northgate_unified_exec
                else "--disable unified_exec"
            )
        ]
        feature_flags.append(
            "--enable code_mode"
            if self._northgate_code_mode
            else "--disable code_mode"
        )
        feature_flags.append(
            "--enable code_mode_only"
            if self._northgate_code_mode_only
            else "--disable code_mode_only"
        )
        if self._northgate_multi_agent:
            feature_flags.append("--enable multi_agent")
        else:
            feature_flags.extend(("--disable multi_agent", "--disable multi_agent_v2"))
        return " ".join(part for part in (flags, *feature_flags) if part)

    @property
    def _bridge_base_url(self) -> str:
        return f"http://127.0.0.1:{self._northgate_bridge_port}/v1"

    def _runtime_model_name(self) -> str:
        model = self.model_name
        if not model:
            raise ValueError("Model name is required")
        provider = self.model_connection.provider
        prefix, separator, remainder = model.partition("/")
        if separator and provider and prefix.lower() == provider.lower():
            return remainder
        return model

    @override
    def _build_effective_config(
        self,
        openai_base_url: str | None = None,
    ) -> dict[str, Any]:
        if not openai_base_url:
            raise ValueError(
                "NorthGate Codex requires OPENAI_BASE_URL, for example "
                "https://northgate.xiaobei.top/v1"
            )

        config = super()._build_effective_config(openai_base_url)
        config.pop("openai_base_url", None)
        config["model_provider"] = self._northgate_provider_name
        config["model_context_window"] = self._northgate_context_window
        config["web_search"] = "disabled"

        providers = config.setdefault("model_providers", {})
        if not isinstance(providers, dict):
            raise ValueError("Codex model_providers config must be a table")
        existing = providers.get(self._northgate_provider_name, {})
        if not isinstance(existing, dict):
            raise ValueError(
                f"Codex provider {self._northgate_provider_name!r} must be a table"
            )
        provider = dict(existing)
        provider.update(
            {
                "name": (
                    "NorthGate via Responses bridge"
                    if self._northgate_responses_bridge
                    else "NorthGate"
                ),
                "base_url": (
                    self._bridge_base_url
                    if self._northgate_responses_bridge
                    else openai_base_url.rstrip("/")
                ),
                "env_key": "OPENAI_API_KEY",
                "wire_api": "responses",
                "supports_websockets": False,
                "supports_standalone_web_search": False,
            }
        )
        providers[self._northgate_provider_name] = provider
        return config

    def _build_trace_config(self, openai_base_url: str | None) -> dict[str, Any]:
        config = self._build_effective_config(openai_base_url)
        history = config.setdefault("history", {})
        if not isinstance(history, dict):
            raise ValueError("Invalid Codex config: history must be a TOML table")
        history["persistence"] = "save-all"
        history.pop("max_bytes", None)
        config["hide_agent_reasoning"] = False
        return config

    async def _start_responses_bridge(
        self,
        environment: BaseEnvironment,
        upstream_base_url: str,
    ) -> None:
        if not self._BRIDGE_SOURCE.is_file():
            raise RuntimeError(f"NorthGate bridge source is missing: {self._BRIDGE_SOURCE}")

        bridge_dir = shlex.quote(self._REMOTE_BRIDGE_DIR.as_posix())
        agent_dir = shlex.quote(REMOTE_LOG_DIR.as_posix())
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {bridge_dir} {agent_dir}",
        )
        await self._stop_responses_bridge(environment)
        await environment.upload_file(
            self._BRIDGE_SOURCE,
            self._REMOTE_BRIDGE_SCRIPT.as_posix(),
        )
        if environment.default_user is not None:
            owner = shlex.quote(str(environment.default_user))
            script = shlex.quote(self._REMOTE_BRIDGE_SCRIPT.as_posix())
            await self.exec_as_root(
                environment,
                command=f"chown {owner} {script} && chmod 700 {script}",
            )

        script = shlex.quote(self._REMOTE_BRIDGE_SCRIPT.as_posix())
        pid_file = shlex.quote(self._REMOTE_BRIDGE_PID.as_posix())
        log_file = shlex.quote(self._REMOTE_BRIDGE_LOG.as_posix())
        health_url = shlex.quote(f"http://127.0.0.1:{self._northgate_bridge_port}/health")
        command = "\n".join(
            (
                "set -euo pipefail",
                f"mkdir -p {bridge_dir} {agent_dir}",
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi",
                f"nohup node {script} > {log_file} 2>&1 </dev/null &",
                "bridge_pid=$!",
                f"printf '%s\\n' \"$bridge_pid\" > {pid_file}",
                "for attempt in $(seq 1 100); do",
                f"  if curl --silent --show-error --fail {health_url} "
                "| grep -Fq '\"status\":\"ok\"'; then",
                "    exit 0",
                "  fi",
                "  if ! kill -0 \"$bridge_pid\" 2>/dev/null; then",
                "    break",
                "  fi",
                "  sleep 0.1",
                "done",
                f"tail -n 50 {log_file} >&2 || true",
                "exit 1",
            )
        )
        bridge_env = {
            "NORTHGATE_UPSTREAM_BASE_URL": upstream_base_url.rstrip("/"),
            "NORTHGATE_BRIDGE_PORT": str(self._northgate_bridge_port),
        }
        if self._northgate_upstream_model_name:
            bridge_env["NORTHGATE_UPSTREAM_MODEL"] = self._northgate_upstream_model_name
        await self.exec_as_agent(
            environment,
            command=command,
            env=bridge_env,
        )

    async def _stop_responses_bridge(self, environment: BaseEnvironment) -> None:
        pid_file = shlex.quote(self._REMOTE_BRIDGE_PID.as_posix())
        await self.exec_as_agent(
            environment,
            command=" ".join(
                (
                    f"if test -s {pid_file}; then",
                    f"bridge_pid=$(tr -d '\\r\\n' < {pid_file});",
                    "if printf '%s' \"$bridge_pid\" | grep -Eq '^[0-9]+$'; then",
                    "kill \"$bridge_pid\" 2>/dev/null || true;",
                    "fi;",
                    "fi",
                )
            ),
        )

    @with_prompt_template
    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        model = self._runtime_model_name()
        escaped_instruction = shlex.quote(instruction)
        cli_flags = self.build_cli_flags()
        cli_flags_arg = f"{cli_flags} " if cli_flags else ""

        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        remote_secrets_dir = self._REMOTE_CODEX_SECRETS_DIR.as_posix()
        remote_auth_path = (self._REMOTE_CODEX_SECRETS_DIR / "auth.json").as_posix()
        remote_config_path = (self._REMOTE_CODEX_HOME / "config.toml").as_posix()
        remote_sessions_dir = (REMOTE_LOG_DIR / CODEX_SESSIONS_DIR_NAME).as_posix()
        events_path = (REMOTE_LOG_DIR / CODEX_EVENTS_NAME).as_posix()
        stderr_path = (REMOTE_LOG_DIR / CODEX_STDERR_NAME).as_posix()
        result_path = (REMOTE_LOG_DIR / CODEX_RESULT_NAME).as_posix()
        history_path = (REMOTE_LOG_DIR / CODEX_HISTORY_NAME).as_posix()

        env: dict[str, str] = {"CODEX_HOME": remote_codex_home}
        await self.exec_as_agent(
            environment,
            command=(
                f'mkdir -p "$CODEX_HOME" {shlex.quote(remote_secrets_dir)} '
                f"{shlex.quote(REMOTE_LOG_DIR.as_posix())} {shlex.quote(remote_sessions_dir)}"
            ),
            env=env,
        )

        if self._load:
            await self._seed_load_trajectory(environment)
        resume_run = self._resume or self._load

        auth_json_path = self._resolve_auth_json_path()
        access = self.model_connection
        if auth_json_path:
            await environment.upload_file(auth_json_path, remote_auth_path)
            if environment.default_user is not None:
                await self.exec_as_root(
                    environment,
                    command=f"chown {environment.default_user} {shlex.quote(remote_auth_path)}",
                )
            setup_command = (
                f"ln -sf {shlex.quote(remote_auth_path)} \"$CODEX_HOME/auth.json\"\n"
            )
        else:
            env["OPENAI_API_KEY"] = access.api_key or ""
            setup_command = (
                f"cat >{shlex.quote(remote_auth_path)} <<EOF\n"
                '{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\nEOF\n'
                f"ln -sf {shlex.quote(remote_auth_path)} \"$CODEX_HOME/auth.json\"\n"
            )

        openai_base_url = access.configured_base_url
        if openai_base_url:
            env["OPENAI_BASE_URL"] = openai_base_url
        effective_config = self._build_trace_config(openai_base_url)
        await self._upload_effective_config(environment, effective_config, remote_config_path)

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"
        setup_command += (
            f"\nrm -rf \"$CODEX_HOME/sessions\"\n"
            f"ln -s {shlex.quote(remote_sessions_dir)} \"$CODEX_HOME/sessions\"\n"
            f"touch {shlex.quote(history_path)}\n"
            f"rm -f \"$CODEX_HOME/history.jsonl\"\n"
            f"ln -s {shlex.quote(history_path)} \"$CODEX_HOME/history.jsonl\""
        )
        await self.exec_as_agent(environment, command=setup_command, env=env)

        common_codex_flags = (
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check "
            f"--model {shlex.quote(model)} "
            "--json "
            f"{cli_flags_arg}"
        )
        codex_command = (
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            f"codex exec {'resume --last ' if resume_run else ''}"
            f"{common_codex_flags}"
            f"-- {escaped_instruction}"
        )
        retry_instruction = shlex.quote(
            "Continue the same task after the transient model request failure. "
            "Preserve and inspect all existing work and side effects, then finish the "
            "implementation and verification without starting over."
        )
        resume_codex_command = (
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            f'codex exec resume "$thread_id" {common_codex_flags}'
            f"-- {retry_instruction}"
        )
        transient_pattern = (
            "429 Too Many Requests|408 Request Timeout|425 Too Early|"
            "500 Internal Server Error|502 Bad Gateway|503 Service Unavailable|"
            "504 Gateway Timeout|request (timed out|timeout)|stream (disconnected|error)|"
            "connection (reset|closed|failed)|error sending request|transport error|"
            "server overloaded|当前分组上游负载已饱和|此模型并发不足负载严重|"
            "encrypted content.*(verify|decrypt|parse)|"
            "(verify|decrypt|parse).*encrypted content"
        )
        run_command = (
            "set +e\n"
            "set -o pipefail\n"
            'started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"\n'
            f"printf '{{\"schema_version\":1,\"status\":\"running\","
            f"\"started_at\":\"%s\",\"finished_at\":null,\"exit_code\":null}}\\n' "
            f'"$started_at" > {shlex.quote(result_path)}\n'
            f": > {shlex.quote(events_path)}\n"
            f": > {shlex.quote(stderr_path)}\n"
            "attempt=0\n"
            "transient_retries=0\n"
            "thread_id=\"\"\n"
            "codex_status=1\n"
            "while true; do\n"
            "  attempt=$((attempt + 1))\n"
            f"  attempt_events={shlex.quote(events_path)}.attempt-$attempt\n"
            f"  attempt_stderr={shlex.quote(stderr_path)}.attempt-$attempt\n"
            "  : > \"$attempt_events\"\n"
            "  : > \"$attempt_stderr\"\n"
            f"  printf '\\n=== Codex attempt %s ===\\n' \"$attempt\" >> "
            f"{shlex.quote(stderr_path)}\n"
            "  if [ \"$attempt\" -eq 1 ]; then\n"
            f"    {codex_command} 2>\"$attempt_stderr\" "
            f"| tee \"$attempt_events\" >> {shlex.quote(events_path)}\n"
            "  else\n"
            f"    {resume_codex_command} 2>\"$attempt_stderr\" "
            f"| tee \"$attempt_events\" >> {shlex.quote(events_path)}\n"
            "  fi\n"
            "  codex_status=\"${PIPESTATUS[0]}\"\n"
            f"  cat \"$attempt_stderr\" >> {shlex.quote(stderr_path)}\n"
            "  if [ \"$codex_status\" -eq 0 ]; then break; fi\n"
            "  if [ -z \"$thread_id\" ]; then\n"
            f"    thread_id=$(grep -m1 '\"type\":\"thread.started\"' "
            f"{shlex.quote(events_path)} | sed -n "
            "'s/.*\"thread_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p')\n"
            "  fi\n"
            f"  if [ \"$transient_retries\" -ge {self._transient_retry_attempts} ]; then "
            "break; fi\n"
            "  if ! printf '%s' \"$thread_id\" | grep -Eq '^[A-Za-z0-9_-]+$'; then "
            "break; fi\n"
            "  failure_events=$(grep -E '\"type\":\"(error|turn.failed)\"' "
            "\"$attempt_events\" | tail -n 20)\n"
            f"  if ! printf '%s\\n' \"$failure_events\" | grep -Eiq "
            f"{shlex.quote(transient_pattern)}; then break; fi\n"
            "  transient_retries=$((transient_retries + 1))\n"
            f"  retry_delay=$(({self._transient_retry_base_delay_sec} * "
            "(1 << (transient_retries - 1))))\n"
            f"  if [ \"$retry_delay\" -gt {self._transient_retry_max_delay_sec} ]; then "
            f"retry_delay={self._transient_retry_max_delay_sec}; fi\n"
            "  retry_jitter=0\n"
            "  if [ \"$retry_delay\" -gt 0 ]; then "
            "retry_jitter=$((RANDOM % (retry_delay / 4 + 1))); fi\n"
            "  retry_sleep=$((retry_delay + retry_jitter))\n"
            f"  printf '{{\"type\":\"harness.retry\",\"attempt\":%s,"
            "\"delay_seconds\":%s,\"reason\":\"transient_model_request\"}\\n' "
            f'"$transient_retries" "$retry_sleep" >> {shlex.quote(events_path)}\n'
            "  sleep \"$retry_sleep\"\n"
            "done\n"
            'finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"\n'
            'if [ "$codex_status" -eq 0 ]; then trace_status=completed; '
            "else trace_status=failed; fi\n"
            f"printf '{{\"schema_version\":1,\"status\":\"%s\","
            f"\"started_at\":\"%s\",\"finished_at\":\"%s\","
            f"\"exit_code\":%s,\"attempts\":%s,\"transient_retries\":%s}}\\n' "
            f'"$trace_status" "$started_at" "$finished_at" "$codex_status" '
            f'"$attempt" "$transient_retries" > {shlex.quote(result_path)}\n'
            'exit "$codex_status"'
        )

        bridge_started = False
        try:
            if self._northgate_responses_bridge:
                if not openai_base_url:
                    raise ValueError("NorthGate Codex requires OPENAI_BASE_URL")
                await self._start_responses_bridge(environment, openai_base_url)
                bridge_started = True
            await self.exec_as_agent(
                environment,
                command=run_command,
                env=env,
            )
        finally:
            if bridge_started:
                try:
                    await self._stop_responses_bridge(environment)
                except Exception:
                    self.logger.warning(
                        "Failed to stop NorthGate Responses bridge",
                        exc_info=True,
                    )
            try:
                await self.exec_as_agent(
                    environment,
                    command=f'rm -rf {shlex.quote(remote_secrets_dir)} "$CODEX_HOME"',
                    env=env,
                )
            except Exception:
                self.logger.warning("Failed to remove temporary Codex credentials", exc_info=True)

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        events_path = self.logs_dir / CODEX_EVENTS_NAME
        sessions_dir = self.logs_dir / CODEX_SESSIONS_DIR_NAME
        result_path = self.logs_dir / CODEX_RESULT_NAME
        stderr_path = self.logs_dir / CODEX_STDERR_NAME
        history_path = self.logs_dir / CODEX_HISTORY_NAME
        bridge_log_path = self.logs_dir / CODEX_BRIDGE_LOG_NAME

        if not any(
            path.exists()
            for path in (
                events_path,
                sessions_dir,
                result_path,
                stderr_path,
                history_path,
                bridge_log_path,
            )
        ):
            self.logger.warning("No Codex trace artifacts were downloaded")
            return

        event_summary = _event_stream_summary(events_path)
        session_files = sorted(sessions_dir.rglob("*.jsonl")) if sessions_dir.is_dir() else []
        usage = event_summary.get("usage") or _rollout_usage(session_files) or {}

        context.n_input_tokens = int(usage.get("input_tokens") or 0)
        context.n_cache_tokens = int(usage.get("cached_input_tokens") or 0)
        context.n_output_tokens = int(usage.get("output_tokens") or 0)

        existing_result: dict[str, Any] = {}
        if result_path.is_file():
            try:
                candidate = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(candidate, dict):
                    existing_result = candidate
            except (OSError, UnicodeError, json.JSONDecodeError):
                self.logger.warning("Could not parse the remote Codex result", exc_info=True)

        process_status = existing_result.get("status")
        trace_status = event_summary["trace_status"]
        exit_code = existing_result.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            status = "failed"
        elif process_status in {"completed", "failed"}:
            status = process_status
        elif trace_status in {"completed", "failed"}:
            status = trace_status
        else:
            status = "incomplete"

        rollout_records = [_jsonl_record(path, self.logs_dir) for path in session_files]
        trace_complete = bool(
            event_summary["trace_complete"]
            and rollout_records
            and all(record["invalid_json_lines"] == 0 for record in rollout_records)
        )
        result = {
            **existing_result,
            "schema_version": 1,
            "status": status,
            "trace_status": trace_status,
            "trace_complete": trace_complete,
            "trajectory_generated": False,
            "model": self._runtime_model_name(),
            "upstream_model": (
                self._northgate_upstream_model_name or self._runtime_model_name()
            ),
            "thread_ids": event_summary["thread_ids"],
            "final_response": event_summary["final_response"],
            "usage": usage or None,
        }
        _atomic_write_json(result_path, result)

        events_record = _jsonl_record(events_path, self.logs_dir) if events_path.is_file() else None
        manifest = {
            "schema_version": 1,
            "source_of_truth": [
                CODEX_EVENTS_NAME,
                CODEX_HISTORY_NAME,
                f"{CODEX_SESSIONS_DIR_NAME}/**/*.jsonl",
            ],
            "trajectory_generated": False,
            "trace_complete": trace_complete,
            "models": {
                "advertised": self._runtime_model_name(),
                "upstream": (
                    self._northgate_upstream_model_name or self._runtime_model_name()
                ),
            },
            "event_summary": event_summary,
            "artifacts": {
                "events": events_record,
                "history": (
                    _jsonl_record(history_path, self.logs_dir)
                    if history_path.is_file()
                    else None
                ),
                "stderr": _file_record(stderr_path, self.logs_dir),
                "bridge_log": _file_record(bridge_log_path, self.logs_dir),
                "result": _file_record(result_path, self.logs_dir),
                "rollouts": rollout_records,
            },
        }
        _atomic_write_json(self.logs_dir / CODEX_TRACE_MANIFEST_NAME, manifest)


CodexTrace = TraceSafeCodexAgent
