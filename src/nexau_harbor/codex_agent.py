"""Trace-preserving Codex harness for Harbor runs."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections import Counter
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from nexau_harbor.constants import NEXAU_DETACHED_EXEC_ENV, REMOTE_LOG_DIR
from nexau_harbor.trace_artifacts import (
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
            elif event_type in {"turn.failed", "error"}:
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

    SUPPORTS_ATIF = False
    SUPPORTS_LOAD_ATIF_TRAJECTORY = False

    @staticmethod
    @override
    def name() -> str:
        return "codex"

    def _runtime_model_name(self) -> str:
        model = self.model_name
        if not model:
            raise ValueError("Model name is required")
        provider = self.model_connection.provider
        prefix, separator, remainder = model.partition("/")
        if separator and provider and prefix.lower() == provider.lower():
            return remainder
        return model

    def _build_trace_config(self, openai_base_url: str | None) -> dict[str, Any]:
        config = self._build_effective_config(openai_base_url)
        history = config.setdefault("history", {})
        if not isinstance(history, dict):
            raise ValueError("Invalid Codex config: history must be a TOML table")
        history["persistence"] = "save-all"
        history.pop("max_bytes", None)
        config["hide_agent_reasoning"] = False
        return config

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

        codex_command = (
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            f"codex exec {'resume --last ' if resume_run else ''}"
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check "
            f"--model {shlex.quote(model)} "
            "--json "
            "--enable unified_exec "
            f"{cli_flags_arg}"
            f"-- {escaped_instruction}"
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
            f"{codex_command} 2>{shlex.quote(stderr_path)} "
            f"| tee {shlex.quote(events_path)} >/dev/null\n"
            'codex_status="${PIPESTATUS[0]}"\n'
            'finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"\n'
            'if [ "$codex_status" -eq 0 ]; then trace_status=completed; '
            "else trace_status=failed; fi\n"
            f"printf '{{\"schema_version\":1,\"status\":\"%s\","
            f"\"started_at\":\"%s\",\"finished_at\":\"%s\","
            f"\"exit_code\":%s}}\\n' \"$trace_status\" \"$started_at\" "
            f'"$finished_at" "$codex_status" > {shlex.quote(result_path)}\n'
            'exit "$codex_status"'
        )

        execution_env = dict(env)
        if getattr(environment, "SUPPORTS_DETACHED_AGENT_EXEC", False):
            execution_env[NEXAU_DETACHED_EXEC_ENV] = "1"
        try:
            await self.exec_as_agent(
                environment,
                command=run_command,
                env=execution_env,
            )
        finally:
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

        if not any(
            path.exists()
            for path in (events_path, sessions_dir, result_path, stderr_path, history_path)
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
            "event_summary": event_summary,
            "artifacts": {
                "events": events_record,
                "history": (
                    _jsonl_record(history_path, self.logs_dir)
                    if history_path.is_file()
                    else None
                ),
                "stderr": _file_record(stderr_path, self.logs_dir),
                "result": _file_record(result_path, self.logs_dir),
                "rollouts": rollout_records,
            },
        }
        _atomic_write_json(self.logs_dir / CODEX_TRACE_MANIFEST_NAME, manifest)


CodexTrace = TraceSafeCodexAgent
