from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from nexau.archs.tracer.core import BaseTracer, Span, SpanType

_DEFAULT_TRACE_CHECKPOINT_SEC = 30.0
_TRACE_CHECKPOINT_ENV = "NEXAU_TRACE_CHECKPOINT_SEC"
_TRACE_EVENTS_NAME = "nexau_trace_events.jsonl"


def _ensure_config_dir_on_sys_path(config_path: Path) -> None:
    config_dir = str(config_path.resolve().parent)
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)


def _load_raw_config(config_path: Path) -> dict[str, Any]:
    from nexau.archs.main_sub.config import ConfigError
    from nexau.archs.main_sub.utils import load_yaml_with_vars

    raw_config = load_yaml_with_vars(str(config_path))
    if not raw_config:
        raise ConfigError(f"Empty or invalid configuration file: {config_path}")
    if not isinstance(raw_config, dict):
        raise ConfigError(f"Invalid configuration type in {config_path}; expected mapping")
    return raw_config


def _is_in_memory_tracer_config(entry: Any) -> bool:
    import_paths = {
        "nexau.archs.tracer.adapters.in_memory:InMemoryTracer",
        "gdpval_nexau_tracer:ThreadSafeInMemoryTracer",
    }
    if isinstance(entry, str):
        return entry.strip() in import_paths
    return isinstance(entry, dict) and entry.get("import") in import_paths


_REASONING_EFFORTS = {
    "default",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}


def _openai_chat_supports_reasoning(model: str) -> bool:
    normalized = model.lower().rsplit("/", 1)[-1]
    return normalized.startswith(("gpt-5", "o1", "o3", "o4", "codex-mini"))


def _anthropic_supports_effort(model: str) -> bool:
    normalized = model.lower().rsplit("/", 1)[-1]
    if "haiku" in normalized:
        return False
    if any(family in normalized for family in ("fable", "mythos")):
        return True

    match = re.search(r"claude-(opus|sonnet)-(\d+)[.-](\d+)", normalized)
    if match is None:
        # Keep aliases and future naming schemes usable; only known unsupported
        # versioned models are filtered below.
        return True

    family, major_text, minor_text = match.groups()
    major, minor = int(major_text), int(minor_text)
    if major >= 5:
        return True
    if major != 4:
        return False
    return minor >= (5 if family == "opus" else 6)


def _gemini_major_version(model: str) -> int | None:
    match = re.search(r"(?:^|/)gemini-(\d+)(?:[.-]|$)", model.lower())
    return int(match.group(1)) if match else None


def _normalize_reasoning_config(config: dict[str, Any]) -> dict[str, Any]:
    """Translate Harbor's neutral effort setting into provider-native fields."""

    normalized = dict(config)
    raw_llm_config = normalized.get("llm_config")
    if not isinstance(raw_llm_config, dict):
        return normalized

    llm_config = dict(raw_llm_config)
    raw_effort = llm_config.pop("harbor_reasoning_effort", None)
    normalized["llm_config"] = llm_config
    if raw_effort is None:
        return normalized

    effort = str(raw_effort).strip().lower()
    if effort not in _REASONING_EFFORTS:
        supported = ", ".join(sorted(_REASONING_EFFORTS))
        raise ValueError(
            f"Unsupported LLM_REASONING_EFFORT {raw_effort!r}; expected one of: {supported}"
        )
    if effort == "default":
        return normalized

    api_type = str(llm_config.get("api_type") or "openai_chat_completion").lower()
    model = str(llm_config.get("model") or "")

    if api_type == "openai_responses":
        reasoning = dict(llm_config.get("reasoning") or {})
        reasoning.setdefault("effort", "xhigh" if effort == "max" else effort)
        reasoning.setdefault("summary", "detailed")
        llm_config["reasoning"] = reasoning
    elif api_type == "openai_chat_completion" and _openai_chat_supports_reasoning(model):
        llm_config.setdefault("reasoning_effort", "xhigh" if effort == "max" else effort)
    elif api_type == "anthropic_chat_completion" and _anthropic_supports_effort(model):
        if effort != "none":
            output_config = dict(llm_config.get("output_config") or {})
            output_config.setdefault("effort", "low" if effort == "minimal" else effort)
            llm_config["output_config"] = output_config
    elif api_type == "gemini_rest":
        thinking_config = dict(llm_config.get("thinkingConfig") or {})
        thinking_config.setdefault("includeThoughts", True)
        major_version = _gemini_major_version(model)
        if major_version is not None and major_version >= 3:
            gemini_effort = {
                "none": "minimal",
                "xhigh": "high",
                "max": "high",
            }.get(effort, effort)
            thinking_config.setdefault("thinkingLevel", gemini_effort)
        llm_config["thinkingConfig"] = thinking_config

    return normalized


def _build_agent(config_path: Path, trace_path: Path | None = None) -> Any:
    from nexau.archs.main_sub.agent import Agent
    from nexau.archs.main_sub.config.config import AgentConfigBuilder
    from nexau.archs.main_sub.config.schema import normalize_agent_config_dict

    _ensure_config_dir_on_sys_path(config_path)
    normalized_config = normalize_agent_config_dict(_load_raw_config(config_path))
    normalized_config = _normalize_reasoning_config(normalized_config)

    tracer_configs = list(normalized_config.get("tracers") or [])
    if trace_path is not None:
        tracer_configs = [
            entry for entry in tracer_configs if not _is_in_memory_tracer_config(entry)
        ]
        tracer_configs.insert(0, JsonlFileTracer(trace_path))
    elif not any(_is_in_memory_tracer_config(entry) for entry in tracer_configs):
        tracer_configs.insert(
            0,
            {"import": "nexau.archs.tracer.adapters.in_memory:InMemoryTracer"},
        )
    normalized_config["tracers"] = tracer_configs

    builder = AgentConfigBuilder(normalized_config, config_path.parent)
    agent_config = (
        builder.build_core_properties()
        .build_llm_config()
        .build_mcp_servers()
        .build_hooks()
        .build_tracers()
        .build_tools()
        .build_sub_agents()
        .build_skills()
        .build_system_prompt_path()
        .build_sandbox()
        .get_agent_config()
    )
    return Agent(config=agent_config)


def _iter_tracers(tracer: Any) -> list[Any]:
    from nexau.archs.tracer.composite import CompositeTracer

    if tracer is None:
        return []
    if isinstance(tracer, CompositeTracer):
        result: list[Any] = []
        for child in tracer.tracers:
            result.extend(_iter_tracers(child))
        return result
    return [tracer]


def _get_in_memory_tracer(agent: Any) -> Any | None:
    from nexau.archs.tracer.adapters.in_memory import InMemoryTracer

    tracer = getattr(agent.config, "resolved_tracer", None)
    if tracer is None and getattr(agent, "global_storage", None) is not None:
        try:
            tracer = agent.global_storage.get("tracer")
        except Exception:
            tracer = None

    for candidate in _iter_tracers(tracer):
        if isinstance(candidate, InMemoryTracer):
            return candidate
    return None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    """Atomically replace a JSON artifact so interrupted writes stay readable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class JsonlFileTracer(BaseTracer):
    """Append span lifecycle events without retaining completed trace payloads."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        self._sequence = 0

    def _append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            payload = {
                "schema_version": 1,
                "sequence": self._sequence,
                **event,
            }
            rendered = json.dumps(_jsonable(payload), ensure_ascii=False) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        span_id = str(uuid4())
        parent_id = None
        if parent_span is not None:
            parent_id = str(parent_span.vendor_obj) if parent_span.vendor_obj else parent_span.id
        span = Span(
            id=span_id,
            name=name,
            type=span_type,
            parent_id=parent_id,
            inputs={},
            attributes={},
            vendor_obj=span_id,
        )
        self._append(
            {
                "event": "start",
                "span": {
                    "id": span.id,
                    "name": span.name,
                    "type": span.type.value,
                    "parent_id": span.parent_id,
                    "start_time": span.start_time,
                    "inputs": inputs or {},
                    "attributes": attributes or {},
                },
            }
        )
        return span

    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        span_id = str(span.vendor_obj) if span.vendor_obj else span.id
        normalized_outputs: dict[str, Any] = {}
        if outputs is not None:
            normalized_outputs = outputs if isinstance(outputs, dict) else {"result": outputs}
        self._append(
            {
                "event": "end",
                "span_id": span_id,
                "end_time": datetime.now().timestamp(),
                "outputs": normalized_outputs,
                "error": str(error) if error is not None else None,
                "attributes": attributes or {},
            }
        )

    def flush(self) -> None:
        return

    def shutdown(self) -> None:
        return


def _dump_history(agent: Any, log_dir: Path) -> None:
    history = [_jsonable(message) for message in list(agent.history)]
    _write_json(log_dir / "nexau_history.json", history)


def _dump_full_trace(agent: Any, log_dir: Path) -> bool:
    """Persist NexAU's append-only, pre-compaction conversation trace."""

    full_trace = getattr(agent, "full_trace", None)
    if callable(full_trace):
        full_trace = full_trace()
    if not full_trace:
        return False
    _write_json(
        log_dir / "nexau_full_trace.json",
        [_jsonable(message) for message in list(full_trace)],
    )
    return True


def _dump_in_memory_traces(agent: Any, log_dir: Path) -> bool:
    tracer = _get_in_memory_tracer(agent)
    if tracer is None:
        return False
    try:
        tracer.flush()
    except Exception:
        pass
    _write_json(log_dir / "nexau_in_memory_tracer.json", tracer.dump_traces())
    return True


def _get_jsonl_file_tracer(agent: Any) -> JsonlFileTracer | None:
    tracer = getattr(agent.config, "resolved_tracer", None)
    if tracer is None and getattr(agent, "global_storage", None) is not None:
        try:
            tracer = agent.global_storage.get("tracer")
        except Exception:
            tracer = None
    for candidate in _iter_tracers(tracer):
        if isinstance(candidate, JsonlFileTracer):
            return candidate
    return None


def _trace_checkpoint_interval() -> float:
    raw = os.environ.get(
        _TRACE_CHECKPOINT_ENV,
        str(_DEFAULT_TRACE_CHECKPOINT_SEC),
    ).strip()
    try:
        interval = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_TRACE_CHECKPOINT_ENV} must be a non-negative number, got {raw!r}"
        ) from exc
    if not math.isfinite(interval) or interval < 0:
        raise ValueError(
            f"{_TRACE_CHECKPOINT_ENV} must be a finite non-negative number, got {raw!r}"
        )
    return interval


def _write_trace_checkpoint_state(
    log_dir: Path,
    *,
    status: str,
    interval_sec: float,
    written_files: list[str] | None = None,
    errors: list[str] | None = None,
) -> None:
    _write_json(
        log_dir / "nexau_trace_checkpoint.json",
        {
            "schema_version": 1,
            "status": status,
            "checkpointed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "interval_sec": interval_sec,
            "written_files": written_files or [],
            "errors": errors or [],
        },
    )


def _checkpoint_trace_artifacts(
    agent: Any,
    log_dir: Path,
    *,
    interval_sec: float,
) -> None:
    """Best-effort running checkpoint; each artifact is independently atomic."""

    written_files: list[str] = []
    errors: list[str] = []
    file_tracer = _get_jsonl_file_tracer(agent)
    if file_tracer is not None:
        try:
            file_tracer.flush()
            if file_tracer.path.exists():
                written_files.append(file_tracer.path.name)
        except Exception as exc:
            errors.append(f"{file_tracer.path.name}: {type(exc).__name__}: {exc}")
        dumpers: tuple[tuple[str, Any], ...] = ()
    else:
        dumpers = (
            ("nexau_history.json", _dump_history),
            ("nexau_full_trace.json", _dump_full_trace),
            ("nexau_in_memory_tracer.json", _dump_in_memory_traces),
        )
    for filename, dumper in dumpers:
        try:
            if dumper(agent, log_dir) is not False:
                written_files.append(filename)
        except Exception as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")

    _write_trace_checkpoint_state(
        log_dir,
        status="running",
        interval_sec=interval_sec,
        written_files=written_files,
        errors=errors,
    )


class _PeriodicTraceCheckpointer:
    def __init__(self, agent: Any, log_dir: Path, interval_sec: float) -> None:
        self._agent = agent
        self._log_dir = log_dir
        self._interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _checkpoint(self) -> None:
        try:
            _checkpoint_trace_artifacts(
                self._agent,
                self._log_dir,
                interval_sec=self._interval_sec,
            )
        except Exception as exc:
            print(
                f"[nexau-harbor] Trace checkpoint failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_sec):
            self._checkpoint()

    def start(self) -> None:
        if self._interval_sec <= 0 or self._thread is not None:
            return
        self._checkpoint()
        self._thread = threading.Thread(
            target=self._run,
            name="nexau-trace-checkpointer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


def _normalize_response(response: Any) -> tuple[Any, Any | None]:
    if isinstance(response, tuple) and len(response) == 2:
        return response[0], response[1]
    return response, None


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _dump_result(response: Any, log_dir: Path) -> None:
    response_text, state = _normalize_response(response)
    _write_json(
        log_dir / "nexau_result.json",
        {
            "response": response_text,
            "parsed_response": _maybe_parse_json(response_text),
            "state": state,
        },
    )


def _build_runtime_context(agent: Any) -> dict[str, Any]:
    sandbox = agent.sandbox_manager.instance
    working_directory = str(sandbox.work_dir if sandbox else os.getcwd())
    current_time = datetime.now().astimezone().isoformat(timespec="seconds")
    username = os.getenv("USER", "user")
    return {
        "date": current_time,
        "username": username,
        "working_directory": working_directory,
        "env_content": {
            "date": current_time,
            "username": username,
            "working_directory": working_directory,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NexAU inside a Harbor task.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--log-dir-path", required=True)
    parser.add_argument("--query", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config_path).resolve()
    log_dir = Path(args.log_dir_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    agent: Any | None = None
    checkpointer: _PeriodicTraceCheckpointer | None = None
    checkpoint_interval = _trace_checkpoint_interval()
    try:
        agent = _build_agent(config_path, log_dir / _TRACE_EVENTS_NAME)
        checkpointer = _PeriodicTraceCheckpointer(
            agent,
            log_dir,
            checkpoint_interval,
        )
        checkpointer.start()
        try:
            response = agent.run(
                message=args.query,
                context=_build_runtime_context(agent),
            )
        finally:
            checkpointer.stop()
        _dump_history(agent, log_dir)
        _dump_full_trace(agent, log_dir)
        _dump_in_memory_traces(agent, log_dir)
        _dump_result(response, log_dir)
        _write_trace_checkpoint_state(
            log_dir,
            status="complete",
            interval_sec=checkpoint_interval,
            written_files=[
                path.name
                for path in (
                    log_dir / "nexau_history.json",
                    log_dir / "nexau_full_trace.json",
                    log_dir / "nexau_in_memory_tracer.json",
                    log_dir / _TRACE_EVENTS_NAME,
                    log_dir / "nexau_result.json",
                )
                if path.exists()
            ],
        )
        return 0
    except Exception:
        if checkpointer is not None:
            checkpointer.stop()
        if agent is not None:
            try:
                _dump_history(agent, log_dir)
            except Exception:
                pass
            try:
                _dump_full_trace(agent, log_dir)
            except Exception:
                pass
            try:
                _dump_in_memory_traces(agent, log_dir)
            except Exception:
                pass
        try:
            _write_trace_checkpoint_state(
                log_dir,
                status="failed",
                interval_sec=checkpoint_interval,
                written_files=[
                    path.name
                    for path in (
                        log_dir / "nexau_history.json",
                        log_dir / "nexau_full_trace.json",
                        log_dir / "nexau_in_memory_tracer.json",
                        log_dir / _TRACE_EVENTS_NAME,
                    )
                    if path.exists()
                ],
            )
        except Exception:
            pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
