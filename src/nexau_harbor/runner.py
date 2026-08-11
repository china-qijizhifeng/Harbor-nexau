from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


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
        raise ConfigError(
            f"Invalid configuration type in {config_path}; expected mapping"
        )
    return raw_config


def _has_in_memory_tracer(tracer_configs: list[Any]) -> bool:
    import_path = "nexau.archs.tracer.adapters.in_memory:InMemoryTracer"
    for entry in tracer_configs:
        if isinstance(entry, str) and entry.strip() == import_path:
            return True
        if isinstance(entry, dict) and entry.get("import") == import_path:
            return True
    return False


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
            f"Unsupported LLM_REASONING_EFFORT {raw_effort!r}; expected one of: "
            f"{supported}"
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


def _build_agent(config_path: Path) -> Any:
    from nexau.archs.main_sub.agent import Agent
    from nexau.archs.main_sub.config.config import AgentConfigBuilder
    from nexau.archs.main_sub.config.schema import normalize_agent_config_dict

    _ensure_config_dir_on_sys_path(config_path)
    normalized_config = normalize_agent_config_dict(_load_raw_config(config_path))
    normalized_config = _normalize_reasoning_config(normalized_config)

    tracer_configs = list(normalized_config.get("tracers") or [])
    if not _has_in_memory_tracer(tracer_configs):
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
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _dump_history(agent: Any, log_dir: Path) -> None:
    history = [_jsonable(message) for message in list(agent.history)]
    _write_json(log_dir / "nexau_history.json", history)


def _dump_in_memory_traces(agent: Any, log_dir: Path) -> None:
    tracer = _get_in_memory_tracer(agent)
    if tracer is None:
        return
    try:
        tracer.flush()
    except Exception:
        pass
    _write_json(log_dir / "nexau_in_memory_tracer.json", tracer.dump_traces())


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
    try:
        agent = _build_agent(config_path)
        response = agent.run(
            message=args.query,
            context=_build_runtime_context(agent),
        )
        _dump_history(agent, log_dir)
        _dump_in_memory_traces(agent, log_dir)
        _dump_result(response, log_dir)
        return 0
    except Exception:
        if agent is not None:
            try:
                _dump_history(agent, log_dir)
            except Exception:
                pass
            try:
                _dump_in_memory_traces(agent, log_dir)
            except Exception:
                pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
