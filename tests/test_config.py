from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from nexau.core.messages import Message, Role

from nexau_harbor.agent import NexAUAgent
from nexau_harbor.runner import (
    JsonlFileTracer,
    _build_agent,
    _iter_tracers,
    _normalize_reasoning_config,
)

_CONFIG_ROOT = Path(__file__).parents[1] / "src" / "nexau_harbor" / "configs"


def test_default_config_references_existing_assets() -> None:
    config_path = _CONFIG_ROOT / "gdpval" / "code_agent.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["system_prompt"] == "./systemprompt_tb2.md"
    assert (config_path.parent / "systemprompt_tb2.md").is_file()
    assert config["tracers"] == [{"import": "gdpval_nexau_tracer:ThreadSafeInMemoryTracer"}]
    assert (config_path.parent / "gdpval_nexau_tracer.py").is_file()
    assert len(config["tools"]) == 6
    for tool in config["tools"]:
        assert (config_path.parent / tool["yaml_path"]).is_file()


def test_terminal_bench_config_references_only_tb_assets() -> None:
    config_path = _CONFIG_ROOT / "terminal_bench" / "code_agent.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["name"] == "nexau_terminal_bench_agent"
    assert config["max_iterations"] == 500
    assert config["max_context_tokens"] == "${env.LLM_CONTEXT_WINDOW_TOKENS}"
    assert config["system_prompt"] == "./systemprompt_tb2.md"
    assert (config_path.parent / "systemprompt_tb2.md").is_file()
    assert [tool["name"] for tool in config["tools"]] == [
        "read_file",
        "replace",
        "run_shell_command",
        "write_todo",
    ]
    assert config["stop_tools"] == []
    assert config["skills"] == []
    assert config["tracers"] == []
    assert config["llm_config"]["additional_drop_params"] == [
        "max_tokens",
        "max_output_tokens",
    ]
    assert len(config["middlewares"]) == 2
    assert config["middlewares"][1]["params"] == {
        "max_context_tokens": "${env.LLM_CONTEXT_WINDOW_TOKENS}",
        "auto_compact": True,
        "threshold": 0.8,
        "compaction_strategy": "llm_summary",
        "keep_iterations": 4,
        "retry_attempts": 3,
    }
    assert "gdpval" not in config_path.read_text().lower()
    for tool in config["tools"]:
        assert (config_path.parent / tool["yaml_path"]).is_file()


def test_nexau_builds_terminal_bench_config(monkeypatch, tmp_path: Path) -> None:
    config_path = _CONFIG_ROOT / "terminal_bench" / "code_agent.yaml"
    integration = NexAUAgent(
        logs_dir=tmp_path,
        model_name="openai/nex-agi/Nex-N2-Pro",
        config_path=config_path,
        extra_env={
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_KIND": "openai_responses",
            "LLM_CONTEXT_WINDOW_TOKENS": "200000",
        },
    )
    for name, value in integration._runtime_env().items():
        monkeypatch.setenv(name, value)

    agent = _build_agent(config_path)

    assert agent.config.name == "nexau_terminal_bench_agent"
    assert [tool.name for tool in agent.config.tools] == [
        "read_file",
        "replace",
        "run_shell_command",
        "write_todo",
    ]
    assert agent.config.max_context_tokens == 200000
    assert agent.config.max_iterations == 500
    assert [type(item).__name__ for item in agent.config.middlewares] == [
        "LongToolOutputMiddleware",
        "ContextCompactionMiddleware",
    ]
    compaction = agent.config.middlewares[1]
    assert compaction.trigger_strategy.threshold == 0.8
    assert compaction.compaction_strategy.__class__.__name__ == "SlidingWindowCompaction"
    assert compaction.compaction_strategy.keep_iterations == 4
    assert compaction.compaction_strategy.retry_attempts == 3
    assert compaction.compaction_strategy.summary_llm_config.model == "nex-agi/Nex-N2-Pro"
    assert compaction.compaction_strategy.summary_llm_config.api_type == "openai_responses"
    assert agent.config.resolved_tracer.__class__.__name__ == "InMemoryTracer"
    assert agent.config.llm_config.stream is False
    assert agent.config.llm_config.temperature == 0.6
    assert agent.config.llm_config.max_retries == 20
    assert agent.config.llm_config.additional_drop_params == (
        "max_tokens",
        "max_output_tokens",
    )
    request_params = agent.config.llm_config.to_openai_params()
    assert "max_tokens" not in request_params
    assert "max_output_tokens" not in request_params
    assert agent.config.llm_config.apply_param_drops(
        {"model": "Nex-N2-Pro", "max_tokens": 32000, "max_output_tokens": 32000}
    ) == {"model": "Nex-N2-Pro"}


def test_llm_summary_keeps_a_user_message_for_single_turn_agents(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = _CONFIG_ROOT / "terminal_bench" / "code_agent.yaml"
    integration = NexAUAgent(
        logs_dir=tmp_path,
        model_name="openai/nex-agi/Nex-N2-Pro",
        config_path=config_path,
        extra_env={
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_KIND": "openai_responses",
            "LLM_CONTEXT_WINDOW_TOKENS": "200000",
        },
    )
    for name, value in integration._runtime_env().items():
        monkeypatch.setenv(name, value)

    agent = _build_agent(config_path)
    strategy = agent.config.middlewares[1].compaction_strategy
    summary_inputs: list[list[Message]] = []

    def summarize(messages: list[Message]) -> str:
        summary_inputs.append(messages)
        return "Original request and completed work summary"

    monkeypatch.setattr(strategy, "_generate_summary_safe", summarize)
    messages = [Message.user("original terminal-bench request")]
    messages.extend(Message.assistant(f"tool iteration {index}") for index in range(6))

    compacted = strategy.compact(messages)

    assert summary_inputs
    assert summary_inputs[0][0].role == Role.USER
    assert summary_inputs[0][0].get_text_content() == "original terminal-bench request"
    assert compacted[0].role == Role.USER
    assert "Original request and completed work summary" in compacted[0].get_text_content()
    assert [message.get_text_content() for message in compacted[1:]] == [
        "tool iteration 2",
        "tool iteration 3",
        "tool iteration 4",
        "tool iteration 5",
    ]


def test_harbor_run_replaces_in_memory_tracer_with_jsonl(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = _CONFIG_ROOT / "terminal_bench" / "code_agent.yaml"
    integration = NexAUAgent(
        logs_dir=tmp_path,
        model_name="openai/nex-agi/Nex-N2-Pro",
        config_path=config_path,
        extra_env={
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_KIND": "openai_responses",
            "LLM_CONTEXT_WINDOW_TOKENS": "200000",
        },
    )
    for name, value in integration._runtime_env().items():
        monkeypatch.setenv(name, value)

    event_path = tmp_path / "nexau_trace_events.jsonl"
    agent = _build_agent(config_path, event_path)
    tracers = _iter_tracers(agent.config.resolved_tracer)

    assert len(tracers) == 1
    assert isinstance(tracers[0], JsonlFileTracer)
    assert event_path.is_file()


def test_nexau_builds_migrated_config(monkeypatch, tmp_path: Path) -> None:
    integration = NexAUAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5",
        extra_env={
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://example.test/v1",
        },
    )
    for name, value in integration._runtime_env().items():
        monkeypatch.setenv(name, value)

    agent = _build_agent(integration._config_path)
    assert agent.config.name == "nexau_harbor_agent"
    assert len(agent.config.tools) == 6
    assert agent.config.llm_config.model == "gpt-5"
    assert agent.config.llm_config.extra_params["reasoning"] == {
        "effort": "medium",
        "summary": "detailed",
    }
    assert agent.config.resolved_tracer.__class__.__name__ == "ThreadSafeInMemoryTracer"
    assert "harbor_reasoning_effort" not in agent.config.llm_config.extra_params


@pytest.mark.parametrize(
    ("api_type", "model", "effort", "expected"),
    [
        (
            "openai_responses",
            "gpt-5",
            "medium",
            {"reasoning": {"effort": "medium", "summary": "detailed"}},
        ),
        (
            "openai_chat_completion",
            "o3",
            "high",
            {"reasoning_effort": "high"},
        ),
        ("openai_chat_completion", "gpt-4.1", "medium", {}),
        (
            "anthropic_chat_completion",
            "claude-sonnet-4-6",
            "medium",
            {"output_config": {"effort": "medium"}},
        ),
        ("anthropic_chat_completion", "claude-haiku-4-5", "medium", {}),
        (
            "gemini_rest",
            "gemini-3.5-flash",
            "xhigh",
            {"thinkingConfig": {"includeThoughts": True, "thinkingLevel": "high"}},
        ),
        (
            "gemini_rest",
            "gemini-2.5-pro",
            "medium",
            {"thinkingConfig": {"includeThoughts": True}},
        ),
    ],
)
def test_reasoning_effort_uses_provider_native_schema(
    api_type: str,
    model: str,
    effort: str,
    expected: dict[str, object],
) -> None:
    config = {
        "llm_config": {
            "api_type": api_type,
            "model": model,
            "harbor_reasoning_effort": effort,
        }
    }

    normalized = _normalize_reasoning_config(config)
    llm_config = normalized["llm_config"]

    assert "harbor_reasoning_effort" not in llm_config
    for key, value in expected.items():
        assert llm_config[key] == value
    assert set(llm_config) == {"api_type", "model", *expected}


def test_native_reasoning_config_takes_precedence() -> None:
    config = {
        "llm_config": {
            "api_type": "openai_responses",
            "model": "gpt-5",
            "harbor_reasoning_effort": "medium",
            "reasoning": {"effort": "high", "summary": "concise"},
        }
    }

    normalized = _normalize_reasoning_config(config)

    assert normalized["llm_config"]["reasoning"] == {
        "effort": "high",
        "summary": "concise",
    }


def test_invalid_reasoning_effort_fails_before_model_call() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM_REASONING_EFFORT"):
        _normalize_reasoning_config(
            {
                "llm_config": {
                    "api_type": "openai_responses",
                    "model": "gpt-5",
                    "harbor_reasoning_effort": "extreme",
                }
            }
        )
