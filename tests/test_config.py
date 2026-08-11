from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nexau_harbor.agent import NexAUAgent
from nexau_harbor.runner import _build_agent, _normalize_reasoning_config


def test_default_config_references_existing_assets() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "src"
        / "nexau_harbor"
        / "configs"
        / "gdpval"
        / "code_agent.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["system_prompt"] == "./systemprompt_tb2.md"
    assert (config_path.parent / "systemprompt_tb2.md").is_file()
    assert len(config["tools"]) == 6
    for tool in config["tools"]:
        assert (config_path.parent / tool["yaml_path"]).is_file()


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
