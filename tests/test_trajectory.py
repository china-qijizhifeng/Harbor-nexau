from __future__ import annotations

import json
import logging
from pathlib import Path

from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Trajectory

from nexau_harbor.trajectory import collect_metrics, populate_context_from_logs


def test_nexau_history_is_converted_to_valid_atif(tmp_path: Path) -> None:
    logs_dir = tmp_path / "agent"
    logs_dir.mkdir()
    history = [
        {
            "role": "system",
            "created_at": "2026-08-11T12:00:00+00:00",
            "content": [{"type": "text", "text": "system prompt"}],
        },
        {
            "role": "user",
            "created_at": "2026-08-11T12:00:01+00:00",
            "content": [{"type": "text", "text": "solve it"}],
        },
        {
            "role": "assistant",
            "created_at": "2026-08-11T12:00:02+00:00",
            "content": [
                {"type": "reasoning", "text": "inspect first"},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "run_shell",
                    "input": {"command": "pwd"},
                },
            ],
        },
        {
            "role": "tool",
            "created_at": "2026-08-11T12:00:03+00:00",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": [{"type": "text", "text": "/app"}],
                }
            ],
        },
        {
            "role": "assistant",
            "created_at": "2026-08-11T12:00:04+00:00",
            "content": [{"type": "text", "text": "finished"}],
        },
    ]
    traces = [
        {
            "type": "ROOT",
            "children": [
                {
                    "type": "LLM",
                    "outputs": {
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "cached_tokens": 10,
                            "cost_usd": 0.01,
                        }
                    },
                },
                {
                    "type": "LLM",
                    "outputs": {
                        "usage": {
                            "prompt_tokens": 120,
                            "completion_tokens": 5,
                            "cost": 0.002,
                        }
                    },
                },
            ],
        }
    ]
    (logs_dir / "nexau_history.json").write_text(json.dumps(history))
    (logs_dir / "nexau_in_memory_tracer.json").write_text(json.dumps(traces))

    context = AgentContext()
    populate_context_from_logs(
        logs_dir=logs_dir,
        context=context,
        agent_name="nexau",
        agent_version="0.4.0",
        model_name="openai/gpt-5",
        reasoning_effort="medium",
        logger=logging.getLogger("test"),
    )

    trajectory = Trajectory.model_validate_json(
        (logs_dir / "trajectory.json").read_text()
    )
    assert trajectory.schema_version == "ATIF-v1.7"
    assert [step.source for step in trajectory.steps] == [
        "system",
        "user",
        "agent",
        "agent",
    ]
    assert trajectory.steps[2].tool_calls[0].tool_call_id == "call-1"
    assert trajectory.steps[2].observation.results[0].content == "/app"
    assert trajectory.final_metrics.total_steps == 4
    assert context.n_input_tokens == 220
    assert context.n_output_tokens == 25
    assert context.n_cache_tokens == 10
    assert context.cost_usd == 0.012


def test_invalid_history_does_not_create_trajectory(tmp_path: Path) -> None:
    logs_dir = tmp_path / "agent"
    logs_dir.mkdir()
    (logs_dir / "nexau_history.json").write_text("[]")

    populate_context_from_logs(
        logs_dir=logs_dir,
        context=AgentContext(),
        agent_name="nexau",
        agent_version="0.4.0",
        model_name=None,
        reasoning_effort=None,
        logger=logging.getLogger("test"),
    )

    assert not (logs_dir / "trajectory.json").exists()


def test_uncompacted_full_trace_is_preferred_over_final_history(
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "agent"
    logs_dir.mkdir()
    compacted_history = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "compacted summary"}],
        }
    ]
    full_trace = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "original task"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "original answer"}],
        },
    ]
    (logs_dir / "nexau_history.json").write_text(json.dumps(compacted_history))
    (logs_dir / "nexau_full_trace.json").write_text(json.dumps(full_trace))

    populate_context_from_logs(
        logs_dir=logs_dir,
        context=AgentContext(),
        agent_name="nexau",
        agent_version="0.4.0",
        model_name="openai/gpt-5",
        reasoning_effort="medium",
        logger=logging.getLogger("test"),
    )

    trajectory = Trajectory.model_validate_json(
        (logs_dir / "trajectory.json").read_text()
    )
    assert [step.message for step in trajectory.steps] == [
        "original task",
        "original answer",
    ]
    assert trajectory.agent.extra == {"history_source": "nexau_full_trace.json"}


def test_gemini_usage_metadata_is_normalized(tmp_path: Path) -> None:
    trace_path = tmp_path / "nexau_in_memory_tracer.json"
    trace_path.write_text(
        json.dumps(
            [
                {
                    "type": "LLM",
                    "outputs": {
                        "usageMetadata": {
                            "promptTokenCount": 12,
                            "candidatesTokenCount": 5,
                            "thoughtsTokenCount": 3,
                            "cachedContentTokenCount": 2,
                            "totalTokenCount": 20,
                        }
                    },
                }
            ]
        )
    )

    metrics, totals = collect_metrics(trace_path, logging.getLogger("test"))

    assert len(metrics) == 1
    assert metrics[0].prompt_tokens == 12
    assert metrics[0].completion_tokens == 8
    assert metrics[0].cached_tokens == 2
    assert metrics[0].extra == {"reasoning_tokens": 3}
    assert totals is not None
    assert totals.total_prompt_tokens == 12
    assert totals.total_completion_tokens == 8
