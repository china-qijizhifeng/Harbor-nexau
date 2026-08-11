from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nexau.archs.tracer.core import SpanType

from nexau_harbor import runner
from nexau_harbor.configs.gdpval.gdpval_nexau_tracer import (
    ThreadSafeInMemoryTracer,
)


class FakeAgent:
    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": "current"}]}
        ]
        self.full_trace: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": "original"}]}
        ]
        self.tracer = ThreadSafeInMemoryTracer()
        span = self.tracer.start_span("agent", SpanType.AGENT, {"query": "test"})
        self.tracer.end_span(span, {"response": "done"})
        self.config = SimpleNamespace(resolved_tracer=self.tracer)
        self.global_storage = None
        self.sandbox_manager = SimpleNamespace(instance=None)

    def run(self, *, message: str, context: dict[str, Any]) -> str:
        assert message == "solve it"
        assert context["working_directory"]
        return "done"


class FailingAgent(FakeAgent):
    def run(self, *, message: str, context: dict[str, Any]) -> str:
        raise RuntimeError("simulated agent failure")


def test_running_checkpoint_is_atomic_and_self_describing(tmp_path: Path) -> None:
    agent = FakeAgent()

    runner._checkpoint_trace_artifacts(agent, tmp_path, interval_sec=30)

    assert (
        json.loads((tmp_path / "nexau_history.json").read_text())[0]["content"][0]["text"]
        == "current"
    )
    assert (
        json.loads((tmp_path / "nexau_full_trace.json").read_text())[0]["content"][0]["text"]
        == "original"
    )
    assert len(json.loads((tmp_path / "nexau_in_memory_tracer.json").read_text())) == 1
    state = json.loads((tmp_path / "nexau_trace_checkpoint.json").read_text())
    assert state["status"] == "running"
    assert state["interval_sec"] == 30
    assert state["errors"] == []
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_periodic_checkpointer_repeats_until_stopped(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = 0
    repeated = threading.Event()

    def checkpoint(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            repeated.set()

    monkeypatch.setattr(runner, "_checkpoint_trace_artifacts", checkpoint)
    checkpointer = runner._PeriodicTraceCheckpointer(FakeAgent(), tmp_path, 0.01)

    checkpointer.start()
    try:
        assert repeated.wait(timeout=1)
    finally:
        checkpointer.stop()

    stopped_at = calls
    assert stopped_at >= 2
    threading.Event().wait(timeout=0.03)
    assert calls == stopped_at


def test_runner_main_persists_full_trace_and_completion_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "code_agent.yaml"
    config_path.write_text("type: agent\n")
    log_dir = tmp_path / "logs"
    agent = FakeAgent()
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: argparse.Namespace(
            config_path=str(config_path),
            log_dir_path=str(log_dir),
            query="solve it",
        ),
    )
    monkeypatch.setattr(runner, "_build_agent", lambda _: agent)
    monkeypatch.setenv("NEXAU_TRACE_CHECKPOINT_SEC", "0")

    assert runner.main() == 0

    assert json.loads((log_dir / "nexau_full_trace.json").read_text()) == agent.full_trace
    assert json.loads((log_dir / "nexau_result.json").read_text())["response"] == "done"
    state = json.loads((log_dir / "nexau_trace_checkpoint.json").read_text())
    assert state["status"] == "complete"
    assert "nexau_full_trace.json" in state["written_files"]


def test_runner_failure_keeps_latest_trace_artifacts(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "code_agent.yaml"
    config_path.write_text("type: agent\n")
    log_dir = tmp_path / "logs"
    agent = FailingAgent()
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: argparse.Namespace(
            config_path=str(config_path),
            log_dir_path=str(log_dir),
            query="solve it",
        ),
    )
    monkeypatch.setattr(runner, "_build_agent", lambda _: agent)
    monkeypatch.setenv("NEXAU_TRACE_CHECKPOINT_SEC", "0")

    assert runner.main() == 1

    assert "simulated agent failure" in capsys.readouterr().err
    assert (log_dir / "nexau_history.json").is_file()
    assert (log_dir / "nexau_full_trace.json").is_file()
    assert (log_dir / "nexau_in_memory_tracer.json").is_file()
    assert not (log_dir / "nexau_result.json").exists()
    state = json.loads((log_dir / "nexau_trace_checkpoint.json").read_text())
    assert state["status"] == "failed"
