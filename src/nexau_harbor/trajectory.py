from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.trajectory_utils import format_trajectory_json


def _stringify_tool_result(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                rendered.append(str(part))
            elif part.get("type") == "text":
                rendered.append(str(part.get("text", "")))
            elif part.get("type") == "image":
                rendered.append("[image]")
            else:
                rendered.append(json.dumps(part, ensure_ascii=False))
        return "\n".join(item for item in rendered if item)
    return json.dumps(content, ensure_ascii=False)


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content", [])
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _message_text(message: dict[str, Any]) -> str:
    return "\n".join(
        str(block.get("text", ""))
        for block in _content_blocks(message)
        if block.get("type") == "text" and block.get("text")
    )


def _message_reasoning(message: dict[str, Any]) -> str | None:
    parts = [
        str(block.get("text", ""))
        for block in _content_blocks(message)
        if block.get("type") in {"reasoning", "thinking"} and block.get("text")
    ]
    return "\n\n".join(parts) or None


def _timestamp(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _usage_metrics(usage: Any) -> Metrics | None:
    if not isinstance(usage, dict):
        return None
    is_gemini_usage = "promptTokenCount" in usage or "candidatesTokenCount" in usage
    prompt = (
        usage.get("promptTokenCount")
        if is_gemini_usage
        else usage.get("prompt_tokens", usage.get("input_tokens"))
    )
    reasoning_tokens = usage.get("thoughtsTokenCount") if is_gemini_usage else None
    if is_gemini_usage:
        candidates = usage.get("candidatesTokenCount")
        completion = sum(
            value for value in (candidates, reasoning_tokens) if isinstance(value, int)
        )
    else:
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
    cached = (
        usage.get("cached_tokens")
        or usage.get("cache_read_tokens")
        or usage.get("cache_read_input_tokens")
        or usage.get("cachedContentTokenCount")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or (usage.get("input_tokens_details") or {}).get("cached_tokens")
    )
    cost = usage.get("cost_usd", usage.get("cost"))
    if not any(isinstance(item, (int, float)) for item in (prompt, completion, cached, cost)):
        return None
    return Metrics(
        prompt_tokens=prompt if isinstance(prompt, int) and prompt > 0 else None,
        completion_tokens=(completion if isinstance(completion, int) and completion > 0 else None),
        cached_tokens=cached if isinstance(cached, int) and cached > 0 else None,
        cost_usd=cost if isinstance(cost, (int, float)) and cost > 0 else None,
        extra=(
            {"reasoning_tokens": reasoning_tokens}
            if isinstance(reasoning_tokens, int) and reasoning_tokens > 0
            else None
        ),
    )


def _iter_trace_spans(node: Any) -> list[dict[str, Any]]:
    if not isinstance(node, dict):
        return []
    spans = [node]
    for child in node.get("children", []):
        spans.extend(_iter_trace_spans(child))
    return spans


def rebuild_trace_from_jsonl(
    event_path: Path,
    trace_path: Path,
    logger: logging.Logger,
) -> bool:
    """Rebuild the legacy nested trace artifact from append-only span events."""

    if not event_path.exists():
        return False

    spans: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    try:
        with event_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    logger.warning(
                        "Skipped incomplete NexAU trace event at %s:%s",
                        event_path,
                        line_number,
                    )
                    continue
                if not isinstance(event, dict):
                    continue
                event_type = event.get("event")
                if event_type == "start" and isinstance(event.get("span"), dict):
                    raw_span = event["span"]
                    span_id = str(raw_span.get("id") or "")
                    if not span_id:
                        continue
                    spans[span_id] = {
                        "id": span_id,
                        "name": str(raw_span.get("name") or ""),
                        "type": str(raw_span.get("type") or ""),
                        "parent_id": raw_span.get("parent_id"),
                        "start_time": raw_span.get("start_time"),
                        "end_time": None,
                        "duration_ms": None,
                        "inputs": raw_span.get("inputs") or {},
                        "outputs": {},
                        "attributes": raw_span.get("attributes") or {},
                        "error": None,
                        "children": [],
                    }
                    ordered_ids.append(span_id)
                elif event_type == "end":
                    span_id = str(event.get("span_id") or "")
                    span = spans.get(span_id)
                    if span is None:
                        continue
                    span["end_time"] = event.get("end_time")
                    span["outputs"] = event.get("outputs") or {}
                    span["error"] = event.get("error")
                    span["attributes"] = {
                        **span["attributes"],
                        **(event.get("attributes") or {}),
                    }
                    if isinstance(span["start_time"], (int, float)) and isinstance(
                        span["end_time"], (int, float)
                    ):
                        span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000
    except OSError:
        logger.exception("Failed to read NexAU JSONL trace file: %s", event_path)
        return False

    roots: list[dict[str, Any]] = []
    for span_id in ordered_ids:
        span = spans[span_id]
        parent = spans.get(str(span["parent_id"])) if span["parent_id"] else None
        if parent is None:
            roots.append(span)
        else:
            parent["children"].append(span)

    temporary_path = trace_path.with_name(f".{trace_path.name}.tmp")
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(roots, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(trace_path)
    except OSError:
        logger.exception("Failed to write rebuilt NexAU trace file: %s", trace_path)
        temporary_path.unlink(missing_ok=True)
        return False
    return True


def collect_metrics(
    trace_path: Path,
    logger: logging.Logger,
) -> tuple[list[Metrics], FinalMetrics | None]:
    if not trace_path.exists():
        return [], None
    try:
        traces = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.exception("Failed to read NexAU trace file: %s", trace_path)
        return [], None
    if not isinstance(traces, list):
        return [], None

    metrics: list[Metrics] = []
    for root in traces:
        for span in _iter_trace_spans(root):
            if str(span.get("type", "")).upper() != "LLM":
                continue
            outputs = span.get("outputs") or {}
            usage = outputs.get("usage")
            if usage is None:
                usage = outputs.get("usageMetadata") or outputs.get("usage_metadata")
            normalized = _usage_metrics(usage)
            if normalized is not None:
                metrics.append(normalized)

    if not metrics:
        return [], None
    return metrics, FinalMetrics(
        total_prompt_tokens=sum(item.prompt_tokens or 0 for item in metrics) or None,
        total_completion_tokens=sum(item.completion_tokens or 0 for item in metrics) or None,
        total_cached_tokens=sum(item.cached_tokens or 0 for item in metrics) or None,
        total_cost_usd=sum(item.cost_usd or 0 for item in metrics) or None,
    )


def history_to_trajectory(
    history: Any,
    llm_metrics: list[Metrics],
    final_metrics: FinalMetrics | None,
    *,
    session_id: str,
    agent_name: str,
    agent_version: str,
    model_name: str | None,
    reasoning_effort: str | None,
    history_source: str = "nexau_history.json",
) -> Trajectory | None:
    if not isinstance(history, list) or not history:
        return None

    steps: list[Step] = []
    agent_step_indexes: list[int] = []
    index = 0
    while index < len(history):
        message = history[index]
        if not isinstance(message, dict):
            index += 1
            continue
        role = str(message.get("role", ""))
        created_at = _timestamp(message.get("created_at"))

        if role in {"system", "user"}:
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=created_at,
                    source=role,
                    message=_message_text(message),
                )
            )
            index += 1
            continue

        if role == "assistant":
            tool_calls: list[ToolCall] = []
            for block in _content_blocks(message):
                if block.get("type") != "tool_use":
                    continue
                call_id = str(block.get("id", "")).strip()
                if not call_id:
                    continue
                arguments = block.get("input", {})
                tool_calls.append(
                    ToolCall(
                        tool_call_id=call_id,
                        function_name=str(block.get("name", "")),
                        arguments=arguments if isinstance(arguments, dict) else {},
                    )
                )

            call_ids = {call.tool_call_id for call in tool_calls}
            observations: list[ObservationResult] = []
            lookahead = index + 1
            while lookahead < len(history):
                next_message = history[lookahead]
                if not isinstance(next_message, dict) or next_message.get("role") != "tool":
                    break
                for block in _content_blocks(next_message):
                    if block.get("type") != "tool_result":
                        continue
                    raw_id = str(block.get("tool_use_id", "")).strip()
                    observations.append(
                        ObservationResult(
                            source_call_id=raw_id if raw_id in call_ids else None,
                            content=_stringify_tool_result(block.get("content")),
                        )
                    )
                lookahead += 1

            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=created_at,
                    source="agent",
                    model_name=model_name,
                    reasoning_effort=reasoning_effort,
                    message=_message_text(message),
                    reasoning_content=_message_reasoning(message),
                    tool_calls=tool_calls or None,
                    observation=(Observation(results=observations) if observations else None),
                    llm_call_count=1,
                )
            )
            agent_step_indexes.append(len(steps) - 1)
            index = lookahead
            continue

        index += 1

    if not steps:
        return None
    if len(llm_metrics) == len(agent_step_indexes):
        for step_index, metrics in zip(agent_step_indexes, llm_metrics, strict=True):
            steps[step_index].metrics = metrics

    totals = final_metrics.model_copy(deep=True) if final_metrics else FinalMetrics()
    totals.total_steps = len(steps)
    if len(llm_metrics) != len(agent_step_indexes):
        totals.extra = {
            **(totals.extra or {}),
            "llm_span_count": len(llm_metrics),
            "agent_step_count": len(agent_step_indexes),
        }

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(
            name=agent_name,
            version=agent_version,
            model_name=model_name,
            extra={"history_source": history_source},
        ),
        steps=steps,
        final_metrics=totals,
    )


def populate_context_from_logs(
    *,
    logs_dir: Path,
    context: AgentContext,
    agent_name: str,
    agent_version: str,
    model_name: str | None,
    reasoning_effort: str | None,
    logger: logging.Logger,
) -> None:
    nested_trace_path = logs_dir / "nexau_in_memory_tracer.json"
    rebuild_trace_from_jsonl(
        logs_dir / "nexau_trace_events.jsonl",
        nested_trace_path,
        logger,
    )
    metrics, totals = collect_metrics(nested_trace_path, logger)
    history_candidates = (
        logs_dir / "nexau_full_trace.json",
        logs_dir / "nexau_history.json",
    )
    for history_path in history_candidates:
        if not history_path.exists():
            continue
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, list) or not history:
                continue
            trajectory = history_to_trajectory(
                history,
                metrics,
                totals,
                session_id=logs_dir.parent.name,
                agent_name=agent_name,
                agent_version=agent_version,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                history_source=history_path.name,
            )
            if trajectory is not None:
                (logs_dir / "trajectory.json").write_text(
                    format_trajectory_json(trajectory.to_json_dict()),
                    encoding="utf-8",
                )
                totals = trajectory.final_metrics
                break
        except (OSError, ValueError):
            logger.exception("Failed to convert NexAU history: %s", history_path)

    if totals is None:
        return
    context.cost_usd = totals.total_cost_usd
    context.n_input_tokens = totals.total_prompt_tokens or 0
    context.n_output_tokens = totals.total_completion_tokens or 0
    context.n_cache_tokens = totals.total_cached_tokens or 0
