from nexau_harbor.trace_artifacts import (
    PriorityTraceAssets,
    parse_priority_trace_probe,
    priority_trace_probe_command,
)


def test_priority_trace_probe_covers_nexau_and_codex_sources() -> None:
    command = priority_trace_probe_command("/logs/agent")

    assert "/logs/agent/nexau_trace_events.jsonl" in command
    assert "/logs/agent/codex_events.jsonl" in command
    assert "/logs/agent/codex_stderr.log" in command
    assert "/logs/agent/codex_result.json" in command
    assert "/logs/agent/codex_history.jsonl" in command
    assert "/logs/agent/sessions" in command


def test_priority_trace_probe_parser_ignores_unknown_and_duplicate_names() -> None:
    assets = parse_priority_trace_probe(
        "file\tcodex_events.jsonl\n"
        "file\tcodex_events.jsonl\n"
        "file\tunknown.log\n"
        "dir\tsessions\n"
        "dir\tunknown\n"
    )

    assert assets == PriorityTraceAssets(
        files=("codex_events.jsonl",),
        directories=("sessions",),
    )
