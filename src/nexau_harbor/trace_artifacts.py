"""Shared trace artifact names and recovery helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

NEXAU_TRACE_EVENTS_NAME = "nexau_trace_events.jsonl"
CODEX_EVENTS_NAME = "codex_events.jsonl"
CODEX_STDERR_NAME = "codex_stderr.log"
CODEX_BRIDGE_LOG_NAME = "northgate-responses-bridge.log"
CODEX_RESULT_NAME = "codex_result.json"
CODEX_HISTORY_NAME = "codex_history.jsonl"
CODEX_SESSIONS_DIR_NAME = "sessions"
CODEX_TRACE_MANIFEST_NAME = "trace_manifest.json"

PRIORITY_TRACE_FILES = (
    NEXAU_TRACE_EVENTS_NAME,
    CODEX_EVENTS_NAME,
    CODEX_STDERR_NAME,
    CODEX_BRIDGE_LOG_NAME,
    CODEX_RESULT_NAME,
    CODEX_HISTORY_NAME,
)
PRIORITY_TRACE_DIRS = (CODEX_SESSIONS_DIR_NAME,)


@dataclass(frozen=True)
class PriorityTraceAssets:
    files: tuple[str, ...] = ()
    directories: tuple[str, ...] = ()

    @property
    def present(self) -> bool:
        return bool(self.files or self.directories)

    @property
    def exclusions(self) -> tuple[str, ...]:
        return (*self.files, *self.directories)


class AgentLogDownloadError(RuntimeError):
    """Record whether a failed log transfer put an existing trace at risk."""

    def __init__(self, message: str, *, trace_expected: bool) -> None:
        super().__init__(message)
        self.trace_expected = trace_expected


@dataclass(frozen=True)
class PendingAgentLogDownload:
    source_dir: str
    target_dir: Path
    trace_expected: bool


def priority_trace_probe_command(source_dir: str) -> str:
    """Build one shell probe that reports known trace files and directories."""

    source = source_dir.rstrip("/")
    commands: list[str] = []
    for name in PRIORITY_TRACE_FILES:
        path = shlex.quote(f"{source}/{name}")
        commands.append(f"if [ -f {path} ]; then printf 'file\\t%s\\n' {shlex.quote(name)}; fi")
    for name in PRIORITY_TRACE_DIRS:
        path = shlex.quote(f"{source}/{name}")
        commands.append(f"if [ -d {path} ]; then printf 'dir\\t%s\\n' {shlex.quote(name)}; fi")
    return "\n".join(commands)


def parse_priority_trace_probe(stdout: str) -> PriorityTraceAssets:
    """Parse the trusted-name output produced by :func:`priority_trace_probe_command`."""

    known_files = set(PRIORITY_TRACE_FILES)
    known_directories = set(PRIORITY_TRACE_DIRS)
    files: list[str] = []
    directories: list[str] = []
    for raw_line in stdout.splitlines():
        kind, separator, name = raw_line.partition("\t")
        if not separator:
            continue
        if kind == "file" and name in known_files and name not in files:
            files.append(name)
        elif kind == "dir" and name in known_directories and name not in directories:
            directories.append(name)
    return PriorityTraceAssets(tuple(files), tuple(directories))
