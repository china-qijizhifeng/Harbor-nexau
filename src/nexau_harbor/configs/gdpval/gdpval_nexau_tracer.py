from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any, override

from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
from nexau.archs.tracer.core import Span, SpanType


class ThreadSafeInMemoryTracer(InMemoryTracer):
    """Serialize span mutation and snapshots used by periodic checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self._snapshot_lock = RLock()

    @override
    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        with self._snapshot_lock:
            return super().start_span(
                name,
                span_type,
                inputs=inputs,
                parent_span=parent_span,
                attributes=attributes,
            )

    @override
    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        with self._snapshot_lock:
            super().end_span(
                span,
                outputs=outputs,
                error=error,
                attributes=attributes,
            )

    @override
    def dump_traces(self) -> list[dict[str, Any]]:
        with self._snapshot_lock:
            return deepcopy(super().dump_traces())


__all__ = ["ThreadSafeInMemoryTracer"]
