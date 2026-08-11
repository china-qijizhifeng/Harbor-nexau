from __future__ import annotations

import importlib
from typing import Any


def _resolve_symbol(*candidates: tuple[str, str]) -> Any:
    last_error: Exception | None = None
    for module_name, symbol_name in candidates:
        try:
            return getattr(importlib.import_module(module_name), symbol_name)
        except Exception as exc:
            last_error = exc
    names = ", ".join(f"{module}:{symbol}" for module, symbol in candidates)
    raise ImportError(f"Could not resolve middleware import: {names}") from last_error


ContextCompactionMiddleware = _resolve_symbol(
    (
        "nexau.archs.main_sub.execution.middleware.context_compaction",
        "ContextCompactionMiddleware",
    ),
    (
        "nexau.archs.main_sub.execution.middleware.context_compaction.middleware",
        "ContextCompactionMiddleware",
    ),
)
LongToolOutputMiddleware = _resolve_symbol(
    (
        "nexau.archs.main_sub.execution.middleware.long_tool_output",
        "LongToolOutputMiddleware",
    ),
)

__all__ = ["ContextCompactionMiddleware", "LongToolOutputMiddleware"]
