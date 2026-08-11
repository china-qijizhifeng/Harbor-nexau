from __future__ import annotations

import os
import shlex

DEFAULT_EXPORT_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "BRAVE_API_KEY",
    "SERPER_API_KEY",
    "BP_HTML_PARSER_URL",
    "BP_HTML_PARSER_API_KEY",
    "BP_HTML_PARSER_SECRET",
    "GDPVAL_DATASET_ROOT",
)


def shell_export_prefix() -> str:
    raw_keys = (os.environ.get("GDPVAL_SHELL_EXPORT_KEYS") or "").strip()
    keys = (
        [key.strip() for key in raw_keys.split(",") if key.strip()]
        if raw_keys
        else list(DEFAULT_EXPORT_KEYS)
    )
    exports = [
        f"export {key}={shlex.quote(value)}"
        for key in keys
        if (value := os.environ.get(key)) is not None
    ]
    return " && ".join(exports) + " && " if exports else ""


def shell_env_setup() -> str:
    enabled = (os.environ.get("GDPVAL_ENABLE_SHELL_ENV_SETUP") or "").lower()
    disabled = (os.environ.get("GDPVAL_DISABLE_SHELL_ENV_SETUP") or "").lower()
    if enabled.strip() not in {"1", "true", "yes", "on"}:
        return ""
    if disabled.strip() in {"1", "true", "yes", "on"}:
        return ""
    value = (os.environ.get("GDPVAL_SHELL_ENV_SETUP") or "").strip()
    return f"{value.rstrip()} " if value else ""


def prefix_shell_command(command: str) -> str:
    prefix = shell_env_setup() + shell_export_prefix()
    return f"{prefix}{command}" if prefix else command
