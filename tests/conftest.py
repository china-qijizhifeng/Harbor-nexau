from __future__ import annotations

import os

# NexAU imports call python-dotenv at module import time. Tests must never inherit
# a developer's real model credentials or silently change expectations based on
# the local .env file.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

_MODEL_ENV_PREFIXES = ("LLM_",)
_MODEL_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "NEXAU_TRACE_CHECKPOINT_SEC",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
}
for _name in list(os.environ):
    if _name in _MODEL_ENV_NAMES or _name.startswith(_MODEL_ENV_PREFIXES):
        os.environ.pop(_name, None)
