#!/usr/bin/env bash
set -euo pipefail

uv lock --upgrade-package harbor
uv sync --locked
uv run ruff check .
uv run pytest
