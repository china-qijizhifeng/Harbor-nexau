#!/usr/bin/env bash
set -euo pipefail

model="${HARBOR_NEXAU_MODEL:?Set HARBOR_NEXAU_MODEL, for example openai/gpt-5}"
dataset="${HARBOR_NEXAU_DATASET:-terminal-bench/terminal-bench-3@sha256:88433fbcecd1a3f81f7a71bff4cc76c394d0edbefb7e028f90d4109b639fefba}"

exec uv run harbor run \
  --dataset "$dataset" \
  --agent nexau_harbor.agent:NexAUAgent \
  --model "$model" \
  --n-tasks "${HARBOR_NEXAU_N_TASKS:-1}" \
  --n-concurrent "${HARBOR_NEXAU_CONCURRENCY:-1}" \
  "$@"
