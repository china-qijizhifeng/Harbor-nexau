#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dotenv_file="${HARBOR_NEXAU_ENV_FILE:-$project_root/.env}"

# Load the project-local dotenv before this shell reads HARBOR_NEXAU_*.
# Re-exec once because a child process cannot mutate its parent shell environment.
if [[ -f "$dotenv_file" && "${_HARBOR_NEXAU_DOTENV_LOADED:-0}" != "1" ]]; then
  exec env _HARBOR_NEXAU_DOTENV_LOADED=1 \
    uv run --project "$project_root" dotenv \
      --file "$dotenv_file" run --no-override -- \
      bash "$project_root/scripts/smoke_tb3.sh" "$@"
fi

model="${HARBOR_NEXAU_MODEL:?Set HARBOR_NEXAU_MODEL, for example openai/gpt-5}"
dataset="${HARBOR_NEXAU_DATASET:-terminal-bench/terminal-bench-3@sha256:88433fbcecd1a3f81f7a71bff4cc76c394d0edbefb7e028f90d4109b639fefba}"
environment="${HARBOR_NEXAU_ENV:-}"

environment_args=()
if [[ -n "$environment" ]]; then
  has_environment_arg=0
  for argument in "$@"; do
    case "$argument" in
      --env | --env=* | -e)
        has_environment_arg=1
        break
        ;;
    esac
  done
  if [[ "$has_environment_arg" == "0" ]]; then
    environment_args=(--env "$environment")
  fi
fi

harbor_command=(uv run --project "$project_root" harbor)
if [[ "${_HARBOR_NEXAU_DOTENV_LOADED:-0}" == "1" ]]; then
  harbor_command=(harbor)
fi

exec "${harbor_command[@]}" run \
  --dataset "$dataset" \
  --agent nexau_harbor.agent:NexAUAgent \
  --model "$model" \
  "${environment_args[@]}" \
  --n-tasks "${HARBOR_NEXAU_N_TASKS:-1}" \
  --n-concurrent "${HARBOR_NEXAU_CONCURRENCY:-1}" \
  "$@"
