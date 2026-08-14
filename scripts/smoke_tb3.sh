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
agent_config="${HARBOR_NEXAU_CONFIG_PATH:-$project_root/src/nexau_harbor/configs/terminal_bench/code_agent.yaml}"
trace_mode=0
trace_timeout_sec="${HARBOR_NEXAU_TRACE_TIMEOUT_SEC:-72000}"
e2b_start_ramp_delay_sec="${HARBOR_NEXAU_E2B_START_RAMP_DELAY_SEC:-0}"
forwarded_args=()

while (($# > 0)); do
  case "$1" in
    --trace-mode)
      trace_mode=1
      shift
      ;;
    --trace-timeout-sec)
      if (($# < 2)); then
        printf '%s\n' '--trace-timeout-sec requires a positive integer argument' >&2
        exit 2
      fi
      trace_mode=1
      trace_timeout_sec="$2"
      shift 2
      ;;
    --trace-timeout-sec=*)
      trace_mode=1
      trace_timeout_sec="${1#*=}"
      shift
      ;;
    --e2b-start-ramp-delay-sec)
      if (($# < 2)); then
        printf '%s\n' '--e2b-start-ramp-delay-sec requires a finite non-negative number' >&2
        exit 2
      fi
      e2b_start_ramp_delay_sec="$2"
      shift 2
      ;;
    --e2b-start-ramp-delay-sec=*)
      e2b_start_ramp_delay_sec="${1#*=}"
      shift
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done
set -- "${forwarded_args[@]}"

if [[ "$trace_mode" == "1" && ! "$trace_timeout_sec" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Trace timeout must be a positive integer, got: %s\n' "$trace_timeout_sec" >&2
  exit 2
fi

if [[ ! "$e2b_start_ramp_delay_sec" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  printf 'E2B start ramp delay must be a finite non-negative number, got: %s\n' \
    "$e2b_start_ramp_delay_sec" >&2
  exit 2
fi
export HARBOR_NEXAU_E2B_START_RAMP_DELAY_SEC="$e2b_start_ramp_delay_sec"
if [[ "$e2b_start_ramp_delay_sec" != "0" ]]; then
  printf 'E2B sandbox starts will be staggered by %s seconds; Harbor concurrency is unchanged.\n' \
    "$e2b_start_ramp_delay_sec" >&2
fi

if [[ ! -f "$agent_config" ]]; then
  printf 'NexAU config file not found: %s\n' "$agent_config" >&2
  exit 2
fi

task_source_args=(--dataset "$dataset")
for argument in "$@"; do
  case "$argument" in
    --path | --path=* | -p | --dataset | --dataset=* | -d)
      # An explicit local path or dataset owns task selection. In particular,
      # --path must bypass Harbor Hub instead of being combined with the default.
      task_source_args=()
      break
      ;;
  esac
done

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

if [[ "$trace_mode" == "1" ]]; then
  for argument in "$@"; do
    case "$argument" in
      --config | --config=* | -c | --agent | --agent=* | -a | --model | --model=* | -m | \
        --timeout-multiplier | --timeout-multiplier=* | \
        --agent-timeout-multiplier | --agent-timeout-multiplier=*)
        printf '%s\n' \
          'Trace mode owns the Harbor agent and timeout config; use HARBOR_NEXAU_MODEL and --trace-timeout-sec instead.' >&2
        exit 2
        ;;
    esac
  done

  yaml_quote() {
    local escaped="${1//\'/\'\'}"
    printf "'%s'" "$escaped"
  }

  trace_config="$(mktemp "${TMPDIR:-/tmp}/harbor-nexau-trace.XXXXXX.yaml")"
  trap 'rm -f -- "$trace_config"' EXIT
  {
    printf '%s\n' 'agents:'
    printf '  - name: %s\n' "$(yaml_quote 'nexau_harbor.agent:NexAUAgent')"
    printf '    model_name: %s\n' "$(yaml_quote "$model")"
    printf '    override_timeout_sec: %s\n' "$trace_timeout_sec"
    printf '%s\n' '    kwargs:'
    printf '      config_path: %s\n' "$(yaml_quote "$agent_config")"
  } >"$trace_config"

  printf 'Trace mode: Harbor task limits are replaced with %s seconds; E2B still has its configured lifetime cap.\n' \
    "$trace_timeout_sec" >&2

  "${harbor_command[@]}" run \
    --config "$trace_config" \
    "${task_source_args[@]}" \
    "${environment_args[@]}" \
    --n-tasks "${HARBOR_NEXAU_N_TASKS:-1}" \
    --n-concurrent "${HARBOR_NEXAU_CONCURRENCY:-1}" \
    "$@"
  exit $?
fi

exec "${harbor_command[@]}" run \
  "${task_source_args[@]}" \
  --agent nexau_harbor.agent:NexAUAgent \
  --agent-kwarg "config_path=$agent_config" \
  --model "$model" \
  "${environment_args[@]}" \
  --n-tasks "${HARBOR_NEXAU_N_TASKS:-1}" \
  --n-concurrent "${HARBOR_NEXAU_CONCURRENCY:-1}" \
  "$@"
