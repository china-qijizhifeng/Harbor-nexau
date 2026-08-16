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
task_partition=""
forwarded_args=()

single_e2b_environment="nexau_harbor.e2b_environment:NexAUE2BEnvironment"
compose_e2b_environment="nexau_harbor.e2b_dind_environment:NexAUE2BDinDEnvironment"

# Compose tasks in the pinned 68-task CPU-only Frontier Bench/TB3 dataset.
# Harbor task-name filters keep the original task directories untouched.
compose_task_names=(
  ctr-optimization
  cumulative-layout-shift
  erp-procurement-planning
  freight-dispatch-shift
  heat-pump-warranty
  intrastat-meldung
  kv-live-surgery
  legacy-utility-triage
  medical-claims-processing
  nextjs-performance
  payments-pipeline-fix
)

# Harbor resolves symlinked local dataset entries before applying name filters,
# so the pinned local CPU68 dataset is identified by task digest instead.
compose_task_digests=(
  bc62e2e43cf5c91f63442cd787d82fc4765461c0b4fd06cab76ac4322c13dbff
  486e3bf5f0fee96a841fa70d63ddcb42bca33ca3393541723760666d4b0f501d
  97bb615dc2dfcaab20c780db1ea07af53367bf48222f61e8dfd2b6fae448ff76
  aadc8a256b8ed0f9f7d0d2b9ce1aa36cc63ea08b5599848339eab24511b0a5ea
  2b9d29e4fc9d132fd010d90e62407bea182ac1826849c46932617400471873cd
  842c2e5b4e5e735f32d85bd8f8b5dde3ea5191cd90ee2ea73792986c7b09590a
  5aa5d33e15bfe9269c4cb03d8fe950e54132ad0b82bc2d55ad72775a6627e1a2
  a61381597247d50df44c6f730767308af1be3da462904cb9d6fb17ae83fbf381
  9f7545eab64bb70b3f787c37040ce6eb537b4dc9c43956a45444dfc43c89c6f3
  7bea6536709f9c63cb7917804dfdb4210086bf9a23b26c0afc2b70ba254a4232
  4d3967affa31a3f0f5e146288b74df0532380bdcdff1669df6d10e80c85bd127
)

while (($# > 0)); do
  case "$1" in
    --auto-partition)
      if [[ -n "$task_partition" ]]; then
        printf '%s\n' \
          '--auto-partition, --compose-only, and --non-compose-only are mutually exclusive' >&2
        exit 2
      fi
      task_partition="auto"
      shift
      ;;
    --compose-only)
      if [[ -n "$task_partition" ]]; then
        printf '%s\n' \
          '--auto-partition, --compose-only, and --non-compose-only are mutually exclusive' >&2
        exit 2
      fi
      task_partition="compose"
      shift
      ;;
    --non-compose-only)
      if [[ -n "$task_partition" ]]; then
        printf '%s\n' \
          '--auto-partition, --compose-only, and --non-compose-only are mutually exclusive' >&2
        exit 2
      fi
      task_partition="non-compose"
      shift
      ;;
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

partition_args=()
if [[ -n "$task_partition" ]]; then
  for argument in "$@"; do
    case "$argument" in
      --env | --env=* | -e | --include-task-name | --include-task-name=* | -i | \
        --exclude-task-name | --exclude-task-name=* | -x)
        printf '%s\n' \
          'Task partition modes own --env and task-name filters; remove the explicit override.' >&2
        exit 2
        ;;
    esac
  done

  if [[ "$task_partition" != "auto" ]]; then
    for task_name in "${compose_task_names[@]}"; do
      if [[ "$task_partition" == "compose" ]]; then
        partition_args+=(
          --include-task-name "$task_name"
          --include-task-name "*/$task_name"
        )
      else
        partition_args+=(
          --exclude-task-name "$task_name"
          --exclude-task-name "*/$task_name"
        )
      fi
    done
    for task_digest in "${compose_task_digests[@]}"; do
      if [[ "$task_partition" == "compose" ]]; then
        partition_args+=(--include-task-name "$task_digest")
      else
        partition_args+=(--exclude-task-name "$task_digest")
      fi
    done

    if [[ "$task_partition" == "compose" ]]; then
      environment="$compose_e2b_environment"
    else
      environment="$single_e2b_environment"
    fi
    printf 'TB3 partition: %s tasks using %s.\n' "$task_partition" "$environment" >&2
  fi
fi

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
if [[ "$e2b_start_ramp_delay_sec" != "0" && "$task_partition" != "auto" ]]; then
  printf 'E2B sandbox starts will be staggered by %s seconds; Harbor concurrency is unchanged.\n' \
    "$e2b_start_ramp_delay_sec" >&2
fi

if [[ ! -f "$agent_config" ]]; then
  printf 'NexAU config file not found: %s\n' "$agent_config" >&2
  exit 2
fi

if [[ "$task_partition" == "auto" ]]; then
  auto_dataset_path=""
  auto_job_name=""
  auto_forwarded_args=()
  auto_args=("$@")
  for ((argument_index = 0; argument_index < ${#auto_args[@]}; argument_index++)); do
    argument="${auto_args[$argument_index]}"
    case "$argument" in
      --path | -p)
        if ((argument_index + 1 >= ${#auto_args[@]})); then
          printf '%s\n' "$argument requires a local dataset path" >&2
          exit 2
        fi
        auto_dataset_path="${auto_args[$((argument_index + 1))]}"
        auto_forwarded_args+=("$argument" "$auto_dataset_path")
        ((argument_index += 1))
        ;;
      --path=*)
        auto_dataset_path="${argument#*=}"
        auto_forwarded_args+=("$argument")
        ;;
      -p=*)
        auto_dataset_path="${argument#*=}"
        auto_forwarded_args+=("$argument")
        ;;
      --job-name)
        if ((argument_index + 1 >= ${#auto_args[@]})); then
          printf '%s\n' '--job-name requires a value' >&2
          exit 2
        fi
        auto_job_name="${auto_args[$((argument_index + 1))]}"
        ((argument_index += 1))
        ;;
      --job-name=*)
        auto_job_name="${argument#*=}"
        ;;
      --dataset | --dataset=* | -d | -d=*)
        printf '%s\n' \
          '--auto-partition currently requires --path to a resolved local dataset' >&2
        exit 2
        ;;
      --n-tasks | --n-tasks=* | -l | -l=* | --n-concurrent | --n-concurrent=* | -n | -n=*)
        printf '%s\n' \
          '--auto-partition sets --n-tasks and --n-concurrent from detected group sizes' >&2
        exit 2
        ;;
      *)
        auto_forwarded_args+=("$argument")
        ;;
    esac
  done

  if [[ -z "$auto_dataset_path" ]]; then
    printf '%s\n' '--auto-partition requires --path to a resolved local dataset' >&2
    exit 2
  fi
  if [[ ! -d "$auto_dataset_path" ]]; then
    printf 'Auto-partition dataset directory not found: %s\n' "$auto_dataset_path" >&2
    exit 2
  fi
  if [[ -z "$auto_job_name" ]]; then
    auto_job_name="nexau-tb3-auto-$(date '+%Y%m%d-%H%M%S')"
  fi

  compose_selectors=()
  total_task_count=0
  shopt -s nullglob
  for task_dir in "$auto_dataset_path"/*; do
    if [[ ! -d "$task_dir" || ! -f "$task_dir/task.toml" ]]; then
      continue
    fi
    ((total_task_count += 1))
    if [[ -f "$task_dir/environment/docker-compose.yaml" || \
      -f "$task_dir/environment/docker-compose.yml" || \
      -f "$task_dir/environment/compose.yaml" || \
      -f "$task_dir/environment/compose.yml" ]]; then
      resolved_task_dir="$(realpath -e -- "$task_dir")"
      compose_selectors+=("${resolved_task_dir##*/}")
    fi
  done
  shopt -u nullglob

  compose_task_count="${#compose_selectors[@]}"
  non_compose_task_count=$((total_task_count - compose_task_count))
  if ((total_task_count == 0)); then
    printf 'No Harbor tasks found directly under auto-partition path: %s\n' \
      "$auto_dataset_path" >&2
    exit 2
  fi

  printf 'Auto partition detected %s Compose and %s non-Compose tasks under %s.\n' \
    "$compose_task_count" "$non_compose_task_count" "$auto_dataset_path" >&2

  auto_common_args=()
  if [[ "$trace_mode" == "1" ]]; then
    auto_common_args+=(--trace-timeout-sec "$trace_timeout_sec")
  fi
  auto_common_args+=(--e2b-start-ramp-delay-sec "$e2b_start_ramp_delay_sec")
  auto_common_args+=("${auto_forwarded_args[@]}")
  auto_child_pids=()
  auto_child_labels=()

  if ((non_compose_task_count > 0)); then
    non_compose_args=(
      "${auto_common_args[@]}"
      --n-tasks "$non_compose_task_count"
      --n-concurrent "$non_compose_task_count"
      --job-name "$auto_job_name-non-compose"
    )
    for selector in "${compose_selectors[@]}"; do
      non_compose_args+=(--exclude-task-name "$selector")
    done
    HARBOR_NEXAU_ENV="$single_e2b_environment" \
      bash "$project_root/scripts/smoke_tb3.sh" "${non_compose_args[@]}" &
    auto_child_pids+=("$!")
    auto_child_labels+=("non-compose")
  fi

  if ((compose_task_count > 0)); then
    compose_args=(
      "${auto_common_args[@]}"
      --n-tasks "$compose_task_count"
      --n-concurrent "$compose_task_count"
      --job-name "$auto_job_name-compose"
    )
    for selector in "${compose_selectors[@]}"; do
      compose_args+=(--include-task-name "$selector")
    done
    HARBOR_NEXAU_ENV="$compose_e2b_environment" \
      bash "$project_root/scripts/smoke_tb3.sh" "${compose_args[@]}" &
    auto_child_pids+=("$!")
    auto_child_labels+=("compose")
  fi

  terminate_auto_children() {
    local child_pid
    for child_pid in "${auto_child_pids[@]}"; do
      kill -TERM "$child_pid" 2>/dev/null || true
    done
  }
  trap terminate_auto_children INT TERM

  auto_exit_code=0
  set +e
  for ((child_index = 0; child_index < ${#auto_child_pids[@]}; child_index++)); do
    wait "${auto_child_pids[$child_index]}"
    child_exit_code=$?
    if ((child_exit_code != 0)); then
      printf 'Auto-partition %s job exited with status %s.\n' \
        "${auto_child_labels[$child_index]}" "$child_exit_code" >&2
      if ((auto_exit_code == 0)); then
        auto_exit_code="$child_exit_code"
      fi
    fi
  done
  set -e
  trap - INT TERM
  exit "$auto_exit_code"
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
    "${partition_args[@]}" \
    "${environment_args[@]}" \
    --n-tasks "${HARBOR_NEXAU_N_TASKS:-1}" \
    --n-concurrent "${HARBOR_NEXAU_CONCURRENCY:-1}" \
    "$@"
  exit $?
fi

exec "${harbor_command[@]}" run \
  "${task_source_args[@]}" \
  "${partition_args[@]}" \
  --agent nexau_harbor.agent:NexAUAgent \
  --agent-kwarg "config_path=$agent_config" \
  --model "$model" \
  "${environment_args[@]}" \
  --n-tasks "${HARBOR_NEXAU_N_TASKS:-1}" \
  --n-concurrent "${HARBOR_NEXAU_CONCURRENCY:-1}" \
  "$@"
