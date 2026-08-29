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
harness="${HARBOR_NEXAU_HARNESS:-nexau}"
codex_version="${HARBOR_CODEX_VERSION:-0.148.0}"
codex_reasoning_effort="${HARBOR_CODEX_REASONING_EFFORT:-high}"
codex_reasoning_summary="${HARBOR_CODEX_REASONING_SUMMARY:-detailed}"
codex_upstream_model="${HARBOR_CODEX_UPSTREAM_MODEL:-nex-agi/Nex-N2-Pro}"
trace_mode=0
trace_timeout_sec="${HARBOR_NEXAU_TRACE_TIMEOUT_SEC:-36000}"
agent_setup_timeout_sec="${HARBOR_AGENT_SETUP_TIMEOUT_SEC:-3600}"
environment_build_timeout_multiplier="${HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER:-4}"
e2b_start_ramp_delay_sec="${HARBOR_NEXAU_E2B_START_RAMP_DELAY_SEC:-0}"
e2b_preflight_timeout_sec="${HARBOR_NEXAU_E2B_PREFLIGHT_TIMEOUT_SEC:-7200}"
e2b_preflight_attempts="${HARBOR_NEXAU_E2B_PREFLIGHT_ATTEMPTS:-3}"
refresh_e2b_routing=0
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
    --harness)
      if (($# < 2)); then
        printf '%s\n' '--harness requires nexau or codex' >&2
        exit 2
      fi
      harness="$2"
      shift 2
      ;;
    --harness=*)
      harness="${1#*=}"
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
    --refresh-e2b-routing)
      refresh_e2b_routing=1
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
if [[ "$harness" != "nexau" && "$harness" != "codex" ]]; then
  printf 'Harness must be nexau or codex, got: %s\n' "$harness" >&2
  exit 2
fi

if [[ ! "$e2b_start_ramp_delay_sec" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  printf 'E2B start ramp delay must be a finite non-negative number, got: %s\n' \
    "$e2b_start_ramp_delay_sec" >&2
  exit 2
fi
if [[ ! "$e2b_preflight_timeout_sec" =~ ^[1-9][0-9]*([.][0-9]*)?$ ]]; then
  printf 'E2B preflight timeout must be positive, got: %s\n' \
    "$e2b_preflight_timeout_sec" >&2
  exit 2
fi
if [[ ! "$e2b_preflight_attempts" =~ ^[1-9][0-9]*$ ]]; then
  printf 'E2B preflight attempts must be a positive integer, got: %s\n' \
    "$e2b_preflight_attempts" >&2
  exit 2
fi
if [[ ! "$agent_setup_timeout_sec" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Agent setup timeout must be a positive integer, got: %s\n' \
    "$agent_setup_timeout_sec" >&2
  exit 2
fi
if [[ ! "$environment_build_timeout_multiplier" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Environment build timeout multiplier must be a positive integer, got: %s\n' \
    "$environment_build_timeout_multiplier" >&2
  exit 2
fi
if [[ "$refresh_e2b_routing" == "1" && "$task_partition" != "auto" ]]; then
  printf '%s\n' '--refresh-e2b-routing requires --auto-partition' >&2
  exit 2
fi
export HARBOR_NEXAU_E2B_START_RAMP_DELAY_SEC="$e2b_start_ramp_delay_sec"
if [[ "$e2b_start_ramp_delay_sec" != "0" && "$task_partition" != "auto" ]]; then
  printf 'E2B sandbox starts will be staggered by %s seconds; Harbor concurrency is unchanged.\n' \
    "$e2b_start_ramp_delay_sec" >&2
fi

if [[ "$harness" == "nexau" && ! -f "$agent_config" ]]; then
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

  total_task_count=0
  shopt -s nullglob
  for task_dir in "$auto_dataset_path"/*; do
    if [[ ! -d "$task_dir" || ! -f "$task_dir/task.toml" ]]; then
      continue
    fi
    ((total_task_count += 1))
  done
  shopt -u nullglob

  if ((total_task_count == 0)); then
    printf 'No Harbor tasks found directly under auto-partition path: %s\n' \
      "$auto_dataset_path" >&2
    exit 2
  fi

  auto_plan_path="$(mktemp "${TMPDIR:-/tmp}/harbor-nexau-route-plan.XXXXXX.json")"
  auto_selectors_path="$(mktemp "${TMPDIR:-/tmp}/harbor-nexau-route-selectors.XXXXXX.tsv")"
  cleanup_auto_route_files() {
    rm -f -- "$auto_plan_path" "$auto_selectors_path"
  }
  trap cleanup_auto_route_files EXIT

  preflight_args=(
    uv run --project "$project_root" python -m nexau_harbor.e2b_preflight
    --dataset "$auto_dataset_path"
    --output "$auto_plan_path"
    --selectors-output "$auto_selectors_path"
    --concurrency "$total_task_count"
    --start-ramp-delay-sec "$e2b_start_ramp_delay_sec"
    --timeout-sec "$e2b_preflight_timeout_sec"
    --attempts "$e2b_preflight_attempts"
  )
  if [[ "$refresh_e2b_routing" == "1" ]]; then
    preflight_args+=(--refresh)
  fi

  printf 'Preflighting plain E2B compatibility for %s task(s); no Agent will run.\n' \
    "$total_task_count" >&2
  # E2B's Dockerfile parser prints ignored COMMENT instructions to stdout.
  # Route decisions and actionable diagnostics are logged to stderr instead.
  "${preflight_args[@]}" >/dev/null

  plain_e2b_selectors=()
  dind_selectors=()
  while IFS=$'\t' read -r route selector; do
    if [[ -z "$route" && -z "$selector" ]]; then
      continue
    fi
    case "$route" in
      plain-e2b)
        plain_e2b_selectors+=("$selector")
        ;;
      dind)
        dind_selectors+=("$selector")
        ;;
      *)
        printf 'Invalid route %q in E2B preflight selector file.\n' "$route" >&2
        exit 2
        ;;
    esac
  done <"$auto_selectors_path"

  plain_e2b_task_count="${#plain_e2b_selectors[@]}"
  dind_task_count="${#dind_selectors[@]}"
  if ((plain_e2b_task_count + dind_task_count != total_task_count)); then
    printf 'E2B preflight returned %s routes for %s tasks.\n' \
      "$((plain_e2b_task_count + dind_task_count))" "$total_task_count" >&2
    exit 2
  fi

  printf 'Auto partition selected %s plain E2B and %s DinD task(s).\n' \
    "$plain_e2b_task_count" "$dind_task_count" >&2

  auto_common_args=()
  if [[ "$trace_mode" == "1" ]]; then
    auto_common_args+=(--trace-timeout-sec "$trace_timeout_sec")
  fi
  auto_common_args+=(--harness "$harness")
  auto_common_args+=(--e2b-start-ramp-delay-sec "$e2b_start_ramp_delay_sec")
  auto_common_args+=("${auto_forwarded_args[@]}")
  auto_child_pids=()
  auto_child_pgids=()
  auto_child_labels=()

  if ! command -v setsid >/dev/null 2>&1; then
    printf '%s\n' 'setsid is required for reliable auto-partition process cleanup' >&2
    exit 2
  fi

  if ((plain_e2b_task_count > 0)); then
    plain_e2b_args=(
      "${auto_common_args[@]}"
      --n-tasks "$plain_e2b_task_count"
      --n-concurrent "$plain_e2b_task_count"
      --job-name "$auto_job_name-plain-e2b"
    )
    for selector in "${plain_e2b_selectors[@]}"; do
      plain_e2b_args+=(--include-task-name "$selector")
    done
    setsid env HARBOR_NEXAU_ENV="$single_e2b_environment" \
      bash "$project_root/scripts/smoke_tb3.sh" "${plain_e2b_args[@]}" &
    auto_child_pids+=("$!")
    auto_child_pgids+=("$!")
    auto_child_labels+=("plain-e2b")
  fi

  if ((dind_task_count > 0)); then
    dind_args=(
      "${auto_common_args[@]}"
      --n-tasks "$dind_task_count"
      --n-concurrent "$dind_task_count"
      --job-name "$auto_job_name-dind"
    )
    for selector in "${dind_selectors[@]}"; do
      dind_args+=(--include-task-name "$selector")
    done
    setsid env HARBOR_NEXAU_ENV="$compose_e2b_environment" \
      bash "$project_root/scripts/smoke_tb3.sh" "${dind_args[@]}" &
    auto_child_pids+=("$!")
    auto_child_pgids+=("$!")
    auto_child_labels+=("dind")
  fi

  terminate_auto_children() {
    local exit_code="$1"
    local child_pid child_pgid any_alive deadline
    trap - INT TERM

    for child_pgid in "${auto_child_pgids[@]}"; do
      kill -TERM -- "-$child_pgid" 2>/dev/null || true
    done

    deadline=$((SECONDS + 90))
    while ((SECONDS < deadline)); do
      any_alive=0
      for child_pgid in "${auto_child_pgids[@]}"; do
        if kill -0 -- "-$child_pgid" 2>/dev/null; then
          any_alive=1
          break
        fi
      done
      if ((any_alive == 0)); then
        break
      fi
      sleep 1
    done

    for child_pgid in "${auto_child_pgids[@]}"; do
      kill -KILL -- "-$child_pgid" 2>/dev/null || true
    done
    for child_pid in "${auto_child_pids[@]}"; do
      wait "$child_pid" 2>/dev/null || true
    done
    exit "$exit_code"
  }
  trap 'terminate_auto_children 130' INT
  trap 'terminate_auto_children 143' TERM

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

environment_timeout_args=()
has_environment_timeout_arg=0
for argument in "$@"; do
  case "$argument" in
    --environment-build-timeout-multiplier | --environment-build-timeout-multiplier=*)
      has_environment_timeout_arg=1
      break
      ;;
  esac
done
if [[ "$has_environment_timeout_arg" == "0" ]]; then
  environment_timeout_args=(
    --environment-build-timeout-multiplier "$environment_build_timeout_multiplier"
  )
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
          'Trace mode owns the Harbor agent and timeout config; use HARBOR_NEXAU_MODEL, --harness, and --trace-timeout-sec instead.' >&2
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
    if [[ "$harness" == "codex" ]]; then
      printf '  - name: %s\n' "$(yaml_quote 'nexau_harbor.codex_agent:TraceSafeCodexAgent')"
    else
      printf '  - name: %s\n' "$(yaml_quote 'nexau_harbor.agent:NexAUAgent')"
    fi
    printf '    model_name: %s\n' "$(yaml_quote "$model")"
    printf '    override_timeout_sec: %s\n' "$trace_timeout_sec"
    printf '    override_setup_timeout_sec: %s\n' "$agent_setup_timeout_sec"
    printf '%s\n' '    kwargs:'
    if [[ "$harness" == "codex" ]]; then
      printf '      version: %s\n' "$(yaml_quote "$codex_version")"
      printf '%s\n' '      unified_exec: true'
      printf '%s\n' '      code_mode: false'
      printf '%s\n' '      code_mode_only: false'
      printf '%s\n' '      multi_agent: false'
      printf '%s\n' '      responses_bridge: true'
      printf '%s\n' '      bridge_port: 48123'
      printf '      provider_name: %s\n' "$(yaml_quote 'northgate')"
      printf '      upstream_model_name: %s\n' "$(yaml_quote "$codex_upstream_model")"
      printf '%s\n' '      context_window: 262144'
      printf '      reasoning_effort: %s\n' "$(yaml_quote "$codex_reasoning_effort")"
      printf '      reasoning_summary: %s\n' "$(yaml_quote "$codex_reasoning_summary")"
    else
      printf '      config_path: %s\n' "$(yaml_quote "$agent_config")"
    fi
  } >"$trace_config"

  printf 'Trace mode (%s harness): Harbor task limits are replaced with %s seconds; E2B still has its configured lifetime cap.\n' \
    "$harness" "$trace_timeout_sec" >&2

  "${harbor_command[@]}" run \
    --config "$trace_config" \
    "${task_source_args[@]}" \
    "${partition_args[@]}" \
    "${environment_args[@]}" \
    "${environment_timeout_args[@]}" \
    --n-tasks "${HARBOR_NEXAU_N_TASKS:-1}" \
    --n-concurrent "${HARBOR_NEXAU_CONCURRENCY:-1}" \
    "$@"
  exit $?
fi

agent_args=()
if [[ "$harness" == "codex" ]]; then
  agent_args=(
    --agent nexau_harbor.codex_agent:TraceSafeCodexAgent
    --agent-kwarg "version=$codex_version"
    --agent-kwarg "unified_exec=true"
    --agent-kwarg "code_mode=false"
    --agent-kwarg "code_mode_only=false"
    --agent-kwarg "multi_agent=false"
    --agent-kwarg "responses_bridge=true"
    --agent-kwarg "bridge_port=48123"
    --agent-kwarg "provider_name=northgate"
    --agent-kwarg "upstream_model_name=$codex_upstream_model"
    --agent-kwarg "context_window=262144"
    --agent-kwarg "reasoning_effort=$codex_reasoning_effort"
    --agent-kwarg "reasoning_summary=$codex_reasoning_summary"
  )
else
  agent_args=(
    --agent nexau_harbor.agent:NexAUAgent
    --agent-kwarg "config_path=$agent_config"
  )
fi

exec "${harbor_command[@]}" run \
  "${task_source_args[@]}" \
  "${partition_args[@]}" \
  "${agent_args[@]}" \
  --model "$model" \
  "${environment_args[@]}" \
  "${environment_timeout_args[@]}" \
  --n-tasks "${HARBOR_NEXAU_N_TASKS:-1}" \
  --n-concurrent "${HARBOR_NEXAU_CONCURRENCY:-1}" \
  "$@"
