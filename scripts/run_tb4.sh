#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dotenv_file="${HARBOR_NEXAU_ENV_FILE:-$project_root/.env}"

if [[ -f "$dotenv_file" && "${_HARBOR_NEXAU_DOTENV_LOADED:-0}" != "1" ]]; then
  exec env _HARBOR_NEXAU_DOTENV_LOADED=1 \
    uv run --project "$project_root" dotenv \
      --file "$dotenv_file" run --no-override -- \
      bash "$project_root/scripts/run_tb4.sh" "$@"
fi

dataset_path=""
job_name=""
jobs_dir=""
path_seen=0
forwarded_args=()

while (($# > 0)); do
  case "$1" in
    --path | -p)
      if (($# < 2)); then
        printf '%s\n' "$1 requires a resolved local dataset path" >&2
        exit 2
      fi
      if ((path_seen == 1)); then
        printf '%s\n' 'run_tb4.sh accepts exactly one --path.' >&2
        exit 2
      fi
      dataset_path="$2"
      path_seen=1
      shift 2
      ;;
    --path=* | -p=*)
      if ((path_seen == 1)); then
        printf '%s\n' 'run_tb4.sh accepts exactly one --path.' >&2
        exit 2
      fi
      dataset_path="${1#*=}"
      path_seen=1
      shift
      ;;
    --job-name)
      if (($# < 2)); then
        printf '%s\n' '--job-name requires a value' >&2
        exit 2
      fi
      job_name="$2"
      shift 2
      ;;
    --job-name=*)
      job_name="${1#*=}"
      shift
      ;;
    --jobs-dir | -o)
      if (($# < 2)); then
        printf '%s\n' "$1 requires a value" >&2
        exit 2
      fi
      jobs_dir="$2"
      forwarded_args+=("$1" "$2")
      shift 2
      ;;
    --jobs-dir=*)
      jobs_dir="${1#*=}"
      forwarded_args+=("$1")
      shift
      ;;
    --dataset | --dataset=* | -d | -d=* | --task | --task=* | -t | -t=*)
      printf '%s\n' \
        'TB4 hybrid routing requires one already-resolved local --path on basestream-cpu.' >&2
      exit 2
      ;;
    --env | --env=* | -e | -e=* | --auto-partition | --compose-only | \
      --non-compose-only | --include-task-name | --include-task-name=* | -i | -i=* | \
      --exclude-task-name | --exclude-task-name=* | -x | -x=* | --n-tasks | \
      --n-tasks=* | -l | -l=* | --n-concurrent | --n-concurrent=* | -n | -n=*)
      printf '%s\n' \
        'run_tb4.sh owns environment routing, task filters, task counts, and concurrency.' >&2
      exit 2
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$dataset_path" ]]; then
  printf '%s\n' \
    'Usage: scripts/run_tb4.sh --path /gpfs/.../terminal-bench-4.0.0 [Harbor options]' >&2
  exit 2
fi
if [[ ! -d "$dataset_path" ]]; then
  printf 'TB4 dataset directory not found: %s\n' "$dataset_path" >&2
  printf '%s\n' \
    'When submitting from a machine without GPFS, use scripts/submit_tb4.sh instead.' >&2
  exit 2
fi
if [[ -z "$job_name" ]]; then
  job_name="nexau-tb4-$(date '+%Y%m%d-%H%M%S')"
fi
if [[ -z "$jobs_dir" ]]; then
  jobs_dir="$project_root/jobs"
  forwarded_args+=(--jobs-dir "$jobs_dir")
fi

pvc_host_root="${HARBOR_BASESTREAM_PVC_HOST_ROOT:-/gpfs/users/i-zhangguoqiang}"
control_root="${HARBOR_BASESTREAM_CONTROL_ROOT:-$pvc_host_root/terminal-code/Harbor-nexau/.basestream}"
route_parent="${HARBOR_NEXAU_TB4_ROUTE_ROOT:-$control_root/routes}"
plan_parent="${HARBOR_NEXAU_TB4_PLAN_ROOT:-$control_root/plans}"
mkdir -p -- "$route_parent" "$plan_parent"
route_dir="$(mktemp -d "$route_parent/tb4.XXXXXXXX")"
safe_job_name="$(printf '%s' "$job_name" | tr -cs 'A-Za-z0-9._-' '_' | cut -c1-80)"
plan_path="$plan_parent/${safe_job_name:-tb4}-$(date '+%Y%m%d-%H%M%S')-$$.json"

cleanup_route_view() {
  if [[ -n "${route_dir:-}" && -d "$route_dir" && "$route_dir" == "$route_parent"/* ]]; then
    rm -rf -- "$route_dir"
  fi
}
trap cleanup_route_view EXIT

routing_command=(
  uv run --project "$project_root" python -m nexau_harbor.tb4_routing
)
if [[ "${_HARBOR_NEXAU_DOTENV_LOADED:-0}" == "1" ]]; then
  routing_command=(python -m nexau_harbor.tb4_routing)
fi
routing_summary="$(
  "${routing_command[@]}" \
    --dataset "$dataset_path" \
    --materialize-root "$route_dir/views" \
    --output "$plan_path" \
    --gpu-host-root "$pvc_host_root"
)"
IFS=$'\t' read -r cpu_count basestream_count cpu_view basestream_view <<<"$routing_summary"
if [[ ! "$cpu_count" =~ ^[0-9]+$ || ! "$basestream_count" =~ ^[0-9]+$ ]]; then
  printf 'Invalid TB4 routing summary: %q\n' "$routing_summary" >&2
  exit 2
fi
if ((cpu_count + basestream_count == 0)); then
  printf '%s\n' 'TB4 routing selected no tasks.' >&2
  exit 2
fi

jobs_dir_absolute="$(realpath -m -- "$jobs_dir")"
pvc_host_root_absolute="$(realpath -e -- "$pvc_host_root")"
if ((basestream_count > 0)) && [[ "$jobs_dir_absolute" != "$pvc_host_root_absolute"/* ]]; then
  printf 'GPU job results must be under the Basestream PVC host root %s, got %s\n' \
    "$pvc_host_root_absolute" "$jobs_dir_absolute" >&2
  exit 2
fi

gpu_concurrency="${HARBOR_NEXAU_BASESTREAM_CONCURRENCY:-1}"
if [[ ! "$gpu_concurrency" =~ ^[1-9][0-9]*$ ]]; then
  printf 'HARBOR_NEXAU_BASESTREAM_CONCURRENCY must be a positive integer, got %s\n' \
    "$gpu_concurrency" >&2
  exit 2
fi
if ((gpu_concurrency > basestream_count && basestream_count > 0)); then
  gpu_concurrency="$basestream_count"
fi

printf 'TB4 routing: %s CPU task(s), %s H100 Basestream task(s). Plan: %s\n' \
  "$cpu_count" "$basestream_count" "$plan_path" >&2

if ! command -v setsid >/dev/null 2>&1; then
  printf '%s\n' 'setsid is required for reliable TB4 child-process cleanup.' >&2
  exit 2
fi

basestream_environment="nexau_harbor.basestream_environment:BasestreamEnvironment"
child_pids=()
child_pgids=()
child_labels=()

if ((cpu_count > 0)); then
  setsid bash "$project_root/scripts/smoke_tb3.sh" \
    --auto-partition \
    --path "$cpu_view" \
    --job-name "$job_name-cpu" \
    "${forwarded_args[@]}" &
  child_pids+=("$!")
  child_pgids+=("$!")
  child_labels+=("cpu")
fi

if ((basestream_count > 0)); then
  setsid env HARBOR_NEXAU_ENV="$basestream_environment" \
    bash "$project_root/scripts/smoke_tb3.sh" \
      --path "$basestream_view" \
      --job-name "$job_name-basestream" \
      --n-tasks "$basestream_count" \
      --n-concurrent "$gpu_concurrency" \
      "${forwarded_args[@]}" &
  child_pids+=("$!")
  child_pgids+=("$!")
  child_labels+=("basestream")
fi

terminate_children() {
  local exit_code="$1"
  local child_pid child_pgid any_alive deadline
  trap - INT TERM
  for child_pgid in "${child_pgids[@]}"; do
    kill -TERM -- "-$child_pgid" 2>/dev/null || true
  done
  deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    any_alive=0
    for child_pgid in "${child_pgids[@]}"; do
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
  for child_pgid in "${child_pgids[@]}"; do
    kill -KILL -- "-$child_pgid" 2>/dev/null || true
  done
  for child_pid in "${child_pids[@]}"; do
    wait "$child_pid" 2>/dev/null || true
  done
  exit "$exit_code"
}
trap 'terminate_children 130' INT
trap 'terminate_children 143' TERM

exit_code=0
set +e
for ((child_index = 0; child_index < ${#child_pids[@]}; child_index++)); do
  wait "${child_pids[$child_index]}"
  child_exit_code=$?
  if ((child_exit_code != 0)); then
    printf 'TB4 %s job exited with status %s.\n' \
      "${child_labels[$child_index]}" "$child_exit_code" >&2
    if ((exit_code == 0)); then
      exit_code="$child_exit_code"
    fi
  fi
done
set -e
trap - INT TERM
exit "$exit_code"
