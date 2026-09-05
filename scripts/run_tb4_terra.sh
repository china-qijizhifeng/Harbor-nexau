#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_tb4_terra.sh [--job-name NAME] [--dry-run]

Launch the TB4 gpt-5.6-terra evaluation as two monitored jobs:
  - 63 CPU tasks from this host through E2B
  - 3 GPU tasks from basestream-cpu through Basestream/H100

The dedicated Terra key is read without printing it. The remote GPU monitor
receives the key through SSH stdin and does not persist it to disk.
EOF
}

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
job_name=""
dry_run=0

while (($# > 0)); do
  case "$1" in
    --job-name)
      if (($# < 2)); then
        printf '%s\n' '--job-name requires a value.' >&2
        exit 2
      fi
      job_name="$2"
      shift 2
      ;;
    --job-name=*)
      job_name="${1#*=}"
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %q\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$job_name" ]]; then
  job_name="tb4-terra-$(date '+%Y%m%d-%H%M%S')"
fi
if [[ ! "$job_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  printf 'TB4 Terra job name must be 1-64 safe characters, got: %q\n' "$job_name" >&2
  exit 2
fi

python_bin="${HARBOR_TB4_TERRA_PYTHON:-$project_root/.venv/bin/python}"
project_env="${HARBOR_TB4_TERRA_PROJECT_ENV:-$project_root/.env}"
credential_env="${HARBOR_TB4_TERRA_CREDENTIAL_ENV:-/root/terminal-code/NexTask-rfc0012/.env}"
local_dataset="${HARBOR_TB4_TERRA_LOCAL_DATASET:-/root/terminal-code/datasets/tb4-4.0.0/terminal-bench}"
local_jobs="${HARBOR_TB4_TERRA_LOCAL_JOBS:-$project_root/jobs}"
route_parent="${HARBOR_TB4_TERRA_ROUTE_ROOT:-$project_root/.basestream/routes}"
plan_parent="${HARBOR_TB4_TERRA_PLAN_ROOT:-$project_root/.basestream/plans}"
monitor_py="${HARBOR_TB4_TERRA_MONITOR:-${CODEX_HOME:-/root/.codex}/skills/monitor-long-task/scripts/long_task_monitor.py}"
ssh_bin="${HARBOR_TB4_TERRA_SSH:-ssh}"
remote_host="${HARBOR_TB4_TERRA_REMOTE_HOST:-basestream-cpu}"
remote_work="${HARBOR_TB4_TERRA_REMOTE_WORK:-/gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau-tb4-smoke}"
remote_runtime="${HARBOR_TB4_TERRA_REMOTE_RUNTIME:-/gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau}"
remote_codex_home="${HARBOR_TB4_TERRA_REMOTE_CODEX_HOME:-/gpfs/users/i-zhangguoqiang/info/.codex}"
remote_dataset="${HARBOR_TB4_TERRA_REMOTE_DATASET:-/gpfs/users/i-zhangguoqiang/datasets/tb4-4.0.0/terminal-bench}"
remote_egress_state="${HARBOR_TB4_TERRA_REMOTE_EGRESS_STATE:-/gpfs/users/i-zhangguoqiang/terminal-code/NexMLTask/gpu_manager/deploy/basestream/.state.json}"
cpu_monitor_timeout="${HARBOR_TB4_TERRA_CPU_MONITOR_TIMEOUT_MINUTES:-2880}"
gpu_monitor_timeout="${HARBOR_TB4_TERRA_GPU_MONITOR_TIMEOUT_MINUTES:-4320}"
gpu_mirror_timeout="${HARBOR_TB4_TERRA_GPU_MIRROR_TIMEOUT_SEC:-57600}"
origin_thread_id="${CODEX_THREAD_ID:-}"

for value in "$cpu_monitor_timeout" "$gpu_monitor_timeout" "$gpu_mirror_timeout"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf 'TB4 Terra timeout values must be positive integers, got: %q\n' "$value" >&2
    exit 2
  fi
done
for path in \
  "$python_bin" \
  "$project_env" \
  "$credential_env" \
  "$local_dataset" \
  "$local_jobs" \
  "$route_parent" \
  "$plan_parent" \
  "$monitor_py" \
  "$remote_work" \
  "$remote_runtime" \
  "$remote_codex_home" \
  "$remote_dataset" \
  "$remote_egress_state"
do
  if [[ "$path" != /* || "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    printf 'TB4 Terra paths must be absolute and single-line, got: %q\n' "$path" >&2
    exit 2
  fi
done

test -x "$python_bin" || {
  printf 'Project Python is unavailable: %s\n' "$python_bin" >&2
  exit 2
}
test -f "$project_env" || {
  printf 'Harbor dotenv is unavailable: %s\n' "$project_env" >&2
  exit 2
}
test -f "$credential_env" || {
  printf 'Terra credential dotenv is unavailable: %s\n' "$credential_env" >&2
  exit 2
}
test -d "$local_dataset" || {
  printf 'Local TB4 dataset is unavailable: %s\n' "$local_dataset" >&2
  exit 2
}
test -f "$monitor_py" || {
  printf 'Long-task monitor is unavailable: %s\n' "$monitor_py" >&2
  exit 2
}
command -v "$ssh_bin" >/dev/null 2>&1 || {
  printf 'SSH command is unavailable: %s\n' "$ssh_bin" >&2
  exit 2
}

cpu_job="$job_name-cpu"
gpu_job="$job_name-gpu"
route_dir="$route_parent/$cpu_job"
plan_path="$plan_parent/$cpu_job.json"
if [[ -e "$route_dir" || -e "$plan_path" ]]; then
  printf 'TB4 Terra routing state already exists for job %s.\n' "$job_name" >&2
  exit 2
fi
if compgen -G "$local_jobs/$cpu_job*" >/dev/null; then
  printf 'TB4 Terra CPU result path already exists for job %s.\n' "$job_name" >&2
  exit 2
fi

terra_key="$({
  "$python_bin" - "$credential_env" <<'PY'
from pathlib import Path
import sys

from dotenv import dotenv_values

path = Path(sys.argv[1])
value = dotenv_values(path).get("Codex_API_KEY")
if not value:
    raise SystemExit(f"Codex_API_KEY is missing from {path}")
print(value)
PY
} 2>/dev/null)"
if [[ -z "$terra_key" || "$terra_key" == *$'\n'* || "$terra_key" == *$'\r'* ]]; then
  printf 'Could not load a single-line Codex_API_KEY from %s.\n' "$credential_env" >&2
  exit 2
fi
trap 'unset terra_key OPENAI_API_KEY' EXIT

if ((dry_run == 0)); then
  (
    export OPENAI_API_KEY="$terra_key"
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    export NO_PROXY=northgate.xiaobei.top
    export no_proxy=northgate.xiaobei.top
    "$python_bin" - "$project_env" <<'PY'
import sys

from dotenv import dotenv_values
from openai import OpenAI

base_url = dotenv_values(sys.argv[1]).get("OPENAI_BASE_URL")
if not base_url:
    raise SystemExit("OPENAI_BASE_URL is missing from Harbor dotenv")
client = OpenAI(base_url=base_url, timeout=60.0, max_retries=0)
response = client.responses.create(
    model="gpt-5.6-terra",
    input="Reply with exactly OK.",
    max_output_tokens=64,
)
if response.status != "completed":
    raise SystemExit(f"Terra API preflight did not complete: {response.status}")
print("Local Terra API preflight: OK")
PY
  )
fi

mkdir -p -- "$route_parent" "$plan_parent" "$local_jobs"
routing_summary="$(
  "$python_bin" -m nexau_harbor.tb4_routing \
    --dataset "$local_dataset" \
    --materialize-root "$route_dir/views" \
    --output "$plan_path"
)"
IFS=$'\t' read -r cpu_count gpu_count cpu_view gpu_view <<<"$routing_summary"
if [[ "$cpu_count" != 63 || "$gpu_count" != 3 ]]; then
  printf 'Unexpected TB4 routing summary: %q\n' "$routing_summary" >&2
  exit 2
fi
if [[ ! -d "$cpu_view" || ! -d "$gpu_view" ]]; then
  printf 'TB4 routing did not create both views: %q\n' "$routing_summary" >&2
  exit 2
fi

printf 'TB4 Terra routing: %s CPU task(s), %s GPU task(s). Plan: %s\n' \
  "$cpu_count" "$gpu_count" "$plan_path" >&2

if ((dry_run == 1)); then
  printf 'DRY_RUN job=%s model=gpt-5.6-terra harness=codex cpu_view=%s gpu_host=%s\n' \
    "$job_name" "$cpu_view" "$remote_host"
  rm -rf -- "$route_dir"
  rm -f -- "$plan_path"
  exit 0
fi

read -r -d '' remote_body <<'REMOTE' || true
set -euo pipefail
IFS= read -r OPENAI_API_KEY
if [[ -z "$OPENAI_API_KEY" ]]; then
  printf '%s\n' 'Terra key was not received over SSH stdin.' >&2
  exit 2
fi
export OPENAI_API_KEY

runtime_python="$remote_runtime/.venv/bin/python"
runtime_env="$remote_runtime/.env"
monitor="$remote_codex_home/skills/monitor-long-task/scripts/long_task_monitor.py"
webhook="$remote_codex_home/secrets/feishu-long-task-webhook"
monitor_state="$remote_work/.basestream/monitors"
control_root="$remote_work/.basestream"

test -x "$runtime_python"
test -f "$runtime_env"
test -x "$remote_work/scripts/smoke_tb3.sh"
test -d "$remote_dataset"
test -f "$remote_egress_state"
test -f "$monitor"
test -f "$webhook"

"$runtime_python" - "$remote_work/jobs" "$gpu_job" <<'PY'
from pathlib import Path
import sys

jobs = Path(sys.argv[1])
job_name = sys.argv[2]
if jobs.is_dir() and any(jobs.glob(f"{job_name}*")):
    raise SystemExit(f"Remote GPU result path already exists for {job_name}")
PY

proxy="$(jq -r .egress_proxy_url "$remote_egress_state")"
if [[ -z "$proxy" || "$proxy" == "null" ]]; then
  printf '%s\n' 'Basestream egress proxy is unavailable.' >&2
  exit 2
fi
export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy"
export http_proxy="$proxy" https_proxy="$proxy"
unset ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,.svc,.cluster.local
export no_proxy="$NO_PROXY"

"$runtime_python" - "$runtime_env" <<'PY'
import sys

from dotenv import dotenv_values
from openai import OpenAI

base_url = dotenv_values(sys.argv[1]).get("OPENAI_BASE_URL")
if not base_url:
    raise SystemExit("OPENAI_BASE_URL is missing from remote Harbor dotenv")
client = OpenAI(base_url=base_url, timeout=60.0, max_retries=0)
response = client.responses.create(
    model="gpt-5.6-terra",
    input="Reply with exactly OK.",
    max_output_tokens=64,
)
if response.status != "completed":
    raise SystemExit(f"Remote Terra API preflight did not complete: {response.status}")
print("Basestream Terra API preflight: OK")
PY

mkdir -p -- "$monitor_state"
python3 "$monitor" launch \
  --project 'Harbor TB4 Terra GPU' \
  --summary "TB4 Terra 3道GPU题 $gpu_job" \
  --cwd "$remote_work" \
  --state-dir "$monitor_state" \
  --webhook-file "$webhook" \
  --timeout-minutes "$gpu_monitor_timeout" \
  -- \
  "$runtime_python" -m dotenv \
    --file "$runtime_env" run --no-override -- \
    /usr/bin/env \
      _HARBOR_NEXAU_DOTENV_LOADED=1 \
      PYTHONPATH="$remote_work/src" \
      PATH="$remote_runtime/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
      HARBOR_NEXAU_MODEL=gpt-5.6-terra \
      HARBOR_NEXAU_HARNESS=codex \
      HARBOR_CODEX_UNIFIED_EXEC=false \
      HARBOR_CODEX_CODE_MODE=true \
      HARBOR_CODEX_CODE_MODE_ONLY=true \
      HARBOR_CODEX_MULTI_AGENT=false \
      HARBOR_CODEX_RESPONSES_BRIDGE=false \
      HARBOR_CODEX_CONTEXT_WINDOW=272000 \
      HARBOR_CODEX_REASONING_EFFORT=max \
      HARBOR_CODEX_REASONING_SUMMARY=detailed \
      HARBOR_BASESTREAM_BUILD_TIMEOUT_SEC="$gpu_mirror_timeout" \
      HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER=20 \
      HARBOR_BASESTREAM_CONTROL_ROOT="$control_root" \
      HARBOR_NEXAU_ENV=nexau_harbor.basestream_environment:BasestreamEnvironment \
      bash "$remote_work/scripts/smoke_tb3.sh" \
        --path "$remote_dataset" \
        --include-task-name fp8-rmsnorm-gemm \
        --include-task-name jax-speedrun-gpu \
        --include-task-name math-eval-grader \
        --jobs-dir "$remote_work/jobs" \
        --trace-mode \
        --harness codex \
        --n-tasks 3 \
        --n-concurrent 1 \
        --job-name "$gpu_job" \
        --yes
REMOTE

remote_command=""
for assignment in \
  "remote_work=$remote_work" \
  "remote_runtime=$remote_runtime" \
  "remote_codex_home=$remote_codex_home" \
  "remote_dataset=$remote_dataset" \
  "remote_egress_state=$remote_egress_state" \
  "gpu_job=$gpu_job" \
  "gpu_monitor_timeout=$gpu_monitor_timeout" \
  "gpu_mirror_timeout=$gpu_mirror_timeout"
do
  name="${assignment%%=*}"
  value="${assignment#*=}"
  printf -v quoted_value '%q' "$value"
  remote_command+="$name=$quoted_value "
done
printf -v quoted_body '%q' "$remote_body"
remote_command+="bash -c $quoted_body"

printf '%s\n' "$terra_key" | \
  "$ssh_bin" -o BatchMode=yes -o ConnectTimeout=15 -- "$remote_host" "$remote_command"

export OPENAI_API_KEY="$terra_key"
local_resume_args=()
if [[ -n "$origin_thread_id" ]]; then
  local_resume_args=(
    --resume-origin-session
    --origin-thread-id "$origin_thread_id"
    --session-idle-timeout-minutes "$cpu_monitor_timeout"
    --session-message \
      'TB4 Terra CPU任务已进入{status}。请读取{summary_file}和{task_log}，核验63道CPU题、Codex trace及E2B清理；结合GPU monitor继续，不要原样重启。'
  )
fi

python3 "$monitor_py" launch \
  --project 'Harbor TB4 Terra CPU' \
  --summary "TB4 Terra 63道CPU题 $cpu_job" \
  --cwd "$project_root" \
  --timeout-minutes "$cpu_monitor_timeout" \
  "${local_resume_args[@]}" \
  -- \
  /usr/bin/env \
    HARBOR_NEXAU_MODEL=gpt-5.6-terra \
    HARBOR_NEXAU_HARNESS=codex \
    HARBOR_CODEX_UNIFIED_EXEC=false \
    HARBOR_CODEX_CODE_MODE=true \
    HARBOR_CODEX_CODE_MODE_ONLY=true \
    HARBOR_CODEX_MULTI_AGENT=false \
    HARBOR_CODEX_RESPONSES_BRIDGE=false \
    HARBOR_CODEX_CONTEXT_WINDOW=272000 \
    HARBOR_CODEX_REASONING_EFFORT=max \
    HARBOR_CODEX_REASONING_SUMMARY=detailed \
    HARBOR_NEXAU_E2B_DIND_TEMPLATE=s-project-e89a/harbor-nexau-dind-16c:v1 \
    HARBOR_NEXAU_DIND_PREWARM=false \
    bash "$project_root/scripts/smoke_tb3.sh" \
      --auto-partition \
      --path "$cpu_view" \
      --jobs-dir "$local_jobs" \
      --trace-mode \
      --harness codex \
      --e2b-start-ramp-delay-sec 5 \
      --job-name "$cpu_job" \
      --yes

unset terra_key OPENAI_API_KEY
trap - EXIT
