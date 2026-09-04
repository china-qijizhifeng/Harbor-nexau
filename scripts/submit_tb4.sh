#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/submit_tb4.sh --path /gpfs/.../terminal-bench [TB4 options]

Submit one complete TB4 run from the current machine to basestream-cpu. The SSH
connection only starts a remote long-task monitor and then returns; the Harbor
controller itself never runs over the lifetime of the SSH connection.

Configuration overrides (export before invoking):
  HARBOR_NEXAU_TB4_SUBMIT_HOST
  HARBOR_NEXAU_TB4_REMOTE_PROJECT_ROOT
  HARBOR_NEXAU_TB4_REMOTE_RUNTIME_ROOT
  HARBOR_NEXAU_TB4_REMOTE_CODEX_HOME
  HARBOR_NEXAU_TB4_MONITOR_TIMEOUT_MINUTES
EOF
}

if (($# == 0)); then
  usage >&2
  exit 2
fi

submit_host="${HARBOR_NEXAU_TB4_SUBMIT_HOST:-basestream-cpu}"
remote_project_root="${HARBOR_NEXAU_TB4_REMOTE_PROJECT_ROOT:-/gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau-tb4-smoke}"
remote_runtime_root="${HARBOR_NEXAU_TB4_REMOTE_RUNTIME_ROOT:-/gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau}"
remote_codex_home="${HARBOR_NEXAU_TB4_REMOTE_CODEX_HOME:-/gpfs/users/i-zhangguoqiang/info/.codex}"
monitor_timeout_minutes="${HARBOR_NEXAU_TB4_MONITOR_TIMEOUT_MINUTES:-4320}"
origin_thread_id="${CODEX_THREAD_ID:-}"

if [[ -z "$submit_host" ]]; then
  printf '%s\n' 'HARBOR_NEXAU_TB4_SUBMIT_HOST cannot be empty.' >&2
  exit 2
fi
for remote_path in "$remote_project_root" "$remote_runtime_root" "$remote_codex_home"; do
  if [[ "$remote_path" != /* || "$remote_path" == *$'\n'* || "$remote_path" == *$'\r'* ]]; then
    printf 'Remote paths must be absolute single-line paths, got: %q\n' "$remote_path" >&2
    exit 2
  fi
done
if [[ ! "$monitor_timeout_minutes" =~ ^[1-9][0-9]*$ ]]; then
  printf 'HARBOR_NEXAU_TB4_MONITOR_TIMEOUT_MINUTES must be positive, got: %s\n' \
    "$monitor_timeout_minutes" >&2
  exit 2
fi

forwarded_args=("$@")
path_seen=0
dataset_path=""
job_name=""
yes_seen=0
for ((argument_index = 0; argument_index < ${#forwarded_args[@]}; argument_index++)); do
  argument="${forwarded_args[$argument_index]}"
  case "$argument" in
    -h | --help)
      usage
      exit 0
      ;;
    --path | -p)
      if ((argument_index + 1 >= ${#forwarded_args[@]})); then
        printf '%s requires a remote dataset path.\n' "$argument" >&2
        exit 2
      fi
      dataset_path="${forwarded_args[$((argument_index + 1))]}"
      ((path_seen += 1))
      ((argument_index += 1))
      ;;
    --path=* | -p=*)
      if [[ -z "${argument#*=}" ]]; then
        printf '%s requires a remote dataset path.\n' "${argument%%=*}" >&2
        exit 2
      fi
      dataset_path="${argument#*=}"
      ((path_seen += 1))
      ;;
    --job-name)
      if ((argument_index + 1 >= ${#forwarded_args[@]})); then
        printf '%s\n' '--job-name requires a value.' >&2
        exit 2
      fi
      job_name="${forwarded_args[$((argument_index + 1))]}"
      ((argument_index += 1))
      ;;
    --job-name=*)
      job_name="${argument#*=}"
      ;;
    --yes | -y)
      yes_seen=1
      ;;
  esac
done

if ((path_seen != 1)); then
  printf '%s\n' 'submit_tb4.sh requires exactly one remote --path.' >&2
  exit 2
fi
if [[ "$dataset_path" != /* || "$dataset_path" == *$'\n'* || "$dataset_path" == *$'\r'* ]]; then
  printf 'The remote TB4 dataset path must be absolute and single-line, got: %q\n' \
    "$dataset_path" >&2
  exit 2
fi
if [[ -z "$job_name" ]]; then
  job_name="nexau-tb4-$(date '+%Y%m%d-%H%M%S')"
  forwarded_args+=(--job-name "$job_name")
fi
if [[ ! "$job_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]]; then
  printf 'TB4 job name must be 1-80 safe characters, got: %q\n' "$job_name" >&2
  exit 2
fi
if ((yes_seen == 0)); then
  forwarded_args+=(--yes)
fi

if ! command -v ssh >/dev/null 2>&1; then
  printf '%s\n' 'ssh is required to submit TB4 to basestream-cpu.' >&2
  exit 2
fi

remote_command='bash -s --'
remote_arguments=(
  "$remote_project_root"
  "$remote_runtime_root"
  "$remote_codex_home"
  "$monitor_timeout_minutes"
  "$job_name"
  "$origin_thread_id"
  "$dataset_path"
  --
  "${forwarded_args[@]}"
)
for argument in "${remote_arguments[@]}"; do
  printf -v quoted_argument '%q' "$argument"
  remote_command+=" $quoted_argument"
done

ssh -o BatchMode=yes -o ConnectTimeout=15 -- "$submit_host" "$remote_command" <<'REMOTE'
set -euo pipefail

remote_project_root="$1"
remote_runtime_root="$2"
remote_codex_home="$3"
monitor_timeout_minutes="$4"
job_name="$5"
origin_thread_id="$6"
dataset_path="$7"
shift 7
if [[ "${1:-}" != "--" ]]; then
  printf '%s\n' 'Internal submit protocol error: missing argument delimiter.' >&2
  exit 2
fi
shift

monitor_py="$remote_codex_home/skills/monitor-long-task/scripts/long_task_monitor.py"
webhook_file="$remote_codex_home/secrets/feishu-long-task-webhook"
monitor_state="$remote_project_root/.basestream/monitors"
control_root="$remote_project_root/.basestream"
runtime_python="$remote_runtime_root/.venv/bin/python"
runtime_env="$remote_runtime_root/.env"

test -x "$remote_project_root/scripts/run_tb4.sh" || {
  printf 'Remote TB4 entrypoint is unavailable: %s\n' \
    "$remote_project_root/scripts/run_tb4.sh" >&2
  exit 2
}
test -d "$dataset_path" || {
  printf 'Remote TB4 dataset directory is unavailable: %s\n' "$dataset_path" >&2
  exit 2
}
test -x "$runtime_python" || {
  printf 'Remote Harbor Python is unavailable: %s\n' "$runtime_python" >&2
  exit 2
}
test -f "$runtime_env" || {
  printf 'Remote Harbor dotenv is unavailable: %s\n' "$runtime_env" >&2
  exit 2
}
test -f "$monitor_py" || {
  printf 'Remote long-task monitor is unavailable: %s\n' "$monitor_py" >&2
  exit 2
}
test -f "$webhook_file" || {
  printf 'Remote long-task webhook is unavailable: %s\n' "$webhook_file" >&2
  exit 2
}

mkdir -p -- "$monitor_state"
"$runtime_python" - "$monitor_state" "$remote_project_root/jobs" "$job_name" <<'PY'
import json
import sys
from pathlib import Path

monitor_state = Path(sys.argv[1])
jobs_root = Path(sys.argv[2])
job_name = sys.argv[3]

for state_path in monitor_state.glob("*/state.json"):
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if state.get("status") not in {"running", "monitoring"}:
        continue
    request_path = state_path.with_name("request.json")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    command = request.get("command")
    if isinstance(command, list) and job_name in map(str, command):
        raise SystemExit(
            f"TB4 job {job_name!r} is already active in monitor {state_path.parent.name}."
        )

if jobs_root.is_dir():
    existing = sorted(path.name for path in jobs_root.glob(f"{job_name}*"))
    if existing:
        raise SystemExit(
            f"TB4 job {job_name!r} already has result paths: {existing!r}."
        )
PY

resume_args=()
if [[ -n "$origin_thread_id" ]]; then
  resume_args=(
    --resume-origin-session
    --origin-thread-id "$origin_thread_id"
    --session-idle-timeout-minutes "$monitor_timeout_minutes"
    --session-message \
      'TB4全量任务已进入{status}。请读取{summary_file}和{task_log}，核验CPU/GPU子结果、异常、reward及残留资源；基于真实结果继续，不要原样重启。'
  )
fi

python3 "$monitor_py" launch \
  --project 'Harbor TB4' \
  --summary "TB4全量评测 $job_name" \
  --cwd "$remote_project_root" \
  --state-dir "$monitor_state" \
  --webhook-file "$webhook_file" \
  --timeout-minutes "$monitor_timeout_minutes" \
  "${resume_args[@]}" \
  -- \
  "$runtime_python" -m dotenv \
    --file "$runtime_env" run --no-override -- \
    /usr/bin/env \
      _HARBOR_NEXAU_DOTENV_LOADED=1 \
      PYTHONPATH="$remote_project_root/src" \
      PATH="$remote_runtime_root/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
      UV_PROJECT_ENVIRONMENT="$remote_runtime_root/.venv" \
      UV_NO_SYNC=1 \
      HARBOR_NEXAU_ENV_FILE="$runtime_env" \
      HARBOR_BASESTREAM_CONTROL_ROOT="$control_root" \
      bash "$remote_project_root/scripts/run_tb4.sh" "$@"
REMOTE
