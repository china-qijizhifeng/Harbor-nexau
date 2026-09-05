from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "run_tb4_terra.sh"


def _write_fake_tools(tmp_path: Path) -> dict[str, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-m" && "${2:-}" == "nexau_harbor.tb4_routing" ]]; then
  materialize=""
  output=""
  previous=""
  for argument in "$@"; do
    case "$previous" in
      --materialize-root) materialize="$argument" ;;
      --output) output="$argument" ;;
    esac
    previous="$argument"
  done
  mkdir -p "$materialize/cpu" "$materialize/basestream" "$(dirname "$output")"
  printf '{}\n' > "$output"
  printf '63\t3\t%s\t%s\n' "$materialize/cpu" "$materialize/basestream"
  exit 0
fi
if [[ "${2:-}" == *credentials.env ]]; then
  printf '%s\n' 'fake-terra-key'
  exit 0
fi
printf '%s\n' 'Local Terra API preflight: OK'
""",
        encoding="utf-8",
    )
    python.chmod(0o755)

    monitor_capture = tmp_path / "monitor.json"
    monitor = bin_dir / "monitor"
    monitor.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path
Path(os.environ["MONITOR_CAPTURE"]).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
print(json.dumps({{"status": "running", "task_id": "cpu-monitor"}}))
""",
        encoding="utf-8",
    )
    monitor.chmod(0o755)

    ssh_capture = tmp_path / "ssh.json"
    ssh = bin_dir / "ssh"
    ssh.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path
Path(os.environ["SSH_CAPTURE"]).write_text(
    json.dumps({{"argv": sys.argv[1:], "stdin": sys.stdin.read()}}),
    encoding="utf-8",
)
print(json.dumps({{"status": "running", "task_id": "gpu-monitor"}}))
""",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    return {
        "bin": bin_dir,
        "python": python,
        "monitor": monitor,
        "monitor_capture": monitor_capture,
        "ssh": ssh,
        "ssh_capture": ssh_capture,
    }


def test_terra_script_builds_cpu_and_gpu_monitors_without_exposing_key(
    tmp_path: Path,
) -> None:
    tools = _write_fake_tools(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    project_env = tmp_path / "project.env"
    project_env.write_text("OPENAI_BASE_URL=https://example.invalid/v1\n", encoding="utf-8")
    credential_env = tmp_path / "credentials.env"
    credential_env.write_text("Codex_API_KEY=unused-by-fake\n", encoding="utf-8")
    remote_work = "/remote/work"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tools['bin']}:{env['PATH']}",
            "CODEX_THREAD_ID": "00000000-0000-0000-0000-000000000001",
            "MONITOR_CAPTURE": str(tools["monitor_capture"]),
            "SSH_CAPTURE": str(tools["ssh_capture"]),
            "HARBOR_TB4_TERRA_PYTHON": str(tools["python"]),
            "HARBOR_TB4_TERRA_PROJECT_ENV": str(project_env),
            "HARBOR_TB4_TERRA_CREDENTIAL_ENV": str(credential_env),
            "HARBOR_TB4_TERRA_LOCAL_DATASET": str(dataset),
            "HARBOR_TB4_TERRA_LOCAL_JOBS": str(tmp_path / "jobs"),
            "HARBOR_TB4_TERRA_ROUTE_ROOT": str(tmp_path / "routes"),
            "HARBOR_TB4_TERRA_PLAN_ROOT": str(tmp_path / "plans"),
            "HARBOR_TB4_TERRA_MONITOR": str(tools["monitor"]),
            "HARBOR_TB4_TERRA_SSH": str(tools["ssh"]),
            "HARBOR_TB4_TERRA_REMOTE_HOST": "fake-basestream",
            "HARBOR_TB4_TERRA_REMOTE_WORK": remote_work,
            "HARBOR_TB4_TERRA_REMOTE_RUNTIME": "/remote/runtime",
            "HARBOR_TB4_TERRA_REMOTE_CODEX_HOME": "/remote/codex",
            "HARBOR_TB4_TERRA_REMOTE_DATASET": "/remote/dataset",
            "HARBOR_TB4_TERRA_REMOTE_EGRESS_STATE": "/remote/egress.json",
            "HARBOR_TB4_TERRA_CPU_MONITOR_TIMEOUT_MINUTES": "111",
            "HARBOR_TB4_TERRA_GPU_MONITOR_TIMEOUT_MINUTES": "222",
            "HARBOR_TB4_TERRA_GPU_MIRROR_TIMEOUT_SEC": "333",
        }
    )

    result = subprocess.run(
        [str(_SCRIPT), "--job-name", "terra-test"],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    combined_output = result.stdout + result.stderr
    assert "fake-terra-key" not in combined_output
    assert "63 CPU task(s), 3 GPU task(s)" in result.stderr

    ssh_call = json.loads(tools["ssh_capture"].read_text(encoding="utf-8"))
    assert ssh_call["stdin"] == "fake-terra-key\n"
    assert "fake-terra-key" not in " ".join(ssh_call["argv"])
    remote_command = ssh_call["argv"][-1]
    assert "HARBOR_NEXAU_MODEL=gpt-5.6-terra" in remote_command
    assert "HARBOR_NEXAU_HARNESS=codex" in remote_command
    assert "HARBOR_CODEX_UNIFIED_EXEC=false" in remote_command
    assert "HARBOR_CODEX_CODE_MODE=true" in remote_command
    assert "HARBOR_CODEX_CODE_MODE_ONLY=true" in remote_command
    assert "HARBOR_CODEX_RESPONSES_BRIDGE=false" in remote_command
    assert "HARBOR_BASESTREAM_BUILD_TIMEOUT_SEC=\"$gpu_mirror_timeout\"" in remote_command
    assert remote_command.count("--include-task-name") == 3
    assert "--auto-partition" not in remote_command

    monitor_args = json.loads(tools["monitor_capture"].read_text(encoding="utf-8"))
    joined = " ".join(shlex.quote(argument) for argument in monitor_args)
    assert "HARBOR_NEXAU_MODEL=gpt-5.6-terra" in joined
    assert "HARBOR_NEXAU_E2B_DIND_TEMPLATE=s-project-e89a/harbor-nexau-dind-16c:v1" in joined
    assert "--auto-partition" in monitor_args
    assert "--harness" in monitor_args
    assert "codex" in monitor_args
    assert f"{tmp_path}/routes/terra-test-cpu/views/cpu" in monitor_args
    assert "fake-terra-key" not in joined


def test_terra_script_rejects_unsafe_job_name_before_launch(tmp_path: Path) -> None:
    tools = _write_fake_tools(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tools['bin']}:{env['PATH']}",
            "MONITOR_CAPTURE": str(tools["monitor_capture"]),
            "SSH_CAPTURE": str(tools["ssh_capture"]),
        }
    )

    result = subprocess.run(
        [str(_SCRIPT), "--job-name", "bad name"],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "must be 1-64 safe characters" in result.stderr
    assert not tools["monitor_capture"].exists()
    assert not tools["ssh_capture"].exists()


def test_terra_script_dry_run_cleans_generated_routing_state(tmp_path: Path) -> None:
    tools = _write_fake_tools(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    project_env = tmp_path / "project.env"
    project_env.write_text("OPENAI_BASE_URL=https://example.invalid/v1\n", encoding="utf-8")
    credential_env = tmp_path / "credentials.env"
    credential_env.write_text("Codex_API_KEY=unused-by-fake\n", encoding="utf-8")
    route_root = tmp_path / "routes"
    plan_root = tmp_path / "plans"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tools['bin']}:{env['PATH']}",
            "MONITOR_CAPTURE": str(tools["monitor_capture"]),
            "SSH_CAPTURE": str(tools["ssh_capture"]),
            "HARBOR_TB4_TERRA_PYTHON": str(tools["python"]),
            "HARBOR_TB4_TERRA_PROJECT_ENV": str(project_env),
            "HARBOR_TB4_TERRA_CREDENTIAL_ENV": str(credential_env),
            "HARBOR_TB4_TERRA_LOCAL_DATASET": str(dataset),
            "HARBOR_TB4_TERRA_LOCAL_JOBS": str(tmp_path / "jobs"),
            "HARBOR_TB4_TERRA_ROUTE_ROOT": str(route_root),
            "HARBOR_TB4_TERRA_PLAN_ROOT": str(plan_root),
            "HARBOR_TB4_TERRA_MONITOR": str(tools["monitor"]),
            "HARBOR_TB4_TERRA_SSH": str(tools["ssh"]),
        }
    )

    result = subprocess.run(
        [str(_SCRIPT), "--job-name", "terra-dry", "--dry-run"],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "DRY_RUN job=terra-dry" in result.stdout
    assert not (route_root / "terra-dry-cpu").exists()
    assert not (plan_root / "terra-dry-cpu.json").exists()
    assert not tools["monitor_capture"].exists()
    assert not tools["ssh_capture"].exists()


def test_terra_script_help_is_local() -> None:
    result = subprocess.run(
        [str(_SCRIPT), "--help"],
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "63 CPU tasks" in result.stdout
    assert "3 GPU tasks" in result.stdout
