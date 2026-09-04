from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "run_tb4.sh"


def _create_task(root: Path, name: str, *, gpus: int) -> None:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Solve it.\n", encoding="utf-8")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    (task / "tests" / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    gpu_types = '["H100"]' if gpus else "[]"
    (task / "task.toml").write_text(
        (
            f'[task]\nname = "terminal-bench/{name}"\n\n'
            '[verifier]\nenvironment_mode = "separate"\n\n'
            "[environment]\n"
            "cpus = 4\n"
            "memory_mb = 8192\n"
            "storage_mb = 20480\n"
            f"gpus = {gpus}\n"
            f"gpu_types = {gpu_types}\n"
        ),
        encoding="utf-8",
    )


def _fake_uv(tmp_path: Path) -> Path:
    capture = tmp_path / "invocations"
    capture.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "uv"
    executable.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
module=""
python_index=-1
for ((index = 1; index <= $#; index++)); do
  argument="${{!index}}"
  if [[ "$argument" == "python" ]]; then
    python_index="$index"
  fi
  previous_index=$((index - 1))
  if ((previous_index > 0)) && [[ "${{!previous_index}}" == "-m" ]]; then
    module="$argument"
  fi
done
if [[ "$module" == "nexau_harbor.tb4_routing" ]]; then
  args=("$@")
  exec "{sys.executable}" "${{args[@]:$python_index}}"
fi
if [[ "$module" == "nexau_harbor.e2b_preflight" ]]; then
  dataset=""
  output=""
  selectors_output=""
  previous=""
  for argument in "$@"; do
    case "$previous" in
      --dataset) dataset="$argument" ;;
      --output) output="$argument" ;;
      --selectors-output) selectors_output="$argument" ;;
    esac
    previous="$argument"
  done
  printf '{{"tasks": []}}\n' > "$output"
  : > "$selectors_output"
  for task in "$dataset"/*; do
    [[ -f "$task/task.toml" ]] || continue
    resolved="$(realpath -e -- "$task")"
    printf 'plain-e2b\t%s\n' "${{resolved##*/}}" >> "$selectors_output"
  done
  exit 0
fi
{{
  printf 'ENV=%s\n' "${{HARBOR_NEXAU_ENV:-}}"
  printf '%s\n' "$@"
}} > "{capture}/$$"
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir


def test_tb4_script_starts_cpu_and_basestream_children(tmp_path: Path) -> None:
    pvc = tmp_path / "pvc"
    dataset = pvc / "dataset"
    dataset.mkdir(parents=True)
    _create_task(dataset, "cpu", gpus=0)
    _create_task(dataset, "gpu", gpus=1)
    bin_dir = _fake_uv(tmp_path)
    jobs = pvc / "jobs"
    env = os.environ.copy()
    env.update(
        {
            "HARBOR_NEXAU_ENV_FILE": str(tmp_path / "missing.env"),
            "HARBOR_NEXAU_MODEL": "gpt-5.5",
            "HARBOR_BASESTREAM_PVC_HOST_ROOT": str(pvc),
            "HARBOR_BASESTREAM_CONTROL_ROOT": str(pvc / "control"),
            "HARBOR_NEXAU_DIND_PREWARM": "false",
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    env.pop("_HARBOR_NEXAU_DOTENV_LOADED", None)

    result = subprocess.run(
        [
            str(_SCRIPT),
            "--path",
            str(dataset),
            "--jobs-dir",
            str(jobs),
            "--job-name",
            "tb4-test",
        ],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "1 CPU task(s), 1 H100 Basestream task(s)" in result.stderr
    invocations = [
        path.read_text(encoding="utf-8").splitlines()
        for path in (tmp_path / "invocations").iterdir()
    ]
    assert len(invocations) == 2
    by_environment = {invocation[0]: invocation[1:] for invocation in invocations}
    assert "ENV=nexau_harbor.e2b_environment:NexAUE2BEnvironment" in by_environment
    gpu_args = by_environment["ENV=nexau_harbor.basestream_environment:BasestreamEnvironment"]
    assert gpu_args[gpu_args.index("--n-tasks") + 1] == "1"
    assert gpu_args[gpu_args.index("--n-concurrent") + 1] == "1"
    assert gpu_args[gpu_args.index("--path") + 1].endswith("/views/basestream")
    plans = list((pvc / "control" / "plans").glob("*.json"))
    assert len(plans) == 1
    plan = json.loads(plans[0].read_text(encoding="utf-8"))
    assert plan["counts"] == {"basestream": 1, "cpu": 1}
    assert list((pvc / "control" / "routes").iterdir()) == []


def test_tb4_script_rejects_gpu_jobs_outside_pvc(tmp_path: Path) -> None:
    pvc = tmp_path / "pvc"
    dataset = pvc / "dataset"
    dataset.mkdir(parents=True)
    _create_task(dataset, "gpu", gpus=1)
    bin_dir = _fake_uv(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "HARBOR_NEXAU_ENV_FILE": str(tmp_path / "missing.env"),
            "HARBOR_NEXAU_MODEL": "gpt-5.5",
            "HARBOR_BASESTREAM_PVC_HOST_ROOT": str(pvc),
            "HARBOR_BASESTREAM_CONTROL_ROOT": str(pvc / "control"),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        [
            str(_SCRIPT),
            "--path",
            str(dataset),
            "--jobs-dir",
            str(tmp_path / "outside"),
        ],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "must be under the Basestream PVC host root" in result.stderr


def test_tb4_script_delegates_remote_gpfs_path_from_development_host(tmp_path: Path) -> None:
    capture = tmp_path / "submit-args.json"
    submit = tmp_path / "submit-tb4"
    submit.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

Path(os.environ["SUBMIT_CAPTURE"]).write_text(
    json.dumps(sys.argv[1:]),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    submit.chmod(0o755)
    remote_dataset = "/gpfs/remote-only/tb4/terminal-bench"
    env = os.environ.copy()
    env.update(
        {
            "HARBOR_NEXAU_ENV_FILE": str(tmp_path / "missing.env"),
            "HARBOR_NEXAU_TB4_SUBMIT_SCRIPT": str(submit),
            "HARBOR_BASESTREAM_PVC_HOST_ROOT": str(tmp_path / "missing-pvc"),
            "SUBMIT_CAPTURE": str(capture),
        }
    )

    arguments = [
        "--path",
        remote_dataset,
        "--trace-mode",
        "--job-name",
        "tb4-from-dev",
        "--yes",
    ]
    result = subprocess.run(
        [str(_SCRIPT), *arguments],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "Local Basestream/GPFS control plane is unavailable" in result.stderr
    assert json.loads(capture.read_text(encoding="utf-8")) == arguments
