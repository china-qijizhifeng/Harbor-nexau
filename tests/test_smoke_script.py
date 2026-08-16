from __future__ import annotations

import os
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "smoke_tb3.sh"
_DEFAULT_TB_CONFIG = (
    _PROJECT_ROOT / "src" / "nexau_harbor" / "configs" / "terminal_bench" / "code_agent.yaml"
)


def _run_with_fake_uv(
    tmp_path: Path,
    *arguments: str,
    config_path: Path | None = None,
    trace_config_capture: Path | None = None,
    ramp_capture: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
previous=""
for argument in "$@"; do
  if [[ "$previous" == "--config" && -n "${TRACE_CONFIG_CAPTURE:-}" ]]; then
    cp -- "$argument" "$TRACE_CONFIG_CAPTURE"
  fi
  previous="$argument"
done
if [[ -n "${RAMP_CAPTURE:-}" ]]; then
  printf '%s' "${HARBOR_NEXAU_E2B_START_RAMP_DELAY_SEC:-}" > "$RAMP_CAPTURE"
fi
printf "%s\\n" "$@"
"""
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HARBOR_NEXAU_ENV_FILE": str(tmp_path / "missing.env"),
            "HARBOR_NEXAU_MODEL": "openai/nex-agi/Nex-N2-Pro",
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    env.pop("_HARBOR_NEXAU_DOTENV_LOADED", None)
    if config_path is not None:
        env["HARBOR_NEXAU_CONFIG_PATH"] = str(config_path)
    else:
        env.pop("HARBOR_NEXAU_CONFIG_PATH", None)
    if trace_config_capture is not None:
        env["TRACE_CONFIG_CAPTURE"] = str(trace_config_capture)
    else:
        env.pop("TRACE_CONFIG_CAPTURE", None)
    if ramp_capture is not None:
        env["RAMP_CAPTURE"] = str(ramp_capture)
    else:
        env.pop("RAMP_CAPTURE", None)

    return subprocess.run(
        [str(_SCRIPT), *arguments],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_smoke_script_selects_terminal_bench_config_by_default(
    tmp_path: Path,
) -> None:
    result = _run_with_fake_uv(
        tmp_path,
        "--path",
        "/tmp/local-task",
        "--job-name",
        "n2-tb-smoke",
    )

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    config_index = arguments.index("--agent-kwarg")
    assert arguments[config_index + 1] == f"config_path={_DEFAULT_TB_CONFIG}"
    assert "--dataset" not in arguments
    assert arguments[-4:] == [
        "--path",
        "/tmp/local-task",
        "--job-name",
        "n2-tb-smoke",
    ]


def test_smoke_script_allows_explicit_harness_config(tmp_path: Path) -> None:
    custom_config = tmp_path / "custom-agent.yaml"
    custom_config.write_text("type: agent\n")

    result = _run_with_fake_uv(tmp_path, config_path=custom_config)

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    config_index = arguments.index("--agent-kwarg")
    assert arguments[config_index + 1] == f"config_path={custom_config}"


def test_smoke_script_trace_mode_overrides_task_timeout(tmp_path: Path) -> None:
    trace_config_capture = tmp_path / "trace-job.yaml"

    result = _run_with_fake_uv(
        tmp_path,
        "--trace-mode",
        "--path",
        "/tmp/local-task",
        "--job-name",
        "n2-trace",
        trace_config_capture=trace_config_capture,
    )

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert "--trace-mode" not in arguments
    assert "--agent" not in arguments
    assert "--agent-kwarg" not in arguments
    assert "--model" not in arguments
    config_index = arguments.index("--config")
    assert arguments[config_index + 1].startswith("/tmp/harbor-nexau-trace.")
    assert trace_config_capture.read_text() == (
        "agents:\n"
        "  - name: 'nexau_harbor.agent:NexAUAgent'\n"
        "    model_name: 'openai/nex-agi/Nex-N2-Pro'\n"
        "    override_timeout_sec: 72000\n"
        "    kwargs:\n"
        f"      config_path: '{_DEFAULT_TB_CONFIG}'\n"
    )
    assert "replaced with 72000 seconds" in result.stderr


def test_smoke_script_trace_mode_accepts_explicit_timeout(tmp_path: Path) -> None:
    trace_config_capture = tmp_path / "trace-job.yaml"

    result = _run_with_fake_uv(
        tmp_path,
        "--trace-timeout-sec=3600",
        trace_config_capture=trace_config_capture,
    )

    assert result.returncode == 0, result.stderr
    assert "override_timeout_sec: 3600" in trace_config_capture.read_text()


def test_smoke_script_trace_mode_rejects_agent_override(tmp_path: Path) -> None:
    result = _run_with_fake_uv(
        tmp_path,
        "--trace-mode",
        "--model",
        "openai/another-model",
    )

    assert result.returncode == 2
    assert "Trace mode owns the Harbor agent and timeout config" in result.stderr


def test_smoke_script_trace_mode_rejects_timeout_multiplier(tmp_path: Path) -> None:
    result = _run_with_fake_uv(
        tmp_path,
        "--trace-mode",
        "--agent-timeout-multiplier=2",
    )

    assert result.returncode == 2
    assert "use HARBOR_NEXAU_MODEL and --trace-timeout-sec" in result.stderr


def test_smoke_script_exports_and_consumes_start_ramp_delay(tmp_path: Path) -> None:
    ramp_capture = tmp_path / "ramp.txt"

    result = _run_with_fake_uv(
        tmp_path,
        "--e2b-start-ramp-delay-sec",
        "5",
        "--path",
        "/tmp/local-task",
        ramp_capture=ramp_capture,
    )

    assert result.returncode == 0, result.stderr
    assert ramp_capture.read_text() == "5"
    assert "--e2b-start-ramp-delay-sec" not in result.stdout.splitlines()
    assert "Harbor concurrency is unchanged" in result.stderr


def test_smoke_script_rejects_invalid_start_ramp_delay(tmp_path: Path) -> None:
    result = _run_with_fake_uv(
        tmp_path,
        "--e2b-start-ramp-delay-sec=-1",
    )

    assert result.returncode == 2
    assert "finite non-negative number" in result.stderr


def test_smoke_script_selects_compose_partition_and_dind_environment(
    tmp_path: Path,
) -> None:
    result = _run_with_fake_uv(
        tmp_path,
        "--compose-only",
        "--path",
        "/tmp/frontier-bench-cpu68",
    )

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    environment_index = arguments.index("--env")
    assert (
        arguments[environment_index + 1]
        == "nexau_harbor.e2b_dind_environment:NexAUE2BDinDEnvironment"
    )
    assert arguments.count("--include-task-name") == 33
    assert "--exclude-task-name" not in arguments
    assert "ctr-optimization" in arguments
    assert "*/ctr-optimization" in arguments
    assert "bc62e2e43cf5c91f63442cd787d82fc4765461c0b4fd06cab76ac4322c13dbff" in arguments
    assert "--compose-only" not in arguments
    assert "TB3 partition: compose tasks" in result.stderr


def test_smoke_script_selects_non_compose_partition_and_plain_e2b_environment(
    tmp_path: Path,
) -> None:
    result = _run_with_fake_uv(
        tmp_path,
        "--non-compose-only",
        "--path",
        "/tmp/frontier-bench-cpu68",
    )

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    environment_index = arguments.index("--env")
    assert arguments[environment_index + 1] == "nexau_harbor.e2b_environment:NexAUE2BEnvironment"
    assert arguments.count("--exclude-task-name") == 33
    assert "--include-task-name" not in arguments
    assert "nextjs-performance" in arguments
    assert "*/nextjs-performance" in arguments
    assert "7bea6536709f9c63cb7917804dfdb4210086bf9a23b26c0afc2b70ba254a4232" in arguments
    assert "--non-compose-only" not in arguments
    assert "TB3 partition: non-compose tasks" in result.stderr


def test_smoke_script_rejects_multiple_task_partitions(tmp_path: Path) -> None:
    result = _run_with_fake_uv(
        tmp_path,
        "--compose-only",
        "--non-compose-only",
    )

    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


def test_smoke_script_partition_rejects_explicit_environment(tmp_path: Path) -> None:
    result = _run_with_fake_uv(
        tmp_path,
        "--non-compose-only",
        "--env",
        "docker",
    )

    assert result.returncode == 2
    assert "partition flags own --env and task-name filters" in result.stderr
