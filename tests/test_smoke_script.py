from __future__ import annotations

import os
import signal
import subprocess
import time
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
    invocation_capture_dir: Path | None = None,
    interrupt_when_children_ready: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
is_preflight=0
for argument in "$@"; do
  if [[ "$argument" == "nexau_harbor.e2b_preflight" ]]; then
    is_preflight=1
    break
  fi
done
if [[ "$is_preflight" == "1" ]]; then
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
  printf '{"tasks": []}\n' > "$output"
  : > "$selectors_output"
  for task in "$dataset"/*; do
    if [[ ! -d "$task" || ! -f "$task/task.toml" ]]; then
      continue
    fi
    resolved="$(realpath -e -- "$task")"
    route="plain-e2b"
    if [[ -f "$task/environment/docker-compose.yaml" || \
      -f "$task/environment/docker-compose.yml" || \
      -f "$task/environment/compose.yaml" || \
      -f "$task/environment/compose.yml" ]]; then
      route="dind"
    fi
    printf '%s\t%s\n' "$route" "${resolved##*/}" >> "$selectors_output"
  done
  exit 0
fi
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
if [[ -n "${INVOCATION_CAPTURE_DIR:-}" ]]; then
  printf "%s\\n" "$@" > "$INVOCATION_CAPTURE_DIR/$$"
fi
if [[ -n "${FORMAL_HOLD_DIR:-}" ]]; then
  printf '%s\n' "$$" > "$FORMAL_HOLD_DIR/$$"
  trap 'exit 0' INT TERM
  while true; do sleep 1; done
fi
printf "%s\\n" "$@"
"""
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HARBOR_NEXAU_ENV_FILE": str(tmp_path / "missing.env"),
            "HARBOR_NEXAU_MODEL": "gpt-5.5",
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    env.pop("_HARBOR_NEXAU_DOTENV_LOADED", None)
    for name in (
        "HARBOR_NEXAU_HARNESS",
        "HARBOR_CODEX_VERSION",
        "HARBOR_CODEX_UPSTREAM_MODEL",
        "HARBOR_CODEX_REASONING_EFFORT",
        "HARBOR_CODEX_REASONING_SUMMARY",
        "HARBOR_CODEX_UNIFIED_EXEC",
        "HARBOR_CODEX_CODE_MODE",
        "HARBOR_CODEX_CODE_MODE_ONLY",
        "HARBOR_CODEX_MULTI_AGENT",
        "HARBOR_CODEX_RESPONSES_BRIDGE",
        "HARBOR_CODEX_CONTEXT_WINDOW",
        "HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER",
    ):
        env.pop(name, None)
    if extra_env is not None:
        env.update(extra_env)
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
    if invocation_capture_dir is not None:
        invocation_capture_dir.mkdir()
        env["INVOCATION_CAPTURE_DIR"] = str(invocation_capture_dir)
    else:
        env.pop("INVOCATION_CAPTURE_DIR", None)

    command = [str(_SCRIPT), *arguments]
    if interrupt_when_children_ready:
        hold_dir = tmp_path / "formal-hold"
        hold_dir.mkdir()
        env["FORMAL_HOLD_DIR"] = str(hold_dir)
        process = subprocess.Popen(
            command,
            cwd=_PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 5
        while len(tuple(hold_dir.iterdir())) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        if len(tuple(hold_dir.iterdir())) < 2:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            raise AssertionError(f"auto-partition children did not start:\n{stdout}\n{stderr}")
        child_pids = [int(path.read_text()) for path in hold_dir.iterdir()]
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
        for child_pid in child_pids:
            assert not Path(f"/proc/{child_pid}").exists()
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    return subprocess.run(
        command,
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _create_auto_partition_dataset(tmp_path: Path) -> Path:
    task_store = tmp_path / "task-store"
    task_store.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    for task_name in ("plain-digest-a", "plain-digest-b", "compose-digest"):
        task = task_store / task_name
        task.mkdir()
        (task / "task.toml").write_text('version = "1.0"\n')

    compose_environment = task_store / "compose-digest" / "environment"
    compose_environment.mkdir()
    (compose_environment / "docker-compose.yaml").write_text("services: {}\n")

    (dataset / "plain-human-a").symlink_to(task_store / "plain-digest-a", target_is_directory=True)
    (dataset / "plain-human-b").symlink_to(task_store / "plain-digest-b", target_is_directory=True)
    (dataset / "compose-human").symlink_to(task_store / "compose-digest", target_is_directory=True)
    return dataset


def _last_option_value(arguments: list[str], option: str) -> str:
    option_index = max(index for index, argument in enumerate(arguments) if argument == option)
    return arguments[option_index + 1]


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
        "    model_name: 'gpt-5.5'\n"
        "    override_timeout_sec: 36000\n"
        "    override_setup_timeout_sec: 3600\n"
        "    kwargs:\n"
        f"      config_path: '{_DEFAULT_TB_CONFIG}'\n"
    )
    assert "replaced with 36000 seconds" in result.stderr


def test_smoke_script_trace_mode_accepts_explicit_timeout(tmp_path: Path) -> None:
    trace_config_capture = tmp_path / "trace-job.yaml"

    result = _run_with_fake_uv(
        tmp_path,
        "--trace-timeout-sec=3600",
        trace_config_capture=trace_config_capture,
    )

    assert result.returncode == 0, result.stderr
    assert "override_timeout_sec: 3600" in trace_config_capture.read_text()


def test_smoke_script_trace_mode_selects_trace_safe_codex(tmp_path: Path) -> None:
    trace_config_capture = tmp_path / "trace-job.yaml"

    result = _run_with_fake_uv(
        tmp_path,
        "--trace-mode",
        "--harness",
        "codex",
        trace_config_capture=trace_config_capture,
    )

    assert result.returncode == 0, result.stderr
    assert trace_config_capture.read_text() == (
        "agents:\n"
        "  - name: 'nexau_harbor.codex_agent:TraceSafeCodexAgent'\n"
        "    model_name: 'gpt-5.5'\n"
        "    override_timeout_sec: 36000\n"
        "    override_setup_timeout_sec: 3600\n"
        "    kwargs:\n"
        "      version: '0.148.0'\n"
        "      unified_exec: true\n"
        "      code_mode: false\n"
        "      code_mode_only: false\n"
        "      multi_agent: false\n"
        "      responses_bridge: true\n"
        "      bridge_port: 48123\n"
        "      provider_name: 'northgate'\n"
        "      upstream_model_name: 'nex-agi/Nex-N2-Pro'\n"
        "      context_window: 262144\n"
        "      reasoning_effort: 'high'\n"
        "      reasoning_summary: 'detailed'\n"
    )
    assert "Trace mode (codex harness)" in result.stderr
    arguments = result.stdout.splitlines()
    timeout_index = arguments.index("--environment-build-timeout-multiplier")
    assert arguments[timeout_index + 1] == "4"


def test_smoke_script_trace_mode_supports_native_terra_code_mode(tmp_path: Path) -> None:
    trace_config_capture = tmp_path / "trace-job.yaml"

    result = _run_with_fake_uv(
        tmp_path,
        "--trace-mode",
        "--harness",
        "codex",
        trace_config_capture=trace_config_capture,
        extra_env={
            "HARBOR_NEXAU_MODEL": "gpt-5.6-terra",
            "HARBOR_CODEX_UNIFIED_EXEC": "false",
            "HARBOR_CODEX_CODE_MODE": "true",
            "HARBOR_CODEX_CODE_MODE_ONLY": "true",
            "HARBOR_CODEX_MULTI_AGENT": "false",
            "HARBOR_CODEX_RESPONSES_BRIDGE": "false",
            "HARBOR_CODEX_CONTEXT_WINDOW": "272000",
            "HARBOR_CODEX_REASONING_EFFORT": "max",
            "HARBOR_CODEX_UPSTREAM_MODEL": "must-not-be-used",
        },
    )

    assert result.returncode == 0, result.stderr
    assert trace_config_capture.read_text() == (
        "agents:\n"
        "  - name: 'nexau_harbor.codex_agent:TraceSafeCodexAgent'\n"
        "    model_name: 'gpt-5.6-terra'\n"
        "    override_timeout_sec: 36000\n"
        "    override_setup_timeout_sec: 3600\n"
        "    kwargs:\n"
        "      version: '0.148.0'\n"
        "      unified_exec: false\n"
        "      code_mode: true\n"
        "      code_mode_only: true\n"
        "      multi_agent: false\n"
        "      responses_bridge: false\n"
        "      bridge_port: 48123\n"
        "      provider_name: 'northgate'\n"
        "      context_window: 272000\n"
        "      reasoning_effort: 'max'\n"
        "      reasoning_summary: 'detailed'\n"
    )
    assert "must-not-be-used" not in trace_config_capture.read_text()


def test_smoke_script_non_trace_mode_selects_trace_safe_codex(tmp_path: Path) -> None:
    result = _run_with_fake_uv(tmp_path, "--harness=codex")

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    agent_index = arguments.index("--agent")
    assert arguments[agent_index + 1] == "nexau_harbor.codex_agent:TraceSafeCodexAgent"
    assert "version=0.148.0" in arguments
    assert "unified_exec=true" in arguments
    assert "code_mode=false" in arguments
    assert "code_mode_only=false" in arguments
    assert "multi_agent=false" in arguments
    assert "responses_bridge=true" in arguments
    assert "bridge_port=48123" in arguments
    assert "provider_name=northgate" in arguments
    assert "upstream_model_name=nex-agi/Nex-N2-Pro" in arguments
    assert "context_window=262144" in arguments
    assert "reasoning_effort=high" in arguments
    assert "reasoning_summary=detailed" in arguments
    timeout_index = arguments.index("--environment-build-timeout-multiplier")
    assert arguments[timeout_index + 1] == "4"


def test_smoke_script_preserves_explicit_environment_timeout_multiplier(
    tmp_path: Path,
) -> None:
    result = _run_with_fake_uv(
        tmp_path,
        "--environment-build-timeout-multiplier",
        "2",
    )

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert arguments.count("--environment-build-timeout-multiplier") == 1
    timeout_index = arguments.index("--environment-build-timeout-multiplier")
    assert arguments[timeout_index + 1] == "2"


def test_smoke_script_rejects_unknown_harness(tmp_path: Path) -> None:
    result = _run_with_fake_uv(tmp_path, "--harness", "unknown")

    assert result.returncode == 2
    assert "Harness must be nexau or codex" in result.stderr


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
    assert "use HARBOR_NEXAU_MODEL, --harness, and --trace-timeout-sec" in result.stderr


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


def test_smoke_script_auto_partitions_local_dataset_into_concurrent_jobs(
    tmp_path: Path,
) -> None:
    dataset = _create_auto_partition_dataset(tmp_path)
    invocation_capture_dir = tmp_path / "invocations"

    result = _run_with_fake_uv(
        tmp_path,
        "--auto-partition",
        "--trace-mode",
        "--path",
        str(dataset),
        "--job-name",
        "auto-run",
        invocation_capture_dir=invocation_capture_dir,
    )

    assert result.returncode == 0, result.stderr
    invocations = [path.read_text().splitlines() for path in invocation_capture_dir.iterdir()]
    assert len(invocations) == 2
    assert all("--auto-partition" not in arguments for arguments in invocations)
    assert all("--config" in arguments for arguments in invocations)
    assert all(arguments.count("--job-name") == 1 for arguments in invocations)

    plain_arguments = next(
        arguments
        for arguments in invocations
        if "nexau_harbor.e2b_environment:NexAUE2BEnvironment" in arguments
    )
    dind_arguments = next(
        arguments
        for arguments in invocations
        if "nexau_harbor.e2b_dind_environment:NexAUE2BDinDEnvironment" in arguments
    )

    assert _last_option_value(plain_arguments, "--job-name") == "auto-run-plain-e2b"
    assert _last_option_value(plain_arguments, "--n-tasks") == "2"
    assert _last_option_value(plain_arguments, "--n-concurrent") == "2"
    assert plain_arguments.count("--include-task-name") == 2
    assert "plain-digest-a" in plain_arguments
    assert "plain-digest-b" in plain_arguments
    assert "--exclude-task-name" not in plain_arguments

    assert _last_option_value(dind_arguments, "--job-name") == "auto-run-dind"
    assert _last_option_value(dind_arguments, "--n-tasks") == "1"
    assert _last_option_value(dind_arguments, "--n-concurrent") == "1"
    assert "--exclude-task-name" not in dind_arguments
    assert dind_arguments[dind_arguments.index("--include-task-name") + 1] == "compose-digest"
    assert "Auto partition selected 2 plain E2B and 1 DinD task(s)" in result.stderr


def test_auto_partition_interrupt_stops_both_process_groups(tmp_path: Path) -> None:
    dataset = _create_auto_partition_dataset(tmp_path)

    result = _run_with_fake_uv(
        tmp_path,
        "--auto-partition",
        "--path",
        str(dataset),
        "--job-name",
        "auto-interrupt",
        interrupt_when_children_ready=True,
    )

    assert result.returncode == 130


def test_smoke_script_auto_partition_requires_local_dataset_path(tmp_path: Path) -> None:
    result = _run_with_fake_uv(tmp_path, "--auto-partition")

    assert result.returncode == 2
    assert "requires --path to a resolved local dataset" in result.stderr


def test_smoke_script_auto_partition_owns_concurrency(tmp_path: Path) -> None:
    dataset = _create_auto_partition_dataset(tmp_path)

    result = _run_with_fake_uv(
        tmp_path,
        "--auto-partition",
        "--path",
        str(dataset),
        "--n-concurrent",
        "3",
    )

    assert result.returncode == 2
    assert "sets --n-tasks and --n-concurrent" in result.stderr


def test_smoke_script_rejects_multiple_task_partitions(tmp_path: Path) -> None:
    result = _run_with_fake_uv(
        tmp_path,
        "--auto-partition",
        "--compose-only",
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
    assert "partition modes own --env and task-name filters" in result.stderr
