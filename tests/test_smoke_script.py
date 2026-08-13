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
    tmp_path: Path, *arguments: str, config_path: Path | None = None
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
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
