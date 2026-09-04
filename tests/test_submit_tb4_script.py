from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "submit_tb4.sh"


def _fake_ssh(tmp_path: Path) -> tuple[Path, Path]:
    capture = tmp_path / "ssh-capture.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "ssh"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

Path(os.environ["SSH_CAPTURE"]).write_text(
    json.dumps({{"argv": sys.argv[1:], "stdin": sys.stdin.read()}}),
    encoding="utf-8",
)
print(json.dumps({{"status": "running", "task_id": "fake-monitor"}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir, capture


def test_submit_tb4_starts_remote_monitor_without_local_dataset(tmp_path: Path) -> None:
    bin_dir, capture = _fake_ssh(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SSH_CAPTURE": str(capture),
            "CODEX_THREAD_ID": "00000000-0000-0000-0000-000000000001",
            "HARBOR_NEXAU_TB4_SUBMIT_HOST": "test-basestream",
            "HARBOR_NEXAU_TB4_REMOTE_PROJECT_ROOT": "/remote/project",
            "HARBOR_NEXAU_TB4_REMOTE_RUNTIME_ROOT": "/remote/runtime",
            "HARBOR_NEXAU_TB4_REMOTE_CODEX_HOME": "/remote/codex",
            "HARBOR_NEXAU_TB4_MONITOR_TIMEOUT_MINUTES": "99",
        }
    )

    result = subprocess.run(
        [
            str(_SCRIPT),
            "--path",
            "/gpfs/dataset with spaces",
            "--trace-mode",
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
    assert '"status": "running"' in result.stdout
    submission = json.loads(capture.read_text(encoding="utf-8"))
    assert submission["argv"][-2] == "test-basestream"
    remote_arguments = shlex.split(submission["argv"][-1])
    assert remote_arguments[:3] == ["bash", "-s", "--"]
    assert remote_arguments[3:10] == [
        "/remote/project",
        "/remote/runtime",
        "/remote/codex",
        "99",
        "tb4-test",
        "00000000-0000-0000-0000-000000000001",
        "/gpfs/dataset with spaces",
    ]
    delimiter = remote_arguments.index("--", 3)
    forwarded = remote_arguments[delimiter + 1 :]
    assert forwarded == [
        "--path",
        "/gpfs/dataset with spaces",
        "--trace-mode",
        "--job-name",
        "tb4-test",
        "--yes",
    ]
    assert "long_task_monitor.py" in submission["stdin"]
    assert "HARBOR_BASESTREAM_CONTROL_ROOT" in submission["stdin"]
    assert "already active in monitor" in submission["stdin"]


def test_submit_tb4_requires_exactly_one_remote_path(tmp_path: Path) -> None:
    bin_dir, capture = _fake_ssh(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SSH_CAPTURE": str(capture),
        }
    )

    result = subprocess.run(
        [str(_SCRIPT), "--trace-mode"],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "requires exactly one remote --path" in result.stderr
    assert not capture.exists()


def test_submit_tb4_preserves_explicit_yes(tmp_path: Path) -> None:
    bin_dir, capture = _fake_ssh(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SSH_CAPTURE": str(capture),
        }
    )

    result = subprocess.run(
        [str(_SCRIPT), "--path=/gpfs/dataset", "--job-name=tb4-y", "-y"],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    submission = json.loads(capture.read_text(encoding="utf-8"))
    forwarded = shlex.split(submission["argv"][-1])
    assert forwarded.count("-y") == 1
    assert "--yes" not in forwarded
