from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "audit_harbor_job.py"


def _write_job(
    tmp_path: Path,
    trials: list[tuple[str, float | None, str | None]],
) -> Path:
    job = tmp_path / "job"
    job.mkdir()
    errors = sum(exception is not None for _, _, exception in trials)
    (job / "result.json").write_text(
        json.dumps(
            {
                "stats": {
                    "n_completed_trials": len(trials),
                    "n_errored_trials": errors,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    for index, (task, reward, exception) in enumerate(trials):
        trial = job / f"{task}__{index}"
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": f"terminal-bench/{task}",
                    "verifier_result": (
                        {"rewards": {"reward": reward}} if reward is not None else None
                    ),
                    "exception_info": (
                        {"exception_type": exception} if exception is not None else None
                    ),
                }
            ),
            encoding="utf-8",
        )
    return job


def _run(job: Path, *tasks: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(_SCRIPT),
        "--job-dir",
        str(job),
        "--expected-trials",
        str(len(tasks)),
    ]
    for task in tasks:
        command.extend(("--expected-task", task))
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_audit_accepts_exact_complete_rewarded_job(tmp_path: Path) -> None:
    job = _write_job(tmp_path, [("fp8", 1.0, None), ("jax", 0.0, None)])

    result = _run(job, "fp8", "jax")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "valid"
    assert payload["tasks"] == {"fp8": 1.0, "jax": 0.0}


def test_audit_rejects_trial_exception_even_when_reward_exists(tmp_path: Path) -> None:
    job = _write_job(tmp_path, [("fp8", 0.0, "NetworkConnectionError")])

    result = _run(job, "fp8")

    assert result.returncode == 1
    assert "n_errored_trials=1" in result.stderr
    assert "exception='NetworkConnectionError'" in result.stderr


def test_audit_rejects_missing_expected_task(tmp_path: Path) -> None:
    job = _write_job(tmp_path, [("math", 1.0, None)])

    result = _run(job, "jax")

    assert result.returncode == 1
    assert "task set mismatch" in result.stderr
