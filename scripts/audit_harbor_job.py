#!/usr/bin/env python3
"""Fail unless a Harbor job contains the expected fully valid trial results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def audit_job(
    job_dir: Path,
    *,
    expected_trials: int,
    expected_tasks: set[str],
) -> dict[str, Any]:
    if expected_trials < 1:
        raise ValueError("expected_trials must be positive")
    job_result = _load_json(job_dir / "result.json")
    stats = job_result.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("job result has no stats object")

    errors: list[str] = []
    expected_stats = {
        "n_completed_trials": expected_trials,
        "n_errored_trials": 0,
        "n_running_trials": 0,
        "n_pending_trials": 0,
        "n_cancelled_trials": 0,
    }
    for key, expected in expected_stats.items():
        if stats.get(key) != expected:
            errors.append(f"{key}={stats.get(key)!r}, expected {expected}")

    trial_paths = sorted(job_dir.glob("*/result.json"))
    if len(trial_paths) != expected_trials:
        errors.append(f"trial result files={len(trial_paths)}, expected {expected_trials}")

    tasks: dict[str, float] = {}
    for path in trial_paths:
        result = _load_json(path)
        task_name = result.get("task_name")
        if not isinstance(task_name, str) or not task_name:
            errors.append(f"{path.parent.name}: missing task_name")
            continue
        task = task_name.rsplit("/", 1)[-1]
        if task in tasks:
            errors.append(f"duplicate task result: {task}")
        exception = result.get("exception_info")
        if exception is not None:
            exception_type = (
                exception.get("exception_type") if isinstance(exception, dict) else exception
            )
            errors.append(f"{task}: exception={exception_type!r}")
        verifier = result.get("verifier_result")
        rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
        reward = rewards.get("reward") if isinstance(rewards, dict) else None
        if (
            isinstance(reward, bool)
            or not isinstance(reward, int | float)
            or not math.isfinite(float(reward))
        ):
            errors.append(f"{task}: missing finite verifier reward")
            continue
        tasks[task] = float(reward)

    actual_tasks = set(tasks)
    if expected_tasks and actual_tasks != expected_tasks:
        errors.append(
            "task set mismatch: "
            f"actual={sorted(actual_tasks)!r}, expected={sorted(expected_tasks)!r}"
        )
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "status": "valid",
        "job_dir": str(job_dir.resolve()),
        "trial_count": len(trial_paths),
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument("--expected-task", action="append", default=[])
    args = parser.parse_args()
    try:
        result = audit_job(
            args.job_dir,
            expected_trials=args.expected_trials,
            expected_tasks=set(args.expected_task),
        )
    except ValueError as exc:
        print(f"Harbor job audit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
