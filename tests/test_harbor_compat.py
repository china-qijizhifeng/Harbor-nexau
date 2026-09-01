from __future__ import annotations

import io
import os
import stat
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexau_harbor.harbor_compat import (
    extract_tar_directory_from_file,
    install_harbor_task_directory_permissions_patch,
    normalize_package_task_directory_permissions,
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _extract_file_only_package(task_dir: Path) -> Path:
    task_dir.parent.mkdir(parents=True)
    archive = task_dir.parent / "task.tar.gz"
    payload = b"dependency==1.0\n"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("environment/src/worker/requirements.txt")
        member.mode = 0o644
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))

    previous_umask = os.umask(0o077)
    try:
        task_dir.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(task_dir, filter="data")
    finally:
        os.umask(previous_umask)

    return task_dir / "environment/src/worker/requirements.txt"


def test_normalizes_implicit_package_directories_under_restrictive_umask(
    tmp_path: Path,
) -> None:
    package_cache = tmp_path / "packages"
    package_cache.mkdir()
    task_dir = package_cache / "org/task/digest"
    requirements = _extract_file_only_package(task_dir)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (task_dir / "outside-link").symlink_to(outside, target_is_directory=True)

    assert _mode(task_dir / "environment/src/worker") == 0o700

    changed = normalize_package_task_directory_permissions(
        task_dir,
        package_cache_dir=package_cache,
    )

    assert changed == 4
    assert _mode(task_dir) == 0o755
    assert _mode(task_dir / "environment") == 0o755
    assert _mode(task_dir / "environment/src") == 0o755
    assert _mode(task_dir / "environment/src/worker") == 0o755
    assert _mode(requirements) == 0o644
    assert _mode(outside) == 0o700


def test_ignores_tasks_outside_package_cache(tmp_path: Path) -> None:
    package_cache = tmp_path / "packages"
    package_cache.mkdir()
    local_task = tmp_path / "local-task"
    local_task.mkdir(mode=0o700)

    changed = normalize_package_task_directory_permissions(
        local_task,
        package_cache_dir=package_cache,
    )

    assert changed == 0
    assert _mode(local_task) == 0o700


async def test_task_client_patch_repairs_local_dataset_symlink_cache_hits(
    tmp_path: Path,
) -> None:
    package_cache = tmp_path / "packages"
    package_cache.mkdir()
    task_dir = package_cache / "org/task/digest"
    _extract_file_only_package(task_dir)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    task_link = dataset / "task"
    task_link.symlink_to(task_dir, target_is_directory=True)

    class FakeTaskClient:
        async def download_tasks(self) -> SimpleNamespace:
            return SimpleNamespace(results=[SimpleNamespace(path=task_link)])

    assert install_harbor_task_directory_permissions_patch(
        task_client_class=FakeTaskClient,
        package_cache_dir=package_cache,
    )
    assert not install_harbor_task_directory_permissions_patch(
        task_client_class=FakeTaskClient,
        package_cache_dir=package_cache,
    )

    result = await FakeTaskClient().download_tasks()

    assert result.results[0].path == task_link
    assert _mode(task_dir / "environment/src/worker") == 0o755


def test_safe_artifact_extraction_preserves_directory_modes_under_umask(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source.chmod(0o755)
    public = source / "public"
    public.mkdir()
    public.chmod(0o755)
    private = source / "private"
    private.mkdir()
    private.chmod(0o700)
    requirements = public / "requirements.txt"
    requirements.write_text("dependency==1.0\n")
    requirements.chmod(0o644)
    archive_path = tmp_path / "artifact.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname=".")

    target = tmp_path / "target"
    previous_umask = os.umask(0o077)
    try:
        target.mkdir()
        extract_tar_directory_from_file(archive_path, target)
    finally:
        os.umask(previous_umask)

    assert _mode(target) == 0o755
    assert _mode(target / "public") == 0o755
    assert _mode(target / "private") == 0o700
    assert _mode(target / "public/requirements.txt") == 0o644


def test_safe_artifact_extraction_keeps_data_filter_path_checks(tmp_path: Path) -> None:
    archive_path = tmp_path / "escape.tar.gz"
    payload = b"escape"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(tarfile.OutsideDestinationError):
        extract_tar_directory_from_file(archive_path, tmp_path / "target")

    assert not (tmp_path / "outside.txt").exists()
