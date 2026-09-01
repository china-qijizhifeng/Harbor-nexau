from __future__ import annotations

import logging
import os
import stat
from functools import wraps
from pathlib import Path
from typing import Any

from harbor.constants import PACKAGE_CACHE_DIR
from harbor.tasks.client import TaskClient

_PACKAGE_DIRECTORY_MODE = 0o755
_PATCH_MARKER = "_nexau_normalizes_package_directory_permissions"

logger = logging.getLogger(__name__)


def normalize_package_task_directory_permissions(
    task_path: Path | str,
    *,
    package_cache_dir: Path | str = PACKAGE_CACHE_DIR,
) -> int:
    """Make registry-package directories deterministic without touching files.

    Harbor 0.21 package archives contain file entries but no directory entries.
    ``tarfile`` therefore creates their parent directories using the process
    umask. Under a restrictive ``0077`` umask, a separate verifier running as a
    non-root user cannot traverse an otherwise public task tree.

    Only resolved paths inside Harbor's package cache are eligible. Symlinked
    directories inside a package are not followed, and file modes are preserved.
    """

    resolved_task = Path(task_path).resolve(strict=True)
    resolved_cache = Path(package_cache_dir).resolve()
    if resolved_task == resolved_cache or not resolved_task.is_relative_to(resolved_cache):
        return 0

    changed = 0
    for root, directory_names, _file_names in os.walk(
        resolved_task,
        topdown=True,
        followlinks=False,
    ):
        root_path = Path(root)
        directory_names[:] = [
            name for name in directory_names if not (root_path / name).is_symlink()
        ]
        current_mode = stat.S_IMODE(root_path.stat().st_mode)
        if current_mode == _PACKAGE_DIRECTORY_MODE:
            continue
        root_path.chmod(_PACKAGE_DIRECTORY_MODE)
        changed += 1

    return changed


def install_harbor_task_directory_permissions_patch(
    *,
    task_client_class: type[Any] = TaskClient,
    package_cache_dir: Path | str = PACKAGE_CACHE_DIR,
) -> bool:
    """Normalize package task trees after every Harbor task resolution.

    Patching the public ``download_tasks`` boundary covers fresh registry
    downloads, cache hits, and local-dataset symlinks that resolve into the
    registry package cache. The patch is process-local and idempotent.
    """

    original = task_client_class.download_tasks
    if getattr(original, _PATCH_MARKER, False):
        return False

    @wraps(original)
    async def download_tasks_with_safe_package_directories(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = await original(self, *args, **kwargs)
        changed = sum(
            normalize_package_task_directory_permissions(
                item.path,
                package_cache_dir=package_cache_dir,
            )
            for item in result.results
        )
        if changed:
            logger.info(
                "Normalized %s Harbor package-cache directories to mode 0755",
                changed,
            )
        return result

    setattr(download_tasks_with_safe_package_directories, _PATCH_MARKER, True)
    task_client_class.download_tasks = download_tasks_with_safe_package_directories
    return True
