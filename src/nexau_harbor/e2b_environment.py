"""E2B compatibility environment for Harbor 0.21.x."""

import asyncio
import os
import re
from typing import override

from harbor.environments import e2b as harbor_e2b
from harbor.models.task.config import NetworkMode
from tenacity import retry, stop_after_attempt, wait_exponential

E2B_SANDBOX_TIMEOUT_SEC = 3_600
E2B_KEEPALIVE_INTERVAL_SEC = 60
E2B_DEFAULT_MAX_LIFETIME_SEC = 24 * 60 * 60
E2B_MAX_LIFETIME_ENV = "HARBOR_NEXAU_E2B_MAX_LIFETIME_SEC"

_DOCKERFILE_FROM_PATTERN = re.compile(
    r"^[ \t]*FROM(?:[ \t]+--platform=\S+)?[ \t]+(?P<image>\S+)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_ASTRAL_UV_COPY_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)COPY[ \t]+--from=ghcr\.io/astral-sh/uv:"
    r"(?P<version>[0-9][0-9A-Za-z.+-]*)[ \t]+/uv[ \t]+/uvx[ \t]+"
    r"/usr/local/bin/[ \t]*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _is_python_image(image: str) -> bool:
    normalized = image.lower()
    return normalized == "python" or normalized.startswith(
        ("python:", "python@", "docker.io/library/python:", "docker.io/library/python@")
    )


def _rewrite_e2b_incompatible_uv_copy(dockerfile: str) -> tuple[str, int]:
    """Replace the uv image copy form that E2B's Dockerfile parser rejects.

    E2B 2.x rejects absolute sources in ``COPY --from``, even though Docker accepts
    them. For Python-based stages, installing the same pinned uv package exposes both
    ``uv`` and ``uvx`` in /usr/local/bin without changing the source task on disk.
    """

    rewrite_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal rewrite_count

        preceding_from = tuple(_DOCKERFILE_FROM_PATTERN.finditer(dockerfile, 0, match.start()))
        if not preceding_from or not _is_python_image(preceding_from[-1].group("image")):
            return match.group(0)

        rewrite_count += 1
        indent = match.group("indent")
        version = match.group("version")
        return f"{indent}RUN python -m pip install --no-cache-dir uv=={version}"

    return _ASTRAL_UV_COPY_PATTERN.sub(replace, dockerfile), rewrite_count


def _resolve_max_lifetime_sec(configured: int | str | None) -> int:
    raw_value: object = configured
    if raw_value is None:
        raw_value = os.environ.get(
            E2B_MAX_LIFETIME_ENV,
            str(E2B_DEFAULT_MAX_LIFETIME_SEC),
        )

    if isinstance(raw_value, bool) or not isinstance(raw_value, int | str):
        raise ValueError(f"{E2B_MAX_LIFETIME_ENV} must be a positive integer")

    try:
        max_lifetime_sec = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{E2B_MAX_LIFETIME_ENV} must be a positive integer, got {raw_value!r}"
        ) from exc

    if max_lifetime_sec <= 0:
        raise ValueError(f"{E2B_MAX_LIFETIME_ENV} must be positive, got {max_lifetime_sec}")
    return max_lifetime_sec


def _bounded_lease_timeout(remaining_lifetime_sec: float) -> int | None:
    remaining_whole_seconds = int(remaining_lifetime_sec)
    if remaining_whole_seconds <= 0:
        return None
    return min(E2B_SANDBOX_TIMEOUT_SEC, remaining_whole_seconds)


class NexAUE2BEnvironment(harbor_e2b.E2BEnvironment):
    """Use a bounded renewable lease without exceeding the E2B plan limit."""

    def __init__(
        self,
        *args,
        max_lifetime_sec: int | str | None = None,
        **kwargs,
    ) -> None:
        self._max_lifetime_sec = _resolve_max_lifetime_sec(max_lifetime_sec)
        self._keepalive_task: asyncio.Task[None] | None = None
        self._keepalive_deadline: float | None = None
        super().__init__(*args, **kwargs)

    async def _keepalive_loop(self) -> None:
        while True:
            sandbox = self._sandbox
            if sandbox is None:
                return

            loop = asyncio.get_running_loop()
            if self._keepalive_deadline is None:
                return
            remaining_lifetime_sec = self._keepalive_deadline - loop.time()
            lease_timeout_sec = _bounded_lease_timeout(remaining_lifetime_sec)
            if lease_timeout_sec is None:
                self.logger.warning(
                    "E2B sandbox %s reached its configured maximum lifetime of %s "
                    "seconds; lease renewal stopped",
                    sandbox.sandbox_id,
                    self._max_lifetime_sec,
                )
                return

            try:
                await sandbox.set_timeout(lease_timeout_sec)
                self.logger.debug(
                    "Refreshed E2B sandbox %s timeout to %s seconds (maximum lifetime %s seconds)",
                    sandbox.sandbox_id,
                    lease_timeout_sec,
                    self._max_lifetime_sec,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.warning(
                    "Failed to refresh E2B sandbox timeout; will retry in %s seconds: %s",
                    E2B_KEEPALIVE_INTERVAL_SEC,
                    exc,
                )

            await asyncio.sleep(min(E2B_KEEPALIVE_INTERVAL_SEC, remaining_lifetime_sec))

    def _start_keepalive(self) -> None:
        keepalive_task = getattr(self, "_keepalive_task", None)
        if keepalive_task is not None and not keepalive_task.done():
            return

        self._keepalive_deadline = asyncio.get_running_loop().time() + self._max_lifetime_sec
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(),
            name=f"e2b-keepalive-{self.session_id}",
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    @override
    async def _create_template(self) -> None:
        if self.task_env_config.docker_image:
            template = harbor_e2b.Template().from_image(
                image=self.task_env_config.docker_image,
            )
        else:
            dockerfile = self._environment_definition_path.read_text(encoding="utf-8")
            dockerfile, rewrite_count = _rewrite_e2b_incompatible_uv_copy(dockerfile)
            if rewrite_count:
                self.logger.info(
                    "Applied %s E2B compatibility rewrite(s) for Astral uv",
                    rewrite_count,
                )

            template = harbor_e2b.Template(
                file_context_path=str(self.environment_dir),
            ).from_dockerfile(
                dockerfile_content_or_path=dockerfile,
            )

        build_kwargs = {
            "template": template,
            "alias": self._template_name,
        }
        if self._effective_cpus is not None:
            build_kwargs["cpu_count"] = self._effective_cpus
        if self._effective_memory_mb is not None:
            build_kwargs["memory_mb"] = self._effective_memory_mb

        await harbor_e2b.AsyncTemplate.build(**build_kwargs)

    async def _stop_keepalive(self) -> None:
        keepalive_task = getattr(self, "_keepalive_task", None)
        self._keepalive_task = None
        if keepalive_task is None:
            return

        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.logger.warning("E2B keepalive task stopped with an error: %s", exc)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    @override
    async def _create_sandbox(self) -> None:
        metadata = {
            "environment_name": self.environment_name,
            "session_id": self.session_id,
        }
        initial_lease_timeout_sec = min(
            E2B_SANDBOX_TIMEOUT_SEC,
            self._max_lifetime_sec,
        )

        self._sandbox = await harbor_e2b.AsyncSandbox.create(
            template=self._template_name,
            metadata=metadata,
            envs=self._startup_env(),
            timeout=initial_lease_timeout_sec,
            allow_internet_access=(self.network_policy.network_mode != NetworkMode.NO_NETWORK),
            network=self._sandbox_create_network_options(),
        )
        self._start_keepalive()

    @override
    async def stop(self, delete: bool) -> None:
        try:
            await self._stop_keepalive()
        finally:
            await super().stop(delete)
