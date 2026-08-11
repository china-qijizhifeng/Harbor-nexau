"""E2B compatibility environment for Harbor 0.21.x."""

import asyncio
from typing import override

from harbor.environments import e2b as harbor_e2b
from harbor.models.task.config import NetworkMode
from tenacity import retry, stop_after_attempt, wait_exponential

E2B_SANDBOX_TIMEOUT_SEC = 3_600
E2B_KEEPALIVE_INTERVAL_SEC = 60


class NexAUE2BEnvironment(harbor_e2b.E2BEnvironment):
    """Use a renewable E2B lease accepted by one-hour-limited accounts."""

    async def _keepalive_loop(self) -> None:
        while True:
            sandbox = self._sandbox
            if sandbox is None:
                return

            try:
                await sandbox.set_timeout(E2B_SANDBOX_TIMEOUT_SEC)
                self.logger.debug(
                    "Refreshed E2B sandbox %s timeout to %s seconds",
                    sandbox.sandbox_id,
                    E2B_SANDBOX_TIMEOUT_SEC,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.warning(
                    "Failed to refresh E2B sandbox timeout; will retry in %s seconds: %s",
                    E2B_KEEPALIVE_INTERVAL_SEC,
                    exc,
                )

            await asyncio.sleep(E2B_KEEPALIVE_INTERVAL_SEC)

    def _start_keepalive(self) -> None:
        keepalive_task = getattr(self, "_keepalive_task", None)
        if keepalive_task is not None and not keepalive_task.done():
            return

        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(),
            name=f"e2b-keepalive-{self.session_id}",
        )

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

        self._sandbox = await harbor_e2b.AsyncSandbox.create(
            template=self._template_name,
            metadata=metadata,
            envs=self._startup_env(),
            timeout=E2B_SANDBOX_TIMEOUT_SEC,
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
