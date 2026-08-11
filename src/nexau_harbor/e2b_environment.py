"""E2B compatibility environment for Harbor 0.21.x."""

from typing import override

from harbor.environments import e2b as harbor_e2b
from harbor.models.task.config import NetworkMode
from tenacity import retry, stop_after_attempt, wait_exponential

E2B_SANDBOX_TIMEOUT_SEC = 3_600


class NexAUE2BEnvironment(harbor_e2b.E2BEnvironment):
    """Use an E2B sandbox lifetime accepted by one-hour-limited accounts."""

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
