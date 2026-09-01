from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from e2b.api import handle_api_exception
from e2b.api.client.api.templates import post_v3_templates
from e2b.api.client.models import Error, TemplateBuildRequestV3
from e2b.api.client.types import UNSET
from e2b.exceptions import BuildException
from e2b.template_async import main as template_main

_STORAGE_MB: ContextVar[int | None] = ContextVar(
    "nexau_e2b_template_storage_mb",
    default=None,
)
_PATCH_MARKER = "_nexau_supports_template_storage_mb"


def install_e2b_template_storage_patch() -> bool:
    """Teach E2B 2.x template builds about the deployment's diskSizeMB field."""

    original = template_main.request_build
    if getattr(original, _PATCH_MARKER, None) is True:
        return False

    @wraps(original)
    async def request_build_with_storage(
        client: Any,
        name: str,
        tags: list[str] | None,
        cpu_count: int,
        memory_mb: int,
    ) -> Any:
        storage_mb = _STORAGE_MB.get()
        if storage_mb is None:
            return await original(
                client,
                name=name,
                tags=tags,
                cpu_count=cpu_count,
                memory_mb=memory_mb,
            )

        body = TemplateBuildRequestV3(
            name=name,
            tags=tags if tags else UNSET,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
        )
        body["diskSizeMB"] = storage_mb
        response = await post_v3_templates.asyncio_detailed(client=client, body=body)
        if response.status_code >= 300:
            raise handle_api_exception(response, BuildException)
        if isinstance(response.parsed, Error):
            raise BuildException(f"API error: {response.parsed.message}")
        if response.parsed is None:
            raise BuildException("Failed to request build")
        return response.parsed

    setattr(request_build_with_storage, _PATCH_MARKER, True)
    template_main.request_build = request_build_with_storage
    return True


@contextmanager
def requested_e2b_template_storage(storage_mb: int | None) -> Iterator[None]:
    if storage_mb is not None and (isinstance(storage_mb, bool) or storage_mb <= 0):
        raise ValueError("E2B template storage must be a positive integer")

    install_e2b_template_storage_patch()
    token = _STORAGE_MB.set(storage_mb)
    try:
        yield
    finally:
        _STORAGE_MB.reset(token)
