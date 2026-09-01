from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from e2b.template_async import main as template_main

from nexau_harbor import e2b_template_storage


async def test_template_storage_patch_injects_disk_without_affecting_default(
    monkeypatch,
) -> None:
    original = AsyncMock(return_value="default")
    monkeypatch.setattr(template_main, "request_build", original)
    response = SimpleNamespace(
        status_code=200,
        parsed=SimpleNamespace(template_id="template", build_id="build", tags=[]),
    )
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(
        e2b_template_storage.post_v3_templates,
        "asyncio_detailed",
        request,
    )

    assert e2b_template_storage.install_e2b_template_storage_patch()
    assert not e2b_template_storage.install_e2b_template_storage_patch()

    with e2b_template_storage.requested_e2b_template_storage(10_240):
        result = await template_main.request_build(
            object(),
            name="task",
            tags=None,
            cpu_count=2,
            memory_mb=4096,
        )

    assert result is response.parsed
    body = request.await_args.kwargs["body"].to_dict()
    assert body["diskSizeMB"] == 10_240
    assert body["cpuCount"] == 2
    assert body["memoryMB"] == 4096

    assert (
        await template_main.request_build(
            object(),
            name="default",
            tags=None,
            cpu_count=1,
            memory_mb=1024,
        )
        == "default"
    )
    original.assert_awaited_once()


async def test_template_storage_context_is_concurrency_safe(monkeypatch) -> None:
    monkeypatch.setattr(template_main, "request_build", AsyncMock(return_value="default"))
    seen: list[int] = []

    async def capture(**kwargs):
        await asyncio.sleep(0)
        seen.append(kwargs["body"].to_dict()["diskSizeMB"])
        return SimpleNamespace(status_code=200, parsed=SimpleNamespace())

    monkeypatch.setattr(
        e2b_template_storage.post_v3_templates,
        "asyncio_detailed",
        AsyncMock(side_effect=capture),
    )
    e2b_template_storage.install_e2b_template_storage_patch()

    async def build(storage_mb: int) -> None:
        with e2b_template_storage.requested_e2b_template_storage(storage_mb):
            await template_main.request_build(
                object(),
                name=str(storage_mb),
                tags=None,
                cpu_count=1,
                memory_mb=1024,
            )

    await asyncio.gather(build(10_240), build(20_480))

    assert sorted(seen) == [10_240, 20_480]
