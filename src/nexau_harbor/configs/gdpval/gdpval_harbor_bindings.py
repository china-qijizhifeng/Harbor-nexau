from __future__ import annotations

import base64
import mimetypes
import os
from typing import TYPE_CHECKING, Any

from gdpval_env_setup import prefix_shell_command

if TYPE_CHECKING:
    from nexau.archs.main_sub.framework_context import FrameworkContext


def finish(
    summary: str = "",
    reason: str | None = None,
    file_paths: list[str] | None = None,
    paths: list[str] | None = None,
    agent_state: Any | None = None,
    **_: Any,
) -> dict[str, Any]:
    normalized_summary = str(summary or reason or "").strip()
    selected_paths = file_paths if file_paths is not None else (paths or [])
    normalized_paths = [str(path).strip() for path in selected_paths if str(path).strip()]

    if normalized_paths and agent_state is not None:
        try:
            from nexau.archs.tool.builtin._sandbox_utils import get_sandbox, resolve_path

            sandbox = get_sandbox(agent_state)
            resolved = [resolve_path(path, sandbox) for path in normalized_paths]
            missing = [path for path in resolved if not sandbox.file_exists(path)]
            if missing:
                return {
                    "status": "error",
                    "content": f"ERROR: Files do not exist: {missing}",
                }
            normalized_paths = resolved
        except Exception:
            pass

    payload = {
        "summary": normalized_summary,
        "reason": normalized_summary,
        "file_paths": normalized_paths,
        "paths": normalized_paths,
    }
    return {"status": "ok", "output": payload}


def _shell_timeout_ms() -> int:
    raw = (os.environ.get("GDPVAL_SHELL_TIMEOUT_SEC") or "1800").strip()
    try:
        return max(1, int(raw)) * 1000
    except ValueError:
        return 1_800_000


def run_shell_command(
    command: str,
    description: str | None = None,
    is_background: bool = False,
    dir_path: str | None = None,
    timeout_ms: int | None = None,
    update_output: Any | None = None,
    agent_state: Any | None = None,
    *,
    ctx: FrameworkContext,
    **kwargs: Any,
) -> dict[str, Any]:
    from nexau.archs.tool.builtin.shell_tools import (
        run_shell_command as builtin_run_shell_command,
    )

    effective_timeout = (
        timeout_ms if isinstance(timeout_ms, int) and timeout_ms > 0 else _shell_timeout_ms()
    )
    return builtin_run_shell_command(
        command=prefix_shell_command(command),
        description=description,
        is_background=is_background,
        dir_path=dir_path,
        timeout_ms=effective_timeout,
        update_output=update_output,
        agent_state=agent_state,
        ctx=ctx,
        **kwargs,
    )


def _read_image(file_path: str, agent_state: Any | None) -> tuple[str, bytes]:
    from nexau.archs.sandbox import SandboxStatus
    from nexau.archs.tool.builtin._sandbox_utils import get_sandbox, resolve_path

    sandbox = get_sandbox(agent_state)
    resolved_path = resolve_path(file_path, sandbox)
    if not sandbox.file_exists(resolved_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if sandbox.get_file_info(resolved_path).is_directory:
        raise IsADirectoryError(f"Path is a directory: {file_path}")

    result = sandbox.read_file(resolved_path, binary=True)
    if result.status != SandboxStatus.SUCCESS:
        raise RuntimeError(result.error or "Failed to read image from sandbox")
    if isinstance(result.content, (bytes, bytearray)):
        payload = bytes(result.content)
    elif isinstance(result.content, str):
        payload = result.content.encode("utf-8", errors="replace")
    else:
        payload = b""
    return resolved_path, payload


def _data_url(file_path: str, payload: bytes) -> str:
    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    try:
        choices = response.choices
        content = choices[0].message.content
        if isinstance(content, str):
            return content.strip()
    except (AttributeError, IndexError, TypeError):
        return ""
    return ""


def describe_image(
    file_path: str,
    question: str | None = None,
    detail: str | None = None,
    max_output_tokens: int | None = None,
    agent_state: Any | None = None,
    **_: Any,
) -> dict[str, Any]:
    model = (os.environ.get("GDPVAL_VLM_MODEL") or "").strip()
    if not model:
        return {
            "content": "ERROR: Set GDPVAL_VLM_MODEL before using describe_image.",
            "error": {"type": "VLM_NOT_CONFIGURED"},
        }

    try:
        from openai import OpenAI

        resolved_path, payload = _read_image(file_path, agent_state)
        image_url = _data_url(resolved_path, payload)
        prompt = (question or "").strip() or (
            "Describe this image and extract all visible text, labels, numbers, "
            "and table structure."
        )
        normalized_detail = (detail or "low").lower()
        if normalized_detail not in {"low", "high", "auto"}:
            normalized_detail = "low"

        client = OpenAI(
            api_key=(
                os.environ.get("GDPVAL_VLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            ),
            base_url=(
                os.environ.get("GDPVAL_VLM_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            ),
            timeout=float(os.environ.get("GDPVAL_VLM_TIMEOUT_SEC", "120")),
        )
        limit = max_output_tokens or int(os.environ.get("GDPVAL_VLM_MAX_TOKENS", "1024"))
        api_type = (os.environ.get("GDPVAL_VLM_API_TYPE") or "openai_chat_completion").lower()
        if api_type in {"open_responses", "openai_responses", "responses"}:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": image_url,
                                "detail": normalized_detail,
                            },
                        ],
                    }
                ],
                max_output_tokens=limit,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                    "detail": normalized_detail,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=limit,
            )
        text = _response_text(response) or "No useful visual content extracted."
        return {"content": text, "returnDisplay": "Image described via external VLM."}
    except Exception as exc:
        return {
            "content": f"ERROR: external VLM failed: {exc}",
            "error": {"type": "VLM_REQUEST_FAILED", "message": str(exc)},
        }
