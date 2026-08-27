from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
BRIDGE_SCRIPT = PROJECT_ROOT / "src" / "nexau_harbor" / "codex_responses_bridge.js"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class UpstreamState:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@contextmanager
def _upstream_server(
    responder: Callable[[BaseHTTPRequestHandler, dict[str, Any]], None],
) -> Iterator[tuple[UpstreamState, int]]:
    state = UpstreamState()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            state.requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "payload": payload,
                }
            )
            responder(self, payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, int(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _bridge_process(
    upstream_port: int,
    upstream_model: str | None = None,
) -> Iterator[int]:
    if shutil.which("node") is None:
        pytest.skip("node is required for the Codex Responses bridge test")

    bridge_port = _free_port()
    env = {
        **os.environ,
        "NORTHGATE_UPSTREAM_BASE_URL": f"http://127.0.0.1:{upstream_port}/v1",
        "NORTHGATE_BRIDGE_PORT": str(bridge_port),
    }
    if upstream_model is not None:
        env["NORTHGATE_UPSTREAM_MODEL"] = upstream_model
    process = subprocess.Popen(
        ["node", str(BRIDGE_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"bridge exited early:\nstdout={stdout}\nstderr={stderr}")
        try:
            with OPENER.open(
                f"http://127.0.0.1:{bridge_port}/health",
                timeout=1,
            ) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    else:
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        raise RuntimeError(f"bridge did not become ready:\nstdout={stdout}\nstderr={stderr}")

    try:
        yield bridge_port
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _post_json(port: int, payload: dict[str, Any]) -> tuple[int, str, str]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/responses",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with OPENER.open(request, timeout=10) as response:
            return response.status, response.headers.get_content_type(), response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get_content_type(), error.read().decode()


def _custom_exec_tool() -> dict[str, Any]:
    return {
        "type": "custom",
        "name": "exec",
        "description": "Run JavaScript code.",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: SOURCE\nSOURCE: /[\\s\\S]+/\n",
        },
    }


def _custom_apply_patch_tool() -> dict[str, Any]:
    return {
        "type": "custom",
        "name": "apply_patch",
        "description": "Apply a patch to files.",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: SOURCE\nSOURCE: /[\\s\\S]+/\n",
        },
    }


def test_bridge_converts_standard_mode_apply_patch_custom_tool() -> None:
    patch = "*** Begin Patch\n*** Add File: probe.txt\n+ok\n*** End Patch"

    def responder(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        assert payload["model"] == "nex-agi/Nex-N2-Pro"
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["name"] == "apply_patch"
        body = json.dumps(
            {
                "id": "resp_patch",
                "object": "response",
                "model": "nex-agi/Nex-N2-Pro",
                "status": "completed",
                "output": [
                    {
                        "id": "fc_patch",
                        "type": "function_call",
                        "call_id": "call_patch",
                        "name": "apply_patch",
                        "arguments": json.dumps({"command": patch}),
                        "status": "completed",
                    }
                ],
            }
        ).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    with _upstream_server(responder) as (_state, upstream_port):
        with _bridge_process(
            upstream_port,
            upstream_model="nex-agi/Nex-N2-Pro",
        ) as bridge_port:
            status, _content_type, body = _post_json(
                bridge_port,
                {
                    "model": "gpt-5.5",
                    "input": "patch the file",
                    "tools": [_custom_apply_patch_tool()],
                    "stream": False,
                },
            )

    assert status == 200
    response = json.loads(body)
    assert response["model"] == "gpt-5.5"
    assert response["output"][0]["type"] == "custom_tool_call"
    assert response["output"][0]["name"] == "apply_patch"
    assert response["output"][0]["input"] == patch


def test_bridge_converts_custom_exec_request_and_nonstreaming_response() -> None:
    def responder(handler: BaseHTTPRequestHandler, _payload: dict[str, Any]) -> None:
        body = json.dumps(
            {
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "tools": [
                    {
                        "type": "function",
                        "name": "exec",
                        "description": "Run JavaScript code.",
                        "parameters": {"type": "object"},
                    }
                ],
                "output": [
                    {
                        "id": "fc_exec",
                        "type": "function_call",
                        "call_id": "call_exec",
                        "name": "exec",
                        "arguments": '{"input":"text(\\"OK\\");"}',
                        "status": "completed",
                    },
                    {
                        "id": "fc_wait",
                        "type": "function_call",
                        "call_id": "call_wait",
                        "name": "wait",
                        "arguments": '{"cell_id":"1"}',
                        "status": "completed",
                    },
                ],
            }
        ).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    with _upstream_server(responder) as (state, upstream_port):
        with _bridge_process(upstream_port) as bridge_port:
            status, content_type, body = _post_json(
                bridge_port,
                {
                    "model": "test-model",
                    "input": [
                        {"role": "user", "content": "continue"},
                        {
                            "type": "custom_tool_call",
                            "id": "ctc_previous",
                            "call_id": "call_previous",
                            "name": "exec",
                            "input": "text('before');",
                            "status": "completed",
                        },
                        {
                            "type": "custom_tool_call_output",
                            "call_id": "call_previous",
                            "output": "done",
                        },
                    ],
                    "tools": [
                        _custom_exec_tool(),
                        {
                            "type": "function",
                            "name": "wait",
                            "parameters": {
                                "type": "object",
                                "properties": {"cell_id": {"type": "string"}},
                                "required": ["cell_id"],
                            },
                        },
                    ],
                    "tool_choice": {"type": "custom", "name": "exec"},
                    "stream": False,
                },
            )

    assert status == 200
    assert content_type == "application/json"
    assert len(state.requests) == 1
    upstream = state.requests[0]
    assert upstream["path"] == "/v1/responses"
    assert upstream["headers"]["authorization"] == "Bearer test-key"

    upstream_payload = upstream["payload"]
    exec_tool = upstream_payload["tools"][0]
    assert exec_tool == {
        "type": "function",
        "name": "exec",
        "description": "Run JavaScript code.",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The exec raw input.",
                }
            },
            "required": ["input"],
            "additionalProperties": False,
        },
    }
    assert upstream_payload["tool_choice"] == {"type": "function", "name": "exec"}
    assert upstream_payload["input"][1] == {
        "type": "function_call",
        "id": "ctc_previous",
        "call_id": "call_previous",
        "name": "exec",
        "arguments": '{"input":"text(\'before\');"}',
        "status": "completed",
    }
    assert upstream_payload["input"][2]["type"] == "function_call_output"

    response = json.loads(body)
    assert response["tools"][0] == _custom_exec_tool()
    assert response["output"][0] == {
        "id": "fc_exec",
        "type": "custom_tool_call",
        "call_id": "call_exec",
        "name": "exec",
        "input": 'text("OK");',
        "status": "completed",
    }
    assert response["output"][1]["type"] == "function_call"


def test_bridge_emits_custom_tool_stream_events_with_raw_input() -> None:
    arguments = '{"input":"text(\\"OK\\");"}'
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp_stream",
                "object": "response",
                "model": "nex-agi/Nex-N2-Pro",
                "status": "in_progress",
                "output": [],
            },
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "fc_stream",
                "type": "function_call",
                "call_id": "call_stream",
                "name": "exec",
                "arguments": "",
                "status": "in_progress",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "item_id": "fc_stream",
            "output_index": 0,
            "delta": arguments[:12],
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 3,
            "item_id": "fc_stream",
            "output_index": 0,
            "delta": arguments[12:],
        },
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": 4,
            "item_id": "fc_stream",
            "output_index": 0,
            "name": "exec",
            "arguments": arguments,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 5,
            "output_index": 0,
            "item": {
                "id": "fc_stream",
                "type": "function_call",
                "call_id": "call_stream",
                "name": "exec",
                "arguments": arguments,
                "status": "completed",
            },
        },
        {
            "type": "response.completed",
            "sequence_number": 6,
            "response": {
                "id": "resp_stream",
                "object": "response",
                "model": "nex-agi/Nex-N2-Pro",
                "status": "completed",
                "output": [
                    {
                        "id": "fc_stream",
                        "type": "function_call",
                        "call_id": "call_stream",
                        "name": "exec",
                        "arguments": arguments,
                        "status": "completed",
                    }
                ],
            },
        },
    ]

    def responder(handler: BaseHTTPRequestHandler, _payload: dict[str, Any]) -> None:
        stream = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
            for event in events
        ) + "data: [DONE]\n\n"
        encoded = stream.encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        for start in range(0, len(encoded), 17):
            handler.wfile.write(encoded[start : start + 17])
            handler.wfile.flush()

    with _upstream_server(responder) as (state, upstream_port):
        with _bridge_process(
            upstream_port,
            upstream_model="nex-agi/Nex-N2-Pro",
        ) as bridge_port:
            status, content_type, body = _post_json(
                bridge_port,
                {
                    "model": "test-model",
                    "input": "call exec",
                    "tools": [_custom_exec_tool()],
                    "tool_choice": "required",
                    "stream": True,
                },
            )

    assert status == 200
    assert content_type == "text/event-stream"
    assert state.requests[0]["payload"]["stream"] is True
    assert state.requests[0]["payload"]["model"] == "nex-agi/Nex-N2-Pro"

    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: {")
    ]
    event_types = [payload["type"] for payload in payloads]
    assert "response.function_call_arguments.delta" not in event_types
    assert "response.function_call_arguments.done" not in event_types
    assert "response.custom_tool_call_input.delta" in event_types
    assert "response.custom_tool_call_input.done" in event_types

    custom_delta = next(
        payload
        for payload in payloads
        if payload["type"] == "response.custom_tool_call_input.delta"
    )
    custom_done = next(
        payload
        for payload in payloads
        if payload["type"] == "response.custom_tool_call_input.done"
    )
    assert custom_delta["delta"] == 'text("OK");'
    assert custom_done["input"] == 'text("OK");'

    item_events = [
        payload
        for payload in payloads
        if payload["type"]
        in {"response.output_item.added", "response.output_item.done"}
    ]
    assert [payload["item"]["type"] for payload in item_events] == [
        "custom_tool_call",
        "custom_tool_call",
    ]
    completed = next(
        payload for payload in payloads if payload["type"] == "response.completed"
    )
    assert completed["response"]["model"] == "test-model"
    assert completed["response"]["output"][0]["type"] == "custom_tool_call"
    assert completed["response"]["output"][0]["input"] == 'text("OK");'

    sequence_numbers = [payload["sequence_number"] for payload in payloads]
    assert sequence_numbers == list(range(len(sequence_numbers)))


def test_bridge_rejects_namespace_tools_before_upstream_request() -> None:
    def responder(_handler: BaseHTTPRequestHandler, _payload: dict[str, Any]) -> None:
        raise AssertionError("namespace request must not reach upstream")

    with _upstream_server(responder) as (state, upstream_port):
        with _bridge_process(upstream_port) as bridge_port:
            status, _content_type, body = _post_json(
                bridge_port,
                {
                    "model": "test-model",
                    "input": "go",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "collaboration",
                            "tools": [],
                        }
                    ],
                },
            )

    assert status == 400
    assert state.requests == []
    assert "disable multi-agent tools" in json.loads(body)["error"]["message"]


def test_bridge_normalizes_n2_exec_and_string_number_arguments() -> None:
    def responder(handler: BaseHTTPRequestHandler, _payload: dict[str, Any]) -> None:
        body = json.dumps(
            {
                "id": "resp_n2",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "id": "fc_custom_exec",
                        "type": "function_call",
                        "call_id": "call_custom_exec",
                        "name": "exec",
                        "arguments": json.dumps(
                            {
                                "input": json.dumps(
                                    {
                                        "cmd": "printf BRIDGE_OK > probe.txt",
                                        "yield_time_ms": "10000",
                                        "max_output_tokens": "2000",
                                    }
                                )
                            }
                        ),
                        "status": "completed",
                    },
                    {
                        "id": "fc_exec_command",
                        "type": "function_call",
                        "call_id": "call_exec_command",
                        "name": "exec_command",
                        "arguments": json.dumps(
                            {
                                "cmd": "cat probe.txt",
                                "yield_time_ms": "10000",
                                "max_output_tokens": "2000",
                            }
                        ),
                        "status": "completed",
                    },
                ],
            }
        ).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    exec_command_tool = {
        "type": "function",
        "name": "exec_command",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "yield_time_ms": {"type": "integer"},
                "max_output_tokens": {"type": "integer"},
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
    }
    with _upstream_server(responder) as (_state, upstream_port):
        with _bridge_process(upstream_port) as bridge_port:
            status, _content_type, body = _post_json(
                bridge_port,
                {
                    "model": "test-model",
                    "input": "create probe",
                    "tools": [_custom_exec_tool(), exec_command_tool],
                    "stream": False,
                },
            )

    assert status == 200
    response = json.loads(body)
    custom_call = response["output"][0]
    assert custom_call["type"] == "custom_tool_call"
    assert "await tools.exec_command" in custom_call["input"]
    code_arguments = custom_call["input"].split("tools.exec_command(", 1)[1].split(");", 1)[0]
    normalized_code_arguments = json.loads(code_arguments)
    assert normalized_code_arguments["yield_time_ms"] == 10000
    assert normalized_code_arguments["max_output_tokens"] == 2000

    function_call = response["output"][1]
    normalized = json.loads(function_call["arguments"])
    assert normalized["yield_time_ms"] == 10000
    assert normalized["max_output_tokens"] == 2000
