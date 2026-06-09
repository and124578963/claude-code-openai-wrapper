"""Per-request MCP server passthrough.

The wrapper isolates filesystem settings (setting_sources=[]), so MCP servers
added via `claude mcp add` are never loaded. The ONLY supported path is the
`mcp_servers` field on the chat completion request, which must reach
claude_cli.run_completion as ClaudeAgentOptions-compatible config together
with matching mcp__<server> allowed_tools entries.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src import main as wrapper_main
from src.constants import DEFAULT_ALLOWED_TOOLS
from src.models import ChatCompletionRequest, Message


STDIO_SERVER = {"command": "python", "args": ["-m", "my_mcp_server"]}
HTTP_SERVER = {"type": "http", "url": "https://mcp.example.com/mcp"}


class _FakeAssistantBlock:
    def __init__(self, text: str) -> None:
        self.text = text


def _capture_call(captured_kwargs: dict[str, Any]):
    async def fake_run_completion(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"content": [_FakeAssistantBlock("OK")]}

    return fake_run_completion


@pytest.fixture
def client() -> TestClient:
    return TestClient(wrapper_main.app)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def _request(**overrides: Any) -> ChatCompletionRequest:
    base: dict[str, Any] = {
        "model": "claude-haiku-4-5-20251001",
        "messages": [Message(role="user", content="ping")],
    }
    base.update(overrides)
    return ChatCompletionRequest(**base)


def test_valid_stdio_and_http_servers_accepted() -> None:
    req = _request(mcp_servers={"files": STDIO_SERVER, "remote": HTTP_SERVER})
    assert set(req.mcp_servers) == {"files", "remote"}


def test_server_name_with_invalid_chars_rejected() -> None:
    with pytest.raises(ValidationError, match="server name"):
        _request(mcp_servers={"bad name!": STDIO_SERVER})


def test_stdio_server_without_command_rejected() -> None:
    with pytest.raises(ValidationError, match="requires 'command'"):
        _request(mcp_servers={"files": {"args": ["x"]}})


def test_http_server_without_url_rejected() -> None:
    with pytest.raises(ValidationError, match="requires 'url'"):
        _request(mcp_servers={"remote": {"type": "http"}})


def test_unknown_server_type_rejected() -> None:
    with pytest.raises(ValidationError, match="unsupported type"):
        _request(mcp_servers={"x": {"type": "websocket", "url": "wss://x"}})


def test_mcp_tools_must_be_prefixed() -> None:
    with pytest.raises(ValidationError, match="must start with 'mcp__'"):
        _request(mcp_servers={"files": STDIO_SERVER}, mcp_tools=["read_file"])


def test_default_allowed_tools_are_server_wildcards() -> None:
    req = _request(mcp_servers={"files": STDIO_SERVER, "remote": HTTP_SERVER})
    assert sorted(req.get_mcp_allowed_tools()) == ["mcp__files", "mcp__remote"]


def test_explicit_mcp_tools_win_over_wildcards() -> None:
    req = _request(
        mcp_servers={"files": STDIO_SERVER},
        mcp_tools=["mcp__files__read_file"],
    )
    assert req.get_mcp_allowed_tools() == ["mcp__files__read_file"]


# ---------------------------------------------------------------------------
# Endpoint wiring (non-streaming branch)
# ---------------------------------------------------------------------------


def test_mcp_servers_reach_run_completion(client: TestClient) -> None:
    captured: dict[str, Any] = {}
    with patch.object(
        wrapper_main.claude_cli, "run_completion", side_effect=_capture_call(captured)
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "ping"}],
                "mcp_servers": {"files": STDIO_SERVER},
            },
        )

    assert resp.status_code == 200, resp.text
    assert captured.get("mcp_servers") == {"files": STDIO_SERVER}
    # MCP-only request: agent gets ONLY the MCP wildcard, no built-in tools.
    assert captured.get("allowed_tools") == ["mcp__files"]
    # The zero-tools lockdown must NOT fire for MCP requests.
    assert captured.get("disallowed_tools") is None
    assert captured.get("permission_mode") == "bypassPermissions"
    # Filesystem isolation stays on — MCP comes from the request, not ~/.claude.
    assert captured.get("setting_sources") == []


def test_mcp_plus_enable_tools_merges_builtin_and_mcp(client: TestClient) -> None:
    captured: dict[str, Any] = {}
    with patch.object(
        wrapper_main.claude_cli, "run_completion", side_effect=_capture_call(captured)
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "ping"}],
                "enable_tools": True,
                "mcp_servers": {"files": STDIO_SERVER},
                "mcp_tools": ["mcp__files__read_file"],
            },
        )

    assert resp.status_code == 200, resp.text
    assert captured.get("allowed_tools") == list(DEFAULT_ALLOWED_TOOLS) + [
        "mcp__files__read_file"
    ]
    assert captured.get("mcp_servers") == {"files": STDIO_SERVER}


def test_no_mcp_keeps_zero_tools_lockdown(client: TestClient) -> None:
    """Regression guard: the default OpenAI-compat path must stay locked down."""
    captured: dict[str, Any] = {}
    with patch.object(
        wrapper_main.claude_cli, "run_completion", side_effect=_capture_call(captured)
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert resp.status_code == 200, resp.text
    assert captured.get("allowed_tools") == []
    assert captured.get("mcp_servers") is None
