"""Regression: permission_mode='bypassPermissions' MUST be passed to
claude_cli.run_completion in the OpenAI-compatibility default branch
(enable_tools=False). Without it, Claude Code 2.x auto-loads interactive
tools (AskUserQuestion, EnterPlanMode, ...), the headless dispatch is
interrupted, and the response leaks the literal string
"[Request interrupted by user]" as assistant content."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src import main as wrapper_main


class _FakeAssistantBlock:
    def __init__(self, text: str) -> None:
        self.text = text


def _capture_call(captured_kwargs: dict[str, Any]):
    async def fake_run_completion(*args, **kwargs):
        captured_kwargs.update(kwargs)
        # Yield one assistant chunk so the wrapper builds a normal response.
        yield {"content": [_FakeAssistantBlock("OK")]}

    return fake_run_completion


@pytest.fixture
def client() -> TestClient:
    return TestClient(wrapper_main.app)


def test_chat_completion_default_passes_bypass_permissions(client: TestClient) -> None:
    captured: dict[str, Any] = {}
    with patch.object(
        wrapper_main.claude_cli,
        "run_completion",
        side_effect=_capture_call(captured),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 10,
            },
        )

    assert resp.status_code == 200, resp.text
    assert (
        captured.get("permission_mode") == "bypassPermissions"
    ), (
        "OpenAI-default chat completion (enable_tools=False) must pass "
        "permission_mode='bypassPermissions' to avoid the "
        "[Request interrupted by user] sentinel from headless interactive tools. "
        f"Got: {captured.get('permission_mode')!r}"
    )


def test_chat_completion_with_tools_also_passes_bypass_permissions(client: TestClient) -> None:
    captured: dict[str, Any] = {}
    with patch.object(
        wrapper_main.claude_cli,
        "run_completion",
        side_effect=_capture_call(captured),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 10,
                "enable_tools": True,
            },
        )

    assert resp.status_code == 200, resp.text
    assert captured.get("permission_mode") == "bypassPermissions"


def test_assistant_content_is_never_the_interrupt_sentinel(client: TestClient) -> None:
    """If claude_cli ever yields the sentinel string as assistant content,
    the wrapper must surface a 5xx instead of pretending it was a real reply."""
    INTERRUPT = "[Request interrupted by user]"

    async def fake_run_completion_emitting_interrupt(*args, **kwargs):
        yield {"content": [_FakeAssistantBlock(INTERRUPT)]}

    with patch.object(
        wrapper_main.claude_cli,
        "run_completion",
        side_effect=fake_run_completion_emitting_interrupt,
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 10,
            },
        )

    if resp.status_code == 200:
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        assert INTERRUPT not in content, (
            "The wrapper must NOT pass the interrupt sentinel through as "
            "assistant content. It should either filter it out or fail with "
            "an error."
        )
