"""Cancel-on-disconnect, MAX_TIMEOUT deadline and thinking-budget plumbing.

Regression coverage for the 2026-06-11 incident (run 628f6e40): the Java
client dropped non-streaming requests on its read-timeout, but the wrapper
never noticed — four orphaned Claude sessions ran 8–42 minutes past the
disconnect, burning the shared subscription rate-cap. Three behaviors are
pinned here:

1. collect_completion_guarded() cancels the SDK task when the client
   disconnects (route answers 204) — non-streaming /v1/chat/completions
   and /v1/messages both go through it.
2. The same loop enforces the server deadline (claude_cli.timeout, from
   MAX_TIMEOUT ms) and returns HTTP 504 — the wrapper must give up BEFORE
   the client's read-timeout (prod: 150s server vs 180s client).
3. max_thinking_tokens flows header → claude_options → run_completion →
   ClaudeAgentOptions.thinking (the deprecated max_thinking_tokens SDK
   field only toggles thinking on/off on current models).
"""

from __future__ import annotations

import asyncio
import tempfile
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src import main as wrapper_main
from src.main import ClientDisconnected, collect_completion_guarded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeRequest:
    """Stand-in for starlette.Request's ASGI receive channel.

    Mirrors a real server: receive() blocks until the client disconnects,
    then yields http.disconnect (uvicorn keeps repeating it afterwards).
    """

    def __init__(self, disconnect_after: float | None = None) -> None:
        self._disconnect_after = disconnect_after

    async def receive(self) -> dict:
        if self._disconnect_after is None:
            await asyncio.sleep(3600)
        else:
            await asyncio.sleep(self._disconnect_after)
        return {"type": "http.disconnect"}


class CrashingReceiveRequest:
    """receive() blows up instead of yielding a disconnect message."""

    async def receive(self) -> dict:
        raise RuntimeError("receive channel exploded")


def make_slow_completion(state: dict):
    """Async generator: one chunk, then a long sleep. Records cancellation."""

    async def slow_completion():
        try:
            yield {"type": "chunk", "i": 0}
            await asyncio.sleep(30)
            yield {"type": "chunk", "i": 1}
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    return slow_completion()


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


def _capture_run_completion(captured_kwargs: dict[str, Any]):
    async def fake_run_completion(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"content": [_FakeTextBlock("OK")]}

    return fake_run_completion


@pytest.fixture
def auth_ok(monkeypatch):
    monkeypatch.setattr(
        wrapper_main, "validate_claude_code_auth", lambda: (True, {"method": "claude_cli"})
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(wrapper_main.app)


# ---------------------------------------------------------------------------
# collect_completion_guarded: unit level
# ---------------------------------------------------------------------------


class TestCollectCompletionGuarded:
    @pytest.mark.asyncio
    async def test_normal_completion_passes_through(self):
        async def quick_completion():
            yield {"i": 0}
            yield {"i": 1}
            yield {"subtype": "success", "result": "done"}

        chunks = await collect_completion_guarded(
            FakeRequest(), quick_completion(), "req-normal"
        )
        assert [c.get("i") for c in chunks[:2]] == [0, 1]
        assert chunks[-1]["result"] == "done"

    @pytest.mark.asyncio
    async def test_disconnect_cancels_completion(self):
        state = {"cancelled": False}
        completion = make_slow_completion(state)

        with pytest.raises(ClientDisconnected):
            await collect_completion_guarded(
                FakeRequest(disconnect_after=0.05), completion, "req-disc"
            )

        # Give the event loop a tick so the cancelled task unwinds.
        await asyncio.sleep(0.1)
        assert state["cancelled"], "SDK generator must receive CancelledError on disconnect"

    @pytest.mark.asyncio
    async def test_deadline_raises_504_and_cancels(self, monkeypatch):
        monkeypatch.setattr(wrapper_main.claude_cli, "timeout", 0.05)
        state = {"cancelled": False}
        completion = make_slow_completion(state)

        with pytest.raises(HTTPException) as exc_info:
            await collect_completion_guarded(FakeRequest(), completion, "req-deadline")

        assert exc_info.value.status_code == 504
        assert "MAX_TIMEOUT" in exc_info.value.detail

        await asyncio.sleep(0.1)
        assert state["cancelled"], "SDK generator must receive CancelledError on deadline"

    @pytest.mark.asyncio
    async def test_completion_error_propagates(self):
        async def broken_completion():
            raise ValueError("boom")
            yield  # pragma: no cover - makes this a generator

        with pytest.raises(ValueError, match="boom"):
            await collect_completion_guarded(FakeRequest(), broken_completion(), "req-err")

    @pytest.mark.asyncio
    async def test_disconnect_wins_over_deadline(self, monkeypatch):
        """A dead client is detected before a later deadline fires."""
        monkeypatch.setattr(wrapper_main.claude_cli, "timeout", 5)
        state = {"cancelled": False}

        with pytest.raises(ClientDisconnected):
            await collect_completion_guarded(
                FakeRequest(disconnect_after=0.05),
                make_slow_completion(state),
                "req-both",
            )

    @pytest.mark.asyncio
    async def test_watcher_crash_does_not_kill_completion(self, monkeypatch):
        """A broken receive channel must not cancel a healthy completion."""
        monkeypatch.setattr(wrapper_main.claude_cli, "timeout", 5)

        async def quick_completion():
            await asyncio.sleep(0.05)
            yield {"subtype": "success", "result": "done"}

        chunks = await collect_completion_guarded(
            CrashingReceiveRequest(), quick_completion(), "req-crash"
        )
        assert chunks[-1]["result"] == "done"


# ---------------------------------------------------------------------------
# Route level: /v1/chat/completions and /v1/messages
# ---------------------------------------------------------------------------


CHAT_BODY = {
    "model": "claude-haiku-4-5-20251001",
    "messages": [{"role": "user", "content": "ping"}],
}

ANTHROPIC_BODY = {
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "ping"}],
}


class TestRouteDeadline:
    def test_chat_completions_returns_504_on_deadline(
        self, auth_ok, client, monkeypatch
    ):
        monkeypatch.setattr(wrapper_main.claude_cli, "timeout", 0.1)

        async def never_ending(*args, **kwargs):
            yield {"type": "system", "subtype": "init", "data": {}}
            await asyncio.sleep(30)

        with patch.object(wrapper_main.claude_cli, "run_completion", never_ending):
            resp = client.post("/v1/chat/completions", json=CHAT_BODY)

        assert resp.status_code == 504
        # The global HTTPException handler wraps errors OpenAI-style.
        error = resp.json()["error"]
        assert error["code"] == "504"
        assert "MAX_TIMEOUT" in error["message"]

    def test_anthropic_messages_returns_504_on_deadline(
        self, auth_ok, client, monkeypatch
    ):
        monkeypatch.setattr(wrapper_main.claude_cli, "timeout", 0.1)

        async def never_ending(*args, **kwargs):
            yield {"type": "system", "subtype": "init", "data": {}}
            await asyncio.sleep(30)

        with patch.object(wrapper_main.claude_cli, "run_completion", never_ending):
            resp = client.post("/v1/messages", json=ANTHROPIC_BODY)

        assert resp.status_code == 504

    def test_chat_completions_normal_response_still_works(
        self, auth_ok, client
    ):
        captured: dict[str, Any] = {}
        with patch.object(
            wrapper_main.claude_cli, "run_completion", _capture_run_completion(captured)
        ):
            resp = client.post("/v1/chat/completions", json=CHAT_BODY)

        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "OK"

    def test_anthropic_messages_normal_response_still_works(
        self, auth_ok, client
    ):
        captured: dict[str, Any] = {}
        with patch.object(
            wrapper_main.claude_cli, "run_completion", _capture_run_completion(captured)
        ):
            resp = client.post("/v1/messages", json=ANTHROPIC_BODY)

        assert resp.status_code == 200
        assert resp.json()["content"][0]["text"] == "OK"


class TestThinkingPlumbing:
    """Header → claude_options → run_completion kwarg."""

    def test_header_budget_reaches_run_completion(self, auth_ok, client):
        captured: dict[str, Any] = {}
        with patch.object(
            wrapper_main.claude_cli, "run_completion", _capture_run_completion(captured)
        ):
            resp = client.post(
                "/v1/chat/completions",
                json=CHAT_BODY,
                headers={"X-Claude-Max-Thinking-Tokens": "1024"},
            )

        assert resp.status_code == 200
        assert captured["max_thinking_tokens"] == 1024

    def test_no_header_passes_none(self, auth_ok, client):
        captured: dict[str, Any] = {}
        with patch.object(
            wrapper_main.claude_cli, "run_completion", _capture_run_completion(captured)
        ):
            resp = client.post("/v1/chat/completions", json=CHAT_BODY)

        assert resp.status_code == 200
        assert captured["max_thinking_tokens"] is None

    def test_body_max_tokens_does_not_become_thinking_budget(
        self, auth_ok, client
    ):
        """The old max_tokens→max_thinking_tokens mapping must stay dead."""
        captured: dict[str, Any] = {}
        body = dict(CHAT_BODY, max_tokens=8192)
        with patch.object(
            wrapper_main.claude_cli, "run_completion", _capture_run_completion(captured)
        ):
            resp = client.post("/v1/chat/completions", json=body)

        assert resp.status_code == 200
        assert captured["max_thinking_tokens"] is None


# ---------------------------------------------------------------------------
# ClaudeCodeCLI.run_completion → ClaudeAgentOptions.thinking
# ---------------------------------------------------------------------------


class TestRunCompletionThinkingConfig:
    @pytest.fixture
    def cli_instance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.auth.validate_claude_code_auth") as mock_validate:
                with patch("src.auth.auth_manager") as mock_auth:
                    mock_validate.return_value = (True, {"method": "anthropic"})
                    mock_auth.get_claude_code_env_vars.return_value = {}

                    from src.claude_cli import ClaudeCodeCLI

                    yield ClaudeCodeCLI(cwd=temp_dir)

    async def _captured_options(self, cli, **kwargs):
        captured = []

        async def mock_query(prompt, options):
            captured.append(options)
            yield {"type": "assistant"}

        with patch("src.claude_cli.query", mock_query):
            async for _ in cli.run_completion("Hello", **kwargs):
                pass

        assert len(captured) == 1
        return captured[0]

    @pytest.mark.asyncio
    async def test_zero_disables_thinking(self, cli_instance, monkeypatch):
        monkeypatch.delenv("DEFAULT_MAX_THINKING_TOKENS", raising=False)
        opts = await self._captured_options(cli_instance, max_thinking_tokens=0)
        assert opts.thinking == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_positive_budget_enables_thinking(self, cli_instance, monkeypatch):
        monkeypatch.delenv("DEFAULT_MAX_THINKING_TOKENS", raising=False)
        opts = await self._captured_options(cli_instance, max_thinking_tokens=1024)
        assert opts.thinking == {"type": "enabled", "budget_tokens": 1024}

    @pytest.mark.asyncio
    async def test_unset_leaves_cli_default(self, cli_instance, monkeypatch):
        monkeypatch.delenv("DEFAULT_MAX_THINKING_TOKENS", raising=False)
        opts = await self._captured_options(cli_instance, max_thinking_tokens=None)
        assert opts.thinking is None
        # The deprecated SDK field must stay untouched too.
        assert opts.max_thinking_tokens is None

    @pytest.mark.asyncio
    async def test_env_default_zero_disables(self, cli_instance, monkeypatch):
        monkeypatch.setenv("DEFAULT_MAX_THINKING_TOKENS", "0")
        opts = await self._captured_options(cli_instance, max_thinking_tokens=None)
        assert opts.thinking == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_env_default_budget_enables(self, cli_instance, monkeypatch):
        monkeypatch.setenv("DEFAULT_MAX_THINKING_TOKENS", "2048")
        opts = await self._captured_options(cli_instance, max_thinking_tokens=None)
        assert opts.thinking == {"type": "enabled", "budget_tokens": 2048}

    @pytest.mark.asyncio
    async def test_request_value_overrides_env_default(self, cli_instance, monkeypatch):
        monkeypatch.setenv("DEFAULT_MAX_THINKING_TOKENS", "0")
        opts = await self._captured_options(cli_instance, max_thinking_tokens=512)
        assert opts.thinking == {"type": "enabled", "budget_tokens": 512}

    @pytest.mark.asyncio
    async def test_invalid_env_default_is_ignored(self, cli_instance, monkeypatch):
        monkeypatch.setenv("DEFAULT_MAX_THINKING_TOKENS", "lots")
        opts = await self._captured_options(cli_instance, max_thinking_tokens=None)
        assert opts.thinking is None
