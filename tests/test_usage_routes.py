"""Маршруты отдают usage из ResultMessage SDK, а не оценку «длина / 4»."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src import main as wrapper_main


class _Text:
    def __init__(self, text: str) -> None:
        self.text = text


RESULT = {
    "subtype": "success",
    "total_cost_usd": 0.0123,
    "result": "OK",
    "usage": {
        "input_tokens": 120,
        "cache_creation_input_tokens": 3000,
        "cache_read_input_tokens": 15000,
        "output_tokens": 450,
    },
}

CHAT = {"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "ping"}]}
ANTHROPIC = {"model": "claude-haiku-4-5-20251001", "max_tokens": 100,
             "messages": [{"role": "user", "content": "ping"}]}


def _completion(*chunks):
    async def fake(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    return fake


@pytest.fixture
def auth_ok(monkeypatch):
    monkeypatch.setattr(
        wrapper_main, "validate_claude_code_auth", lambda: (True, {"method": "claude_cli"})
    )


@pytest.fixture(autouse=True)
def fresh_rate_limit():
    # slowapi: 10 запросов/мин на IP общие для всего прогона; тесты мимо этого
    # лимита не должны ни упираться в него, ни съедать его у соседей.
    if wrapper_main.limiter:
        wrapper_main.limiter.reset()
    yield
    if wrapper_main.limiter:
        wrapper_main.limiter.reset()


@pytest.fixture
def client() -> TestClient:
    return TestClient(wrapper_main.app)


def test_chat_completion_reports_sdk_usage_with_cached_tokens(auth_ok, client):
    fake = _completion({"content": [_Text("OK")]}, RESULT)
    with patch.object(wrapper_main.claude_cli, "run_completion", fake):
        resp = client.post("/v1/chat/completions", json=CHAT)

    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] == 18120
    assert usage["completion_tokens"] == 450
    assert usage["prompt_tokens_details"] == {"cached_tokens": 15000}


def test_chat_completion_without_sdk_usage_keeps_the_estimate(auth_ok, client):
    fake = _completion({"content": [_Text("OK")]})
    with patch.object(wrapper_main.claude_cli, "run_completion", fake):
        resp = client.post("/v1/chat/completions", json=CHAT)

    usage = resp.json()["usage"]
    assert usage["completion_tokens"] == len("OK") // 4
    assert usage.get("prompt_tokens_details") is None


def test_stream_final_chunk_reports_sdk_usage(auth_ok, client):
    body = {**CHAT, "stream": True, "stream_options": {"include_usage": True}}
    fake = _completion({"content": [_Text("OK")]}, RESULT)
    with patch.object(wrapper_main.claude_cli, "run_completion", fake):
        resp = client.post("/v1/chat/completions", json=body)

    chunks = [json.loads(line[len("data: "):]) for line in resp.text.splitlines()
              if line.startswith("data: {")]
    usage = [c["usage"] for c in chunks if c.get("usage")][-1]
    assert usage["prompt_tokens"] == 18120
    assert usage["prompt_tokens_details"]["cached_tokens"] == 15000


def test_anthropic_messages_report_sdk_usage(auth_ok, client):
    fake = _completion({"content": [_Text("OK")]}, RESULT)
    with patch.object(wrapper_main.claude_cli, "run_completion", fake):
        resp = client.post("/v1/messages", json=ANTHROPIC)

    usage = resp.json()["usage"]
    assert usage["input_tokens"] == 120
    assert usage["cache_creation_input_tokens"] == 3000
    assert usage["cache_read_input_tokens"] == 15000
    assert usage["output_tokens"] == 450
