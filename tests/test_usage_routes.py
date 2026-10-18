"""Маршруты отдают usage из ResultMessage SDK, а не оценку «длина / 4»."""
from __future__ import annotations

import json
import logging
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

# Синтетический итог, который wrapper сам отдаёт при исключении SDK (claude_cli.run_completion).
SDK_ERROR = {
    "type": "result",
    "subtype": "error_during_execution",
    "is_error": True,
    "error_message": "boom",
}

CHAT = {"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "ping"}]}
ANTHROPIC = {"model": "claude-haiku-4-5-20251001", "max_tokens": 100,
             "messages": [{"role": "user", "content": "ping"}]}


def _sse(resp):
    """Строки `data: ...` стрима: (JSON-чанки, все data-строки)."""
    lines = [line for line in resp.text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(line[len("data: "):]) for line in lines if line.startswith("data: {")]
    return chunks, lines


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
    assert usage["total_tokens"] == 18570
    assert usage["prompt_tokens_details"] == {"cached_tokens": 15000}


def test_chat_completion_without_sdk_usage_keeps_the_estimate(auth_ok, client):
    fake = _completion({"content": [_Text("OK")]})
    with patch.object(wrapper_main.claude_cli, "run_completion", fake):
        resp = client.post("/v1/chat/completions", json=CHAT)

    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["prompt_tokens"] != 18120
    assert usage["completion_tokens"] == len("OK") // 4
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens_details"] is None


def test_stream_final_chunk_reports_sdk_usage(auth_ok, client):
    body = {**CHAT, "stream": True, "stream_options": {"include_usage": True}}
    fake = _completion({"content": [_Text("OK")]}, RESULT)
    with patch.object(wrapper_main.claude_cli, "run_completion", fake):
        resp = client.post("/v1/chat/completions", json=body)

    chunks, lines = _sse(resp)
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert usage_chunks, resp.text
    # usage приходит ровно в последнем JSON-чанке, прямо перед [DONE]
    assert usage_chunks == [chunks[-1]]
    assert lines[-1] == "data: [DONE]"
    usage = usage_chunks[-1]["usage"]
    assert usage["prompt_tokens"] == 18120
    assert usage["completion_tokens"] == 450
    assert usage["total_tokens"] == 18570
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


def test_stream_without_include_usage_emits_no_usage(auth_ok, client):
    body = {**CHAT, "stream": True}
    fake = _completion({"content": [_Text("OK")]}, RESULT)
    with patch.object(wrapper_main.claude_cli, "run_completion", fake):
        resp = client.post("/v1/chat/completions", json=body)

    chunks, lines = _sse(resp)
    assert chunks
    assert all(c.get("usage") is None for c in chunks)
    assert lines[-1] == "data: [DONE]"


def test_stream_without_sdk_usage_falls_back_to_the_estimate(auth_ok, client):
    body = {**CHAT, "stream": True, "stream_options": {"include_usage": True}}
    fake = _completion({"content": [_Text("OK")]})
    with patch.object(wrapper_main.claude_cli, "run_completion", fake):
        resp = client.post("/v1/chat/completions", json=body)

    chunks, _ = _sse(resp)
    usage = chunks[-1]["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["prompt_tokens"] != 18120
    assert usage["completion_tokens"] == 1  # claude_cli.estimate_token_usage: не меньше 1
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens_details"] is None


def test_anthropic_messages_without_sdk_usage_fall_back_to_the_estimate(auth_ok, client):
    fake = _completion({"content": [_Text("OK")]})
    with patch.object(wrapper_main.claude_cli, "run_completion", fake):
        resp = client.post("/v1/messages", json=ANTHROPIC)

    usage = resp.json()["usage"]
    assert usage["input_tokens"] > 0
    assert usage["input_tokens"] != 120
    assert usage["output_tokens"] == len("OK") // 4
    assert usage["cache_creation_input_tokens"] is None
    assert usage["cache_read_input_tokens"] is None


def test_stream_sdk_failure_still_ends_well_formed_with_estimated_usage(auth_ok, client, caplog):
    # SDK упал: run_completion отдаёт только синтетический итог без usage. Ответ
    # остаётся целым, usage — оценка, и это не повод для предупреждения о дрейфе формата.
    body = {**CHAT, "stream": True, "stream_options": {"include_usage": True}}
    fake = _completion(SDK_ERROR)
    with caplog.at_level(logging.WARNING, logger="src.usage_report"):
        with patch.object(wrapper_main.claude_cli, "run_completion", fake):
            resp = client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    chunks, lines = _sse(resp)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert lines[-1] == "data: [DONE]"
    usage = chunks[-1]["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] == 1
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens_details"] is None
    assert [r for r in caplog.records if r.name == "src.usage_report"] == []
