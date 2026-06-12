"""Regression tests for the RequestValidationError handler.

A malformed request body (garbage bytes, wrong/absent content-type) makes
pydantic report the *raw request bytes* as the validation error's ``input``.
The handler echoed that ``input`` straight into a ``JSONResponse``, and bytes
aren't JSON-serializable, so ``json.dumps`` raised
``TypeError: Object of type bytes is not JSON serializable``. The handler
blew up and the client got a 500 Internal Server Error instead of a clean
422. See ``_json_safe_error_input`` in ``src/main.py``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import main as wrapper_main


CHAT_URL = "/v1/chat/completions"


@pytest.fixture
def client() -> TestClient:
    # raise_server_exceptions=True (the default) means that if the handler
    # regresses and raises again, this surfaces as a test error rather than a
    # silent 500 body.
    return TestClient(wrapper_main.app)


def _assert_clean_422(resp) -> dict:
    assert resp.status_code == 422, resp.text
    payload = resp.json()  # must be valid JSON, not the bare 500 body
    assert payload["error"]["type"] == "validation_error"
    assert isinstance(payload["error"]["details"], list)
    return payload


def test_raw_bytes_body_returns_422_not_500(client: TestClient) -> None:
    """Non-JSON byte body with a non-JSON content-type must yield a 422."""
    resp = client.post(
        CHAT_URL, content=b"not json", headers={"Content-Type": "text/plain"}
    )
    payload = _assert_clean_422(resp)
    # The raw bytes are surfaced as a decoded, JSON-serializable string.
    inputs = [d.get("input") for d in payload["error"]["details"]]
    assert "not json" in inputs


def test_data_bytes_body_returns_422(client: TestClient) -> None:
    """The exact repro from the bug report: POST with ``data=b"not json"``."""
    resp = client.post(CHAT_URL, data=b"not json")
    _assert_clean_422(resp)


def test_non_utf8_bytes_body_returns_422(client: TestClient) -> None:
    """Invalid UTF-8 must not crash the decode either (errors='replace')."""
    resp = client.post(
        CHAT_URL,
        content=b"\xff\xfe not even text",
        headers={"Content-Type": "application/octet-stream"},
    )
    _assert_clean_422(resp)


def test_broken_json_returns_422(client: TestClient) -> None:
    resp = client.post(
        CHAT_URL, content=b"{bad", headers={"Content-Type": "application/json"}
    )
    _assert_clean_422(resp)


def test_empty_body_returns_422(client: TestClient) -> None:
    resp = client.post(CHAT_URL, content=b"")
    _assert_clean_422(resp)


def test_json_safe_error_input_helper() -> None:
    """Unit guard for the serialization helper independent of HTTP plumbing."""
    f = wrapper_main._json_safe_error_input
    # bytes -> decoded string
    assert f(b"hi") == "hi"
    assert f(bytearray(b"hi")) == "hi"
    # invalid utf-8 is replaced, not raised
    assert isinstance(f(b"\xff\xfe"), str)
    # long bytes are truncated with a marker
    truncated = f(b"x" * 5000)
    assert truncated.endswith("...[truncated]")
    assert len(truncated) < 5000
    # already-serializable values pass through untouched
    assert f({"a": 1}) == {"a": 1}
    assert f(None) is None
    assert f("plain") == "plain"
    assert f(42) == 42
