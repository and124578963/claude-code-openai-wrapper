"""Regression: the caller's system prompt MUST reach ClaudeAgentOptions in a
form the SDK transport actually serializes.

The SDK's subprocess transport only emits a --system-prompt flag for THREE
shapes of ClaudeAgentOptions.system_prompt:
  - str                                   -> --system-prompt <text>
  - {"type": "file", "path": ...}         -> --system-prompt-file
  - {"type": "preset", ..., "append": ..} -> --append-system-prompt
Any other dict (notably the old {"type": "text", "text": ...}) is silently
dropped, so the caller's system prompt never reaches the model. These tests
pin the contract: a non-empty system prompt is passed as a plain str.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from src.claude_cli import ClaudeCodeCLI


class _OptionsRecorder:
    """Captures the ClaudeAgentOptions instance handed to query()."""

    def __init__(self) -> None:
        self.options: Any = None

    def fake_query(self, *, prompt: str, options: Any):  # noqa: ARG002
        self.options = options

        async def _agen():
            # Minimal ResultMessage-like dict so run_completion can finish.
            yield {"subtype": "success", "result": "ok"}

        return _agen()


async def _drive(cli: ClaudeCodeCLI, **kwargs: Any) -> None:
    async for _ in cli.run_completion(**kwargs):
        pass


@pytest.fixture
def cli_and_recorder(monkeypatch: pytest.MonkeyPatch):
    import src.claude_cli as mod

    recorder = _OptionsRecorder()
    monkeypatch.setattr(mod, "query", recorder.fake_query)
    cli = ClaudeCodeCLI(cwd="/tmp")
    return cli, recorder


@pytest.mark.asyncio
async def test_non_empty_system_prompt_passed_as_str(cli_and_recorder) -> None:
    cli, recorder = cli_and_recorder
    await _drive(
        cli,
        prompt="Human: hi",
        system_prompt="You are a pirate. Always say ARR.",
        stream=False,
    )
    sp = recorder.options.system_prompt
    assert isinstance(sp, str), (
        f"system_prompt must be a plain str so the SDK emits --system-prompt; "
        f"got {type(sp).__name__}: {sp!r}"
    )
    assert sp == "You are a pirate. Always say ARR."


@pytest.mark.asyncio
async def test_system_prompt_is_never_the_dropped_text_dict(cli_and_recorder) -> None:
    cli, recorder = cli_and_recorder
    await _drive(cli, prompt="Human: hi", system_prompt="x", stream=False)
    sp = recorder.options.system_prompt
    # The {"type": "text", ...} dict is the exact shape the transport drops.
    assert not (isinstance(sp, dict) and sp.get("type") == "text"), (
        "system_prompt must NOT be {'type': 'text', ...} — the SDK transport "
        "ignores it and the prompt never reaches the model."
    )


@pytest.mark.asyncio
async def test_empty_system_prompt_falls_back_to_claude_code_preset(cli_and_recorder) -> None:
    cli, recorder = cli_and_recorder
    await _drive(cli, prompt="Human: hi", system_prompt=None, stream=False)
    sp = recorder.options.system_prompt
    assert sp == {"type": "preset", "preset": "claude_code"}


def test_transport_serializes_str_but_not_text_dict() -> None:
    """Guard against an SDK upgrade silently changing the contract: confirm the
    installed transport emits --system-prompt for a str and emits NOTHING for
    a {'type': 'text'} dict (the reason the plain-str form is required)."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    src = inspect.getsource(subprocess_cli)
    assert 'isinstance(self._options.system_prompt, str)' in src
    assert '"--system-prompt"' in src
    # No branch recognizes a "text"-typed system prompt dict.
    assert 'get("type") == "text"' not in src
