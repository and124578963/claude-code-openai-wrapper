"""Настоящий расход токенов из сообщений Claude Agent SDK.

ResultMessage (run_completion отдаёт его словарём) несёт usage за весь query —
все раунды агента: input_tokens, cache_creation_input_tokens,
cache_read_input_tokens, output_tokens. До этого модуля wrapper отвечал оценкой
len(text) // 4, и учёт расхода в ai-connector не видел ни кэша, ни системного
промпта Claude Code, ни раундов с инструментами. Оценка осталась запасным путём —
на случай, когда SDK usage не прислал.
"""
from typing import Any, Dict, List, Optional

from src.models import AnthropicUsage, PromptTokensDetails, Usage


def _count(value: Any) -> int:
    """Неотрицательное целое поле usage; None, строка, bool — ноль."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _is_result(message: Dict[str, Any]) -> bool:
    # У ResultMessage, превращённого в dict, есть total_cost_usd; старый формат
    # CLI помечает итог type == "result". Только здесь usage — сумма за query.
    return "total_cost_usd" in message or message.get("type") == "result"


def sdk_usage(messages: List[Any]) -> Optional[Dict[str, int]]:
    """Счётчики последнего ResultMessage с usage; None — SDK их не прислал."""
    for message in reversed(messages):
        if not isinstance(message, dict) or not _is_result(message):
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        fresh = _count(usage.get("input_tokens"))
        written = _count(usage.get("cache_creation_input_tokens"))
        read = _count(usage.get("cache_read_input_tokens"))
        output = _count(usage.get("output_tokens"))
        prompt = fresh + written + read
        if prompt == 0 and output == 0:
            return None
        return {
            "prompt_tokens": prompt,
            "completion_tokens": output,
            "total_tokens": prompt + output,
            "cached_tokens": read,
            "cache_creation_tokens": written,
            "fresh_input_tokens": fresh,
        }
    return None


def openai_usage(messages: List[Any], estimated_prompt: int, estimated_completion: int) -> Usage:
    """Usage для /v1/chat/completions: вход — весь (с кэшем), кэш — в prompt_tokens_details."""
    real = sdk_usage(messages)
    if real is None:
        return Usage(
            prompt_tokens=estimated_prompt,
            completion_tokens=estimated_completion,
            total_tokens=estimated_prompt + estimated_completion,
        )
    return Usage(
        prompt_tokens=real["prompt_tokens"],
        completion_tokens=real["completion_tokens"],
        total_tokens=real["total_tokens"],
        prompt_tokens_details=PromptTokensDetails(cached_tokens=real["cached_tokens"]),
    )


def anthropic_usage(messages: List[Any], estimated_input: int, estimated_output: int) -> AnthropicUsage:
    """Usage для /v1/messages в смысле Anthropic: input_tokens — без кэша."""
    real = sdk_usage(messages)
    if real is None:
        return AnthropicUsage(input_tokens=estimated_input, output_tokens=estimated_output)
    return AnthropicUsage(
        input_tokens=real["fresh_input_tokens"],
        output_tokens=real["completion_tokens"],
        cache_creation_input_tokens=real["cache_creation_tokens"],
        cache_read_input_tokens=real["cached_tokens"],
    )
