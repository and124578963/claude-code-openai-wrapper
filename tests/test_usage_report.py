"""Настоящий расход из ResultMessage SDK вместо оценки «длина / 4».

ResultMessage несёт usage за весь query (все раунды агента). Оценка по длине не видит
ни кэша, ни системного промпта Claude Code, ни раундов с инструментами, а кириллицу
недосчитывает вдвое — учёт расхода в ai-connector по ней ничего не значит.
"""
from src.usage_report import anthropic_usage, openai_usage, sdk_usage

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


def test_sdk_usage_sums_fresh_written_and_read_input():
    assert sdk_usage([{"content": []}, RESULT]) == {
        "prompt_tokens": 18120,
        "completion_tokens": 450,
        "total_tokens": 18570,
        "cached_tokens": 15000,
        "cache_creation_tokens": 3000,
        "fresh_input_tokens": 120,
    }


def test_sdk_usage_without_result_message_is_none():
    assert sdk_usage([{"content": []}, {"subtype": "init", "data": {}}]) is None


def test_sdk_usage_ignores_usage_outside_result_messages():
    # У сообщений ассистента usage бывает поразовым — итог за query только в ResultMessage.
    assert sdk_usage([{"content": [], "usage": {"input_tokens": 5, "output_tokens": 1}}]) is None


def test_sdk_usage_takes_the_last_result_message_with_usage():
    first = {**RESULT, "usage": {"input_tokens": 1, "output_tokens": 1}}
    error_without_usage = {"type": "result", "subtype": "error_during_execution", "is_error": True}
    assert sdk_usage([first, RESULT, error_without_usage])["prompt_tokens"] == 18120


def test_sdk_usage_with_zero_counts_is_none():
    assert sdk_usage([{**RESULT, "usage": {"input_tokens": 0, "output_tokens": 0}}]) is None


def test_sdk_usage_tolerates_missing_and_non_numeric_fields():
    message = {**RESULT, "usage": {"input_tokens": 10, "cache_read_input_tokens": None,
                                    "output_tokens": "7"}}
    assert sdk_usage([message]) == {
        "prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10,
        "cached_tokens": 0, "cache_creation_tokens": 0, "fresh_input_tokens": 10,
    }


def test_openai_usage_reports_cached_tokens_in_openai_shape():
    usage = openai_usage([RESULT], 5, 5)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (18120, 450, 18570)
    assert usage.prompt_tokens_details.cached_tokens == 15000


def test_openai_usage_falls_back_to_the_estimate_without_sdk_usage():
    usage = openai_usage([{"content": []}], 40, 9)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (40, 9, 49)
    assert usage.prompt_tokens_details is None


def test_anthropic_usage_keeps_anthropic_semantics():
    usage = anthropic_usage([RESULT], 5, 5)
    assert usage.input_tokens == 120
    assert usage.cache_creation_input_tokens == 3000
    assert usage.cache_read_input_tokens == 15000
    assert usage.output_tokens == 450


def test_anthropic_usage_falls_back_to_the_estimate():
    usage = anthropic_usage([], 40, 9)
    assert (usage.input_tokens, usage.output_tokens) == (40, 9)
    assert usage.cache_read_input_tokens is None
