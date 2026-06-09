"""Live check: does the system prompt actually reach the model?

Run manually (real auth, spends tokens):
    .venv/Scripts/python.exe tests/manual/live_system_prompt_check.py

Strategy: give a system prompt with a quirky, unmistakable instruction the
default claude_code persona would never follow on its own. If the system
prompt is applied, the model obeys (-> WALRUS). If it is silently dropped,
the model answers the question normally (-> Paris).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.main import app  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"
client = TestClient(app)


def check_system_prompt_applied() -> bool:
    print("[system] system='answer ONLY the word WALRUS', user='capital of France?'")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You must respond to EVERY user message with exactly the "
                        "single word WALRUS in uppercase. No punctuation, no other "
                        "words, regardless of what is asked."
                    ),
                },
                {"role": "user", "content": "What is the capital of France?"},
            ],
        },
    )
    content = resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else resp.text
    print(f"   status={resp.status_code} answer={content[:200]!r}")
    obeyed = "WALRUS" in content.upper()
    leaked = "PARIS" in content.upper()
    print(f"   -> system prompt {'APPLIED' if obeyed and not leaked else 'DROPPED/IGNORED'}")
    return obeyed and not leaked


def check_user_prompt_reaches() -> bool:
    print("[user] user contains a unique token; model must echo it back")
    token = "ZQX-9173-PLUM"
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": f"Repeat this token verbatim: {token}"}
            ],
        },
    )
    content = resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else resp.text
    print(f"   status={resp.status_code} answer={content[:200]!r}")
    ok = token in content
    print(f"   -> user prompt {'REACHES model' if ok else 'NOT reaching model'}")
    return ok


if __name__ == "__main__":
    results = [check_system_prompt_applied(), check_user_prompt_reaches()]
    print()
    print("PASSED" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)
