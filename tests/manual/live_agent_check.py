"""Live end-to-end check: the agent ACTUALLY uses tools through the wrapper.

Not a pytest test (needs real Claude auth + spends tokens) — run manually:

    .venv/Scripts/python.exe tests/manual/live_agent_check.py

Checks:
  1. enable_tools=true  -> agent reads a temp file with the Read tool and
     reports a magic string it could not have guessed.
  2. mcp_servers        -> agent calls get_secret on a per-request stdio MCP
     server (tests/manual/secret_mcp_server.py) and reports its secret.
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.main import app  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"
FILE_MAGIC = "MANGO-77-COMET"
MCP_MAGIC = "PINEAPPLE-42-ZEBRA"
MCP_SERVER_SCRIPT = Path(__file__).resolve().parent / "secret_mcp_server.py"

client = TestClient(app)


def check_builtin_tools() -> bool:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(f"The magic string is: {FILE_MAGIC}\n")
        path = f.name

    print(f"[1/2] built-in tools: asking agent to Read {path} ...")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Use the Read tool to read the file {path} "
                        "and repeat the magic string from it verbatim."
                    ),
                }
            ],
            "enable_tools": True,
        },
    )
    print(f"      status={resp.status_code}")
    content = resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else resp.text
    print(f"      answer: {content[:300]!r}")
    ok = resp.status_code == 200 and FILE_MAGIC in content
    print(f"      -> {'PASS' if ok else 'FAIL'} (magic string {'found' if ok else 'NOT found'})")
    return ok


def check_mcp_tools() -> bool:
    print("[2/2] MCP passthrough: asking agent to call mcp__secret__get_secret ...")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call the get_secret tool from the 'secret' MCP server "
                        "and repeat the secret phrase it returns verbatim."
                    ),
                }
            ],
            "mcp_servers": {
                "secret": {
                    "command": sys.executable,
                    "args": [str(MCP_SERVER_SCRIPT)],
                }
            },
        },
    )
    print(f"      status={resp.status_code}")
    content = resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else resp.text
    print(f"      answer: {content[:300]!r}")
    ok = resp.status_code == 200 and MCP_MAGIC in content
    print(f"      -> {'PASS' if ok else 'FAIL'} (MCP secret {'found' if ok else 'NOT found'})")
    return ok


if __name__ == "__main__":
    results = [check_builtin_tools(), check_mcp_tools()]
    print()
    print("PASSED" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)
