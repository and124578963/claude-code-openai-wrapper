"""Tiny stdio MCP server for live-testing the wrapper's mcp_servers passthrough.

Exposes one tool, get_secret, returning a magic phrase the model cannot guess.
Run by the Claude Agent SDK as a subprocess — see live_agent_check.py.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("secret")


@mcp.tool()
def get_secret() -> str:
    """Return the secret code phrase stored on this server."""
    return "PINEAPPLE-42-ZEBRA"


if __name__ == "__main__":
    mcp.run()  # stdio transport
