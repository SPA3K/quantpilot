"""Allow running the MCP server with `python -m quantpilot.mcp`."""

from quantpilot.mcp.server import mcp_server

if __name__ == "__main__":
    mcp_server.run()
