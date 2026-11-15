"""Serve the same MCP tools over Streamable HTTP.

Start in its own PowerShell terminal:
    python 04_mcp/http_server.py

Connect with an MCP Client at http://127.0.0.1:8000/mcp.
Only binds localhost; do not expose publicly without auth/TLS.
"""
from server import mcp

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        stateless_http=True,
        json_response=True,
    )
