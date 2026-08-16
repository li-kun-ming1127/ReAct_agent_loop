import os

import uvicorn

import mcp_server


uvicorn.run(
    mcp_server.mcp.streamable_http_app(),
    host="127.0.0.1",
    port=int(os.getenv("TEST_MCP_PORT", "8010")),
    log_level="warning",
)
