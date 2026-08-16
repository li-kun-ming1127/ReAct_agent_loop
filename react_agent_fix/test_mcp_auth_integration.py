import asyncio
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from langchain_mcp_adapters.client import MultiServerMCPClient


SECRET = os.environ.setdefault(
    "MCP_TOKEN_SECRET",
    "integration-test-secret-value-32-bytes",
)


def encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def create_token() -> str:
    payload = {
        "sub": "integration-user",
        "sid": "integration-browser",
        "kb": "matplotlib-course-v1",
        "aud": "matplotlib-course-mcp",
        "scopes": ["knowledge:read"],
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
    }
    payload_part = encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    )
    signature = hmac.new(
        SECRET.encode(), payload_part.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{payload_part}.{encode(signature)}"


async def discover(token: str):
    client = MultiServerMCPClient(
        {
            "test": {
                "url": "http://127.0.0.1:8010/mcp",
                "transport": "streamable_http",
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    )
    return await client.get_tools()


server = subprocess.Popen(
    [sys.executable, str(Path(__file__).with_name("run_test_mcp_server.py"))],
    cwd=Path(__file__).parent,
    env=os.environ.copy(),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
try:
    for _ in range(40):
        try:
            with socket.create_connection(("127.0.0.1", 8010), timeout=0.25):
                break
        except OSError:
            if server.poll() is not None:
                raise RuntimeError("Temporary MCP server exited before listening")
            time.sleep(0.25)
    else:
        raise RuntimeError("Temporary MCP server did not listen on port 8010")

    tools = asyncio.run(discover(create_token()))
    assert {tool.name for tool in tools} == {
        "list_matplotlib_course_documents",
        "search_matplotlib_knowledge",
    }

    try:
        asyncio.run(discover("invalid-token"))
    except Exception:
        pass
    else:
        raise AssertionError("Invalid MCP token unexpectedly discovered protected tools")
finally:
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)

print("MCP bearer authentication integration tests passed")
