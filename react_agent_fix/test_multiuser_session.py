import base64
import hashlib
import hmac
import json
import os
import time
import uuid

os.environ["MCP_TOKEN_SECRET"] = "test-only-secret-32-bytes-minimum-value"

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import HumanMessage

import mcp_server


def encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def make_token(**overrides) -> str:
    payload = {
        "sub": "alice",
        "sid": "browser-session-a",
        "kb": mcp_server.KNOWLEDGE_BASE_ID,
        "aud": mcp_server.MCP_TOKEN_AUDIENCE,
        "scopes": [mcp_server.MCP_REQUIRED_SCOPE],
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
    }
    payload.update(overrides)
    payload_part = encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    )
    signature = hmac.new(
        os.environ["MCP_TOKEN_SECRET"].encode(),
        payload_part.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_part}.{encode(signature)}"


valid = mcp_server.verify_context_token(make_token())
assert valid and valid["sub"] == "alice"
assert mcp_server.verify_context_token(make_token(exp=int(time.time()) - 1)) is None
assert mcp_server.verify_context_token(make_token(kb="other-kb")) is None
assert mcp_server.verify_context_token(make_token(iat="not-an-integer")) is None
assert mcp_server.verify_context_token(make_token(scopes="knowledge:read")) is None
assert mcp_server.verify_context_token("not-base64.not-base64") is None
original_token = make_token()
tampered = ("A" if original_token[0] != "A" else "B") + original_token[1:]
assert mcp_server.verify_context_token(tampered) is None

# Verify Redis key isolation and TTL using disposable, narrowly scoped keys.
prefix = f"rag_chat:test:{uuid.uuid4().hex}:history:"
history_a = RedisChatMessageHistory(
    session_id="kb=course:user=alice:session=browser-a",
    url="redis://localhost:6379/0",
    key_prefix=prefix,
    ttl=300,
)
history_b = RedisChatMessageHistory(
    session_id="kb=course:user=bob:session=browser-a",
    url="redis://localhost:6379/0",
    key_prefix=prefix,
    ttl=300,
)
try:
    history_a.add_message(HumanMessage(content="alice-private"))
    history_b.add_message(HumanMessage(content="bob-private"))
    assert [message.content for message in history_a.messages] == ["alice-private"]
    assert [message.content for message in history_b.messages] == ["bob-private"]
    assert 0 < history_a.redis_client.ttl(history_a.key) <= 300
    assert 0 < history_b.redis_client.ttl(history_b.key) <= 300
finally:
    history_a.clear()
    history_b.clear()

print("Multi-user token, Redis namespace, and TTL tests passed")
