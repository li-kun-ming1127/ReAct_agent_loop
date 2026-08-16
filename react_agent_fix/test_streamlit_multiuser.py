import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from streamlit.testing.v1 import AppTest


ACCESS_TOKEN = "test-user-access-token"
os.environ["MCP_TOKEN_SECRET"] = "streamlit-test-secret-value-32-bytes"
os.environ["APP_USER_TOKENS_JSON"] = json.dumps(
    {"alice": hashlib.sha256(ACCESS_TOKEN.encode()).hexdigest()}
)
os.environ["APP_USER_KB_ACCESS_JSON"] = json.dumps(
    {"alice": ["matplotlib-course-v1"]}
)
os.environ["REDIS_TTL_SECONDS"] = "900"
os.environ["TEST_MCP_PORT"] = "8000"

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
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.25):
                break
        except OSError:
            if server.poll() is not None:
                raise RuntimeError("Temporary MCP server exited before listening")
            time.sleep(0.25)
    else:
        raise RuntimeError("Temporary MCP server did not listen on port 8000")

    app = AppTest.from_file(
        str(Path(__file__).with_name("ReAct_agent_loop_with_toolnode.py"))
    ).run(timeout=30)
    assert [field.label for field in app.text_input] == ["用户 ID", "访问令牌"]
    app.text_input[0].input("alice")
    app.text_input[1].input(ACCESS_TOKEN)
    login_button = next(button for button in app.button if button.label == "登录")
    app = login_button.click().run(timeout=90)
    assert app.session_state.authenticated_user_id == "alice"
    assert app.session_state.authenticated_token_hash == hashlib.sha256(
        ACCESS_TOKEN.encode()
    ).hexdigest()
    assert app.session_state.browser_session_id
    assert app.selectbox[0].value == "matplotlib-course-v1"
finally:
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)

print("Streamlit login and isolated session context tests passed")
