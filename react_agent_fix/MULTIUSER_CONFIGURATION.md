# 多用户会话配置

客户端和 MCP 服务端都从以下文件读取配置：

`D:\pycharm\langchain\Tool_Settings\Deepseek_API_Key.env`

## 必需配置

```dotenv
# 至少 32 字节，客户端用它签发短期上下文令牌，MCP 服务端用它验签。
MCP_TOKEN_SECRET=replace-with-a-random-secret-at-least-32-bytes

# 用户 ID -> 登录访问令牌的 SHA-256 摘要。不要在这里保存明文登录令牌。
APP_USER_TOKENS_JSON={"alice":"replace-with-64-char-sha256-hex"}

# 用户 ID -> 允许访问的知识库 ID。配置此变量后，未列出的用户默认无权限。
APP_USER_KB_ACCESS_JSON={"alice":["matplotlib-course-v1"]}
```

生成随机登录令牌、摘要和 MCP 密钥：

```powershell
python -c "import hashlib,secrets; token=secrets.token_urlsafe(32); print('登录令牌=',token); print('令牌摘要=',hashlib.sha256(token.encode()).hexdigest()); print('MCP密钥=',secrets.token_urlsafe(48))"
```

只有“令牌摘要”写入 `APP_USER_TOKENS_JSON`。用户登录时输入原始“登录令牌”。

## 可选配置

```dotenv
REDIS_URL=redis://localhost:6379/0
REDIS_TTL_SECONDS=604800
MCP_TOKEN_TTL_SECONDS=3600
MATPLOTLIB_KNOWLEDGE_BASE_ID=matplotlib-course-v1
```

- `REDIS_TTL_SECONDS` 默认 7 天，最小 300 秒。
- `MCP_TOKEN_TTL_SECONDS` 默认 1 小时，令牌按时间窗口轮换。
- Redis 键格式为：
  `rag_chat:v2:<record>:kb=<kb_id>:user=<user_id>:session=<browser_session_id>`。
- 当前 MCP 服务注册一个知识库 `matplotlib-course-v1`。增加第二个知识库时，必须在服务端注册独立索引后再授予用户权限。

## 启动顺序

```powershell
cd D:\pycharm\langchain\LangChain_project
python mcp_server.py
streamlit run ReAct_agent_loop_with_toolnode.py
```

服务端缺少有效 `MCP_TOKEN_SECRET` 时会拒绝启动；客户端缺少用户配置时只显示配置错误，不会加载模型或连接 MCP。
