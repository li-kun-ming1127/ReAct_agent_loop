# 代码修改日志

本文件用于持续记录本对话中的代码修改。后续修改应追加新条目，不覆盖历史条目；每个条目至少记录
日期、修改前快照、SHA-256、根因、修改内容、验证结果和已知限制。

## 2026-08-14：私有 PDF Multi-Vector RAG 重构

### 输入与快照

- 参考旧版：`matplotlib_course_rag_streamlist.reference.py`
  - SHA-256：`9527FFB7C1CFBA84E9984F74509768FF0F8FC2D6B3772D17B6C03C39AF9302CD`
- 修改前客户端：`ReAct_agent_loop_with_toolnode.before_multivector.py`
  - SHA-256：`7AE59633CBBA3AE6F8BCB1A71E9E94E5848D46690153E346021EE78532C0AED7`
- 修改前服务端：`mcp_server.before_multivector.py`
  - SHA-256：`CD3017F106CA471899EFBDDD2331F2C0A69114C9D70B5B82951709416D5C9E70`
- 修改前运行日志：`ReAct_agent_loop_with_toolnode.before_multivector.log`

### 根因

1. 旧服务端把 Chroma 持久化到磁盘，却把父文档放在 `InMemoryStore`。重启后向量仍在、父文档映射已空，
   `MultiVectorRetriever` 因而固定返回 `[]`。
2. 第二章配置名为“艺术画笔见真章”，实际 PDF 是“艺术画笔见乾坤”。
3. 三份资料共 22 页，均属于扫描/低文本页面；仅靠 `pdfplumber.extract_text()` 无法读取主要内容。
4. MCP adapter 把结果包装在 `ToolMessage.artifact.structured_content.result`，客户端原解析器没有读取该字段。
5. 首次索引在用户提问路径中发生，造成约 200 秒等待；Agent 在空结果后还会重复调用搜索工具。

### 架构修改

- 使用 PDF 文件路径、大小、修改时间及模型配置生成索引指纹。
- 新索引写入独立的 `docs/multivector/matplotlib_v2`，旧 Chroma 不删除。
- Chroma 保存每个父内容的摘要表示和内容/文本化表示。
- `LocalFileStore` 持久保存父 `Document`，通过稳定 `doc_id` 与多个子向量关联。
- 检索采用 MMR：`k=8`、`fetch_k=32`、`lambda_mult=0.55`。
- 每页先判断文本量；低文本页使用 Poppler 300 DPI 渲染并通过 Tesseract OCR。
- 文本由文本模型摘要；表格转 Markdown 后由文本模型语义化；图片裁剪落盘并由可配置视觉模型分析。
- 视觉模型要求返回图片描述和 Markdown 表格列表，因此扫描表格可独立建立 `table` 父文档。
- 没有视觉模型时回退 OCR，并明确记录“扫描表格不会结构化恢复”。
- MCP 新增 `list_matplotlib_course_documents`，并保留 `search_matplotlib_knowledge`。
- MCP 启动时预热索引，避免第一次用户提问承担构建时间。
- Agent 在本轮已有 `ToolMessage` 时不再绑定工具，防止重复搜索。
- Streamlit 客户端解析 MCP artifact，并可在侧栏显示抽取图片及其文字描述。

### 输出文件

- 客户端：`ReAct_agent_loop_with_toolnode.py`
  - SHA-256：`BEA0CDAAD6B9BCE2359BD9966EB3125F0ED3919EB58DA8C60C3933BEE7092118`
- 服务端：`mcp_server.py`
  - SHA-256：`FA0BBD69505DA20194EACE51B089240E8F622FF2490F2F1F88968F002C7982E0`
- 客户端 diff：`2026-08-14_ReAct_multivector.diff`
- 服务端 diff：`2026-08-14_mcp_server_multivector.diff`
- 运行配置：`RAG_CONFIGURATION.md`

### 验证

- 两个 Python 文件均通过项目 `venv312` 的 `py_compile`。
- 第一份 4 页扫描 PDF 实测抽取：7 个文本块、6 个图片对象。
- 三份 22 页 PDF 完整离线索引：42 个文本父文档、38 个图片父文档、共 80 个父文档和 108 个向量。
- 查询“Matplotlib 课程内容和绘图基础”返回 8 个父文档。
- 第二个独立 Python 进程复用 108 个向量并再次返回 8 个父文档，验证 Docstore 可跨进程持久化。
- MCP artifact 单元测试成功恢复 `content` 和 `doc_id`。

### 已知限制

- 当前环境没有结构化扫描表格识别库。文本型 PDF 的表格可由 pdfplumber 提取；扫描型表格必须配置视觉模型，
  否则只能通过 OCR/图片描述检索，不能保证恢复单元格结构。
- DeepSeek 官方 V4 API 是纯文本模型，因此不再使用无效的 `deepseek-vl` 默认配置。
- 正式首次索引默认调用文本摘要模型，耗时和费用取决于内容量；可通过
  `ENABLE_MODEL_ENRICHMENT=0` 执行完全本地的 OCR/启发式摘要索引。

## 2026-08-14：实现真正的三查询 RAG Fusion

### 修改前快照

- `mcp_server.before_rag_fusion.py`
  - SHA-256：`FA0BBD69505DA20194EACE51B089240E8F622FF2490F2F1F88968F002C7982E0`
- `ReAct_agent_loop_with_toolnode.before_rag_fusion.py`
  - SHA-256：`BEA0CDAAD6B9BCE2359BD9966EB3125F0ED3919EB58DA8C60C3933BEE7092118`
- 服务端运行日志快照：`mcp_server.before_rag_fusion.log`

### 根因

- 原 `search_matplotlib_knowledge` 只执行一次
  `get_multi_vector_retriever().invoke(query)`，没有生成多个查询，也没有融合多个排名列表。

### 修改内容

- 使用已配置的文本 LLM 生成恰好 3 个语义相关但角度不同的查询。
- 支持 JSON 数组、JSON 对象、编号列表等模型输出，并对空值和重复查询去重。
- 模型未配置、调用失败或输出不合法时，记录异常并生成 3 个确定性回退查询，保证检索可用。
- 使用 3 个线程并发执行 Multi-Vector MMR 检索，每个查询最多取 8 个父文档。
- 以稳定 `doc_id` 去重，采用标准 RRF：`score(d) = Σ 1 / (60 + rank_i(d))`。
- 融合后返回前 8 个文档，并附带 `fusion_score`、`fusion_rank`、
  `matched_queries` 和每条查询中的原始排名。
- MCP 工具响应增加顶层 `queries` 与 `results`；Streamlit 客户端解析器已兼容该格式。
- Agent 提示词明确只调用一次检索工具；三查询生成与 RRF 全部由 MCP 服务端完成，保留 ToolNode 架构。

### 输出文件

- `mcp_server.py`
  - SHA-256：`1C449B6B152B0A757A3B874B26117F43839EA65585D849D02CCD0096B20B94B9`
- `ReAct_agent_loop_with_toolnode.py`
  - SHA-256：`3EBDEBE2CF2A4F75C18263C1EB9D869BC7D237E6275602650C3C56734F9796E3`
- 单元测试：`test_rag_fusion.py`
  - SHA-256：`3E9C2A74C74D913063093CCA9DCEBD33614BDBDB7AE85E78A77EBF35AF76C111`
- 差异文件：`2026-08-14_mcp_server_rag_fusion.diff`、
  `2026-08-14_ReAct_rag_fusion.diff`

### 验证

- 两个正式 Python 文件和测试文件均通过项目 `venv312` 的 `py_compile`。
- 查询解析、去重补全、RRF 分数与排序单元测试通过。
- 使用实际私有 PDF 向量库执行离线验证：生成 3 个查询，每个查询命中 8 个父文档，
  最终融合为 8 个结果；最高结果同时命中全部 3 个查询。
- 三路检索实测约 170.5 ms（不含首次索引构建和远程 LLM 查询生成）。

### 运行说明

- 修改后必须重启 `mcp_server.py`，再重启 Streamlit，旧进程不会自动加载新代码。
- 正式默认使用 LLM 生成查询；只有 `ENABLE_MODEL_ENRICHMENT=0`、模型未配置或调用失败时才进入回退路径。

## 2026-08-14：多用户会话、Redis TTL 与 MCP 访问鉴权

### 修改前快照

- `mcp_server.before_multiuser.py`
  - SHA-256：`69E8607AC3360B3527D14A326834B731DE1EF77B0BC3E26142890687E526FD47`
- `ReAct_agent_loop_with_toolnode.before_multiuser.py`
  - SHA-256：`3EBDEBE2CF2A4F75C18263C1EB9D869BC7D237E6275602650C3C56734F9796E3`

### 根因

- 客户端使用固定 `SESSION_ID="user123"`，所有浏览器和用户写入同一 Redis 历史。
- Redis 历史、摘要和摘要进度没有统一 TTL。
- Redis 键不包含用户 ID 和知识库 ID，无法形成租户边界。
- MCP Streamable HTTP 端点没有 Bearer 鉴权，工具调用也没有经过知识库权限校验。
- 模型和 MCP 工具在用户认证前加载，未授权访问者也可能消耗本机资源。

### 修改内容

- 新增令牌登录页；环境变量只保存高熵登录令牌的 SHA-256 摘要，不保存明文令牌。
- 每个 Streamlit 浏览器会话使用 `secrets.token_urlsafe(24)` 生成独立会话 ID。
- 会话上下文固定包含 `user_id / browser_session_id / knowledge_base_id`。
- Redis 历史键、摘要键和摘要进度键统一使用 `rag_chat:v2` 命名空间，键中包含知识库、用户和会话。
- `RedisChatMessageHistory` 与摘要记录统一配置 `REDIS_TTL_SECONDS`，默认 7 天；活跃会话刷新 TTL。
- 增加“新建会话”“清空当前会话”“退出登录”，操作范围不会越过当前命名空间。
- 客户端为 MCP 生成短期 HMAC-SHA256 上下文令牌；令牌声明包含用户、会话、知识库、作用域、签发时间和到期时间。
- FastMCP 使用 `TokenVerifier + AuthSettings` 保护 `/mcp`；无效、过期、被篡改、错误知识库或错误作用域令牌均被拒绝。
- MCP 工具从已验证认证上下文读取身份，身份令牌不会作为工具参数暴露给 LLM，保留原 ToolNode schema。
- 服务端缺少至少 32 字节的 `MCP_TOKEN_SECRET` 时拒绝启动，客户端缺少用户配置时在加载模型前停止。
- 当前注册知识库 ID 为 `matplotlib-course-v1`；用户权限由 `APP_USER_KB_ACCESS_JSON` 控制。

### 输出文件

- `mcp_server.py`
  - SHA-256：`9C9923307EDD25EC9DCB3274BA0D9F25FDBAE06ECD70FC7F23486C982A6EAF11`
- `ReAct_agent_loop_with_toolnode.py`
  - SHA-256：`F0909EC8109B70A674AB2C316E5E575B38C0D296B97D7111E7426CEEC34E6A20`
- `test_multiuser_session.py`
  - SHA-256：`F9B45457FC9D9A3795E4C8C2D05B07A51D3614805C6CC9ADE4EA68DCBD4F71A0`
- `test_mcp_auth_integration.py`
  - SHA-256：`35BF37074B868E15CFF21EA2993C2F1A366536E379E37B6D7E11DD7708FE1B15`
- `test_streamlit_multiuser.py`
  - SHA-256：`A245D5AD7F32F1103461E33B30FB4FB655B7B5B2E906ACAE1BED026E473170AA`
- 配置说明：`MULTIUSER_CONFIGURATION.md`
- 差异文件：`2026-08-14_mcp_server_multiuser.diff`、`2026-08-14_ReAct_multiuser.diff`

### 验证

- 所有正式文件和测试文件通过项目 `venv312` 的 `py_compile`。
- 有效、过期、篡改、格式错误、错误作用域和错误知识库令牌单元测试通过。
- Redis 实测验证两个用户在相同会话名下仍完全隔离，TTL 正确，测试键已全部清理。
- FastMCP 集成测试验证有效 Bearer 令牌可以发现工具，无效令牌无法发现受保护工具。
- Streamlit 1.61.1 `AppTest` 验证认证前只显示登录页；认证后生成浏览器会话 ID、选择授权知识库并成功连接受保护 MCP。
- 测试用 8000/8010 服务均已终止，未遗留监听端口。

### 配置要求与已知限制

- 正式运行前必须按 `MULTIUSER_CONFIGURATION.md` 配置 MCP 密钥、用户令牌摘要和知识库权限。
- 当前只有一个实际索引注册为 `matplotlib-course-v1`；逻辑隔离和鉴权已完成，新增第二套知识库仍需注册对应的独立索引路径。
- 当前 LangChain 输出 `RedisChatMessageHistory` 和 `RunnableWithMessageHistory` 的弃用警告；本次未扩大范围迁移记忆框架，现有行为已测试通过。

## 2026-08-15：AgentMonitorHandler 本轮 Token 统计

### 修改前快照

- `ReAct_agent_loop_with_toolnode.before_token_monitor.py`
  - SHA-256：`F0909EC8109B70A674AB2C316E5E575B38C0D296B97D7111E7426CEEC34E6A20`
- 修改前运行日志：`ReAct_agent_loop_with_toolnode.before_token_monitor.log`（存在时归档）。

### 修改内容

- 保留 `ChatOpenAI`、`bind_tools`、`ToolNode`、MCP、RAG Fusion、鉴权与 Redis 会话结构。
- `AgentMonitorHandler` 增加线程安全的本轮累计：输入、输出、总 token 和 LLM 调用次数。
- 依次兼容 `LLMResult.llm_output.token_usage`、`message.usage_metadata`、
  `message.response_metadata.token_usage/usage`。
- 支持 OpenAI/DeepSeek 的 `prompt_tokens/completion_tokens` 与 LangChain 的
  `input_tokens/output_tokens` 字段映射；缺少 `total_tokens` 时自动求和。
- 每次模型调用记录本次与本轮累计用量；每轮结束写入应用日志。
- Streamlit 侧栏显示本轮用量，并在回答完成后通过占位符即时更新。
- 新建会话、清空历史、退出登录或用户/知识库上下文变化时清空旧用量，避免跨会话混淆。

### 输出文件

- `ReAct_agent_loop_with_toolnode.py`
  - SHA-256：`746920AC93951FBC46A80B7E667E69023DB23205E8318309EFA19D8AE9FC2B6D`
- 单元测试：`test_agent_token_monitor.py`
  - SHA-256：`EB4655AB88D0F674444E7F4B9382ED3914CDA243F679C6537120947AE7082820`
- 差异文件：`2026-08-15_ReAct_token_monitor.diff`

### 验证

- 正式文件与测试文件通过项目 `venv312` 的 `py_compile`。
- 单元测试覆盖三种 token 元数据位置，累计结果为：输入 21、输出 10、合计 31、调用 3 次。
- 写回前确认 D 盘正式文件哈希仍与修改前快照一致，没有覆盖并发修改。
- 写回后尝试启动实际 MCP/Streamlit 链路；MCP 因 `MCP_TOKEN_SECRET` 未配置为至少 32 字节而按设计拒绝无鉴权启动，未绕过安全检查，因此本次未执行浏览器端实测。

## 2026-08-15：Token 预算、滚动摘要与完整轮次滑动窗口

### 修改前快照

- `ReAct_agent_loop_with_toolnode.before_context_window.py`
  - SHA-256：`746920AC93951FBC46A80B7E667E69023DB23205E8318309EFA19D8AE9FC2B6D`
- 修改前运行日志：`ReAct_agent_loop_with_toolnode.before_context_window.log`（存在时归档）。

### 修改内容

- 用 token 预算替代原来的固定消息条数压缩；每次主模型调用前均检查输入预算，包括动态 system、历史摘要、近期对话、工具结果和工具 schema。
- 默认采用 131072 token 的应用工作窗口：输入预算 106496、输出预留 8192、安全余量 8192、system/工具预留 8192。
- `ChatOpenAI.max_tokens` 与 `OUTPUT_RESERVE_TOKENS` 绑定，使预算预留和实际输出上限一致。
- 环境变量会自动钳制：输入预算不超过工作窗口可用值，Redis 历史目标不超过输入预算。
- 历史消息按“用户消息 + 随后的 AI/工具消息”组成完整轮次；压缩只淘汰最旧完整轮次，不拆开 AI 工具调用和对应 `ToolMessage`。
- 常规压缩至少保留最近 2 个完整轮次；硬预算不足时仍至少保留当前完整轮次。
- 被淘汰旧轮次与已有摘要合并；摘要输入和输出分别限制为 12000/2048 token。
- Redis 近期历史按 65536 token 或 80 条消息触发持久化摘要，保留原有 7 天 TTL 和用户/会话/知识库隔离。
- 单条工具消息上限为 12000 token，裁剪时保留头尾、`tool_call_id` 和消息元数据。
- 保留原有 ToolNode、MCP、RAG Fusion、多用户鉴权、Streamlit UI 和 `AgentMonitorHandler` token 统计。

### 输出文件

- `ReAct_agent_loop_with_toolnode.py`
  - SHA-256：`35E02A42BC454B40A53E679D45D7F52ACB699B76CB359559FB93A758305F482F`
- 单元测试：`test_context_window.py`
  - SHA-256：`8EE3F8B005D7DD1A3E559480E5325E508AB6AD1EF5156F0D86C8520C8D2FE205`
- 参数说明：`CONTEXT_WINDOW_CONFIGURATION.md`
  - SHA-256：`91BB24CD09EC5171229B9599DE8CA012112DAC6CE62C7CBFA1E3EC76C6467D51`
- 差异文件：`2026-08-15_ReAct_context_window.diff`

### 验证

- 正式文件与测试文件通过项目 `venv312` 的 `py_compile`。
- 窗口测试验证 system 首位、完整轮次淘汰、滚动摘要注入、最新完整问答保留，以及工具消息裁剪后 `tool_call_id` 不变。
- 原 `AgentMonitorHandler` 回归测试通过：累计输入 21、输出 10、总计 31、调用 3 次。
- 写回前确认 D 盘正式文件与本次修改前快照 SHA-256 一致。
- 因正式 MCP 鉴权配置未在本次测试环境中提供，没有绕过安全检查执行真实模型、Redis、MCP 和浏览器端到端测试。
