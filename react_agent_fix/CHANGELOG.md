# ReAct Agent 修复记录

生成日期：2026-08-14

## 原始文件

- 来源：`D:\pycharm\langchain\LangChain_project\ReAct_agent_loop.py`
- 快照：`ReAct_agent_loop.original.py`
- SHA-256：`C11C2E51FD635DCDB5B604EF821F4772A4053D3E3537FB0542D9EEAC25AE2C16`
- `mcp_server.py` 未修改。

## 报错根因

`tool_vectorstore` 与 few-shot 示例选择器都使用 Chroma 的默认 collection。示例文档的 metadata
没有 `name` 字段，但 `get_relevant_tools()` 使用 `doc.metadata["name"]` 强制取值，因此 Agent
在调用 LLM/ToolNode 之前抛出 `KeyError('name')`。

终端中的 Transformers `[ERROR]` 是第三方包的文档检查提示，不是本次页面报错的原因。

## 两个版本共有的修复

1. 工具目录改用独立 collection：`react_agent_mcp_tools`。
2. 用 `doc.metadata.get("name")` 安全读取，并按已注册工具名过滤。
3. 不再重建 LangChain 已标准化的 `AIMessage.tool_calls`，避免损坏 `name/args/id` 协议结构。
4. 工具检索结果从 `ToolMessage` 恢复为 `Document`，继续供 Streamlit 侧栏展示。
5. 页面捕获异常时使用 `logger.exception()` 保存完整 traceback。
6. 每个运行版本会在自身旁边生成同名 `.log` 文件。

## 保留 ToolNode 版本

- 文件：`ReAct_agent_loop_with_toolnode.py`
- SHA-256：`B8BBEBCAEB10E5CC8B2ACEC0A47112E9BDC32A3B1A61376403982C6EBC3032AE`
- 流程：`agent -> ToolNode(tools) -> agent`。
- 所有工具统一经过 ToolNode，适合后续扩展标准多工具智能体。
- 相对原始文件的完整差异：`with_toolnode.diff`。

## 删除 ToolNode 版本

- 文件：`ReAct_agent_loop_without_toolnode.py`
- SHA-256：`C0AA4F4CB68BE36504AE367946A5E509F8807F804EF71001EA700A5C0E367CC1`
- 流程：`agent -> manual_tools_node -> agent`。
- 手动实现工具查找、未知工具处理、参数传递、异常捕获和 `ToolMessage` 封装。
- 相对原始文件的完整差异：`without_toolnode.diff`。

## 验证结果

- 使用项目 `venv312` 对两个版本运行 `python -m py_compile`，均通过。
- 检查确认保留版注册 `ToolNode(tools)`，删除版只注册 `manual_tools_node`。
- 未执行完整在线问答：完整运行依赖已启动的 Redis、MCP 服务、本地向量库和 DeepSeek API。

## 2026-08-14 第二次修复：MCP 工具仅支持异步调用

第一次修复后的保留版已经能够正确生成标准工具调用，但同步入口
`rag_chain_with_memory.invoke()` 会让 `ToolNode` 调用 `tool.invoke()`。当前
`langchain-mcp-adapters 0.3.2` 返回的 `StructuredTool` 仅实现异步调用，因此报错：

`NotImplementedError: StructuredTool does not support sync invocation.`

第二次修复内容：

1. Streamlit 同步入口通过独立事件循环运行 `rag_chain_with_memory.ainvoke()`。
2. 保留版的 `ToolNode` 因此自动进入异步执行路径并调用 `tool.ainvoke()`。
3. 无 ToolNode 版的 `manual_tools_node` 改为 `async def`，显式执行
   `await tool.ainvoke()`。
4. 归档第一次失败版本：`ReAct_agent_loop_with_toolnode.failed_sync.py`。
5. 归档对应完整 traceback：`ReAct_agent_loop_with_toolnode.failed_sync.log`。
6. 使用同版本依赖构造仅异步 `StructuredTool`，通过
   `StateGraph.ainvoke -> ToolNode -> tool.ainvoke` 冒烟测试，结果为 `ok:mcp`。

## 推荐运行方式

两个可运行版本也会以不同文件名放在原项目根目录，不覆盖原文件。先启动依赖服务，再在
`D:\pycharm\langchain\LangChain_project` 中运行所选版本：

```powershell
python mcp_server.py
streamlit run ReAct_agent_loop_with_toolnode.py
```

推荐采用保留 ToolNode 版本作为后续开发主线。
