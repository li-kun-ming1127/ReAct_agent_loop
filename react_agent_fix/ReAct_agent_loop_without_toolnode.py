"""基于 LangGraph 与 MCP 的 Matplotlib 课程教学助手。"""

from __future__ import annotations

import asyncio
import os
import time
import json
import logging
import nest_asyncio
from pathlib import Path
from typing import Annotated, Any, TypedDict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import concurrent.futures
from langchain_chroma import Chroma

# Hugging Face 镜像必须在相关依赖导入前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from langchain_milvus import Milvus

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_CONNECTION = {"host": MILVUS_HOST, "port": MILVUS_PORT}

import redis
import streamlit as st
from dotenv import load_dotenv

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage,ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableWithMessageHistory,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


from mcp_server import (
    EMBEDDING_MODEL,
    MCP_HOST,
    MCP_PATH,
    MCP_PORT,
    SECOND_CHAPTER_FILENAME,
)

st.set_page_config(page_title="Matplotlib 教学助手", layout="wide")

ENV_FILE_PATH = Path(
    r"D:\pycharm\langchain\Tool_Settings\Deepseek_API_Key.env"
)
REDIS_URL = "redis://localhost:6379/0"
SESSION_ID = "user123"
SUMMARY_THRESHOLD = 6
RECENT_KEEP_COUNT = 4
MCP_SERVER_URL = f"http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}"

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def get_cached_summary(session_id: str) -> tuple[str | None, int]:
    """读取指定会话的摘要及累计压缩消息数。"""
    summary = redis_client.get(f"rag_chat:summary:{session_id}")
    compacted_count = redis_client.get(
        f"rag_chat:summary_until:{session_id}"
    )
    return summary, int(compacted_count or 0)


def set_cached_summary(
    session_id: str,
    summary: str,
    compacted_count: int,
) -> None:
    """将会话摘要和压缩进度写入 Redis。"""
    redis_client.set(f"rag_chat:summary:{session_id}", summary)
    redis_client.set(
        f"rag_chat:summary_until:{session_id}",
        str(compacted_count),
    )


class AgentMonitorHandler(BaseCallbackHandler):
    """记录 LangChain、模型和 MCP 工具的执行阶段与耗时。"""

    def __init__(self) -> None:
        self.step_times: dict[Any, float] = {}

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        serialized = serialized or {}
        name = serialized.get("name", "Unknown")
        run_id = kwargs.get("run_id")
        print(f"链开始：{name} | 输入：{inputs}")
        if run_id is not None:
            self.step_times[run_id] = time.time()

    def on_chain_end(self, outputs, **kwargs) -> None:
        run_id = kwargs.get("run_id")
        started_at = self.step_times.pop(run_id, None)
        elapsed = time.time() - started_at if started_at else 0.0
        print(f"链结束：耗时 {elapsed:.2f}s | 输出：{outputs}")

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        serialized = serialized or {}
        print(f"工具开始：{serialized.get('name', 'Unknown')} | 输入：{input_str}")

    def on_tool_end(self, output, **kwargs) -> None:
        print(f"工具结束：输出：{output}")

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        print(f"LLM 开始调用：提示数量 {len(prompts)}")

    def on_llm_end(self, response, **kwargs) -> None:
        llm_output = response.llm_output or {}
        print(f"LLM 调用结束：token 用量 {llm_output.get('token_usage', {})}")





@st.cache_resource(show_spinner="正在加载中文嵌入模型...")
def get_embedding() -> HuggingFaceEmbeddings:
    # ✅ 改成你刚才用 Git 下载的本地路径（注意前面的字母 r 不能漏掉！）
    local_model_path = r"D:\pycharm\langchain\models\text2vec-base-chinese"
    return HuggingFaceEmbeddings(model_name=local_model_path)


embedding = get_embedding()

# 保留客户端检索器和过滤函数，仅用于展示 MCP 本轮检索到的来源文档。
# 智能体绑定和实际调用的工具全部来自 MCP 服务端。


nest_asyncio.apply()

async def _discover_mcp_tools():
    client = MultiServerMCPClient(
        {
            "matplotlib_knowledge": {
                "url": MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        }
    )
    return await client.get_tools()

def _sync_discover_mcp_tools():
    """同步封装异步函数，避免 Streamlit 和 asyncio 循环冲突"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_discover_mcp_tools())
    finally:
        loop.close()


def run_coroutine_sync(coroutine):
    """在 Streamlit 同步脚本中运行异步 Agent，确保 MCP 工具走 ainvoke。"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


@st.cache_resource(show_spinner="正在连接本机 MCP 知识库服务...")
def get_mcp_tools():
    discovered_tools = _sync_discover_mcp_tools()
    if not discovered_tools:
        raise RuntimeError("MCP 服务未返回任何可用工具。")
    return discovered_tools


try:
    load_dotenv(ENV_FILE_PATH)  # 加载环境变量
    text_llm = ChatOpenAI(
        model="deepseek-chat",  # 纯文本模型，用于处理表格和最终答案
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )
    vision_llm = ChatOpenAI(
        model="deepseek-vl",  # 或其他兼容 OpenAI 接口的模型
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",  # DeepSeek 官方接口
        temperature=0,
    )
    tools = get_mcp_tools()
    # 构建工具描述向量库，用于解决“工具选择错误”
    tool_docs = [
        Document(
            page_content=tool.description or tool.name,
            metadata={"name": tool.name, "catalog": "mcp_tools"},
        )
        for tool in tools
    ]
    # 使用独立 collection，避免和 few-shot 示例的默认 Chroma collection 混在一起。
    tool_vectorstore = Chroma.from_documents(
        documents=tool_docs,
        embedding=embedding,
        collection_name="react_agent_mcp_tools",
    )


    def get_relevant_tools(query: str) -> list:
        docs = tool_vectorstore.similarity_search(query, k=2)
        available_tool_names = {tool.name for tool in tools}
        # metadata.get() 是本次 KeyError('name') 的直接修复；同时过滤非工具文档。
        relevant_tool_names = {
            name
            for doc in docs
            if (name := doc.metadata.get("name")) in available_tool_names
        }
        # 保证核心检索工具永不丢失
        relevant_tool_names.add("search_matplotlib_knowledge")
        return [tool for tool in tools if tool.name in relevant_tool_names]

except Exception as exc:
    st.error(
        "无法连接 Matplotlib MCP 服务。请先在另一个终端运行 "
        f"`python mcp_server.py`。服务地址：`{MCP_SERVER_URL}`\n\n"
        f"错误详情：{exc}"
    )
    st.stop()

if "search_matplotlib_knowledge" not in {tool.name for tool in tools}:
    st.error("MCP 服务未暴露 search_matplotlib_knowledge 工具。")
    st.stop()

def get_dynamic_prompt(user_role: str, example_text: str) -> str:
    role_guide = {
        "beginner": "请用通俗易懂的语言解释，多举生活中的例子。",
        "expert": "请提供深入的技术细节和代码优化建议。",
        "user": "请根据问题正常回答。",
    }
    return f"""你是一位专业的 Matplotlib 教学助手。
当前用户水平：{role_guide.get(user_role, role_guide['user'])}
你可以参考以下相似问题的回答风格：
{example_text}
如果用户询问 Matplotlib 相关知识，请使用 search_matplotlib_knowledge 工具查询课程讲义。
回答结束时请说“感谢您的提问！”。
"""

def table_to_text(table_content: str) -> str:
    """将 Markdown/HTML 表格转换为自然语言描述。"""
    prompt = f"请将以下表格转换为通顺的自然语言段落，突出关键数据和趋势：\n\n{table_content}"
    return text_llm.invoke(prompt).content

def image_to_text(image_base64: str) -> str:
    """用多模态模型为图片生成文字描述。"""
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "请详细描述这张图片中的内容，包括文字、图表、数据趋势等。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]
    )
    resp = vision_llm.invoke([msg])
    return resp.content




llm_with_tools = text_llm.bind_tools(tools)


@st.cache_resource(show_spinner=False)
def get_example_selector() -> SemanticSimilarityExampleSelector:
    """构建 Matplotlib 教学场景的语义相似示例选择器。"""
    examples = [
        {
            "question": "Matplotlib 是什么？",
            "answer": (
                "Matplotlib 是 Python 中常用的可视化库，可绘制静态、动态和"
                "交互式图表。感谢您的提问！"
            ),
        },
        {
            "question": "如何绘制折线图？",
            "answer": (
                "可使用 plt.plot() 传入 x、y 轴数据，再调用 plt.show() 展示。"
                "感谢您的提问！"
            ),
        },
        {
            "question": "subplot 怎么用？",
            "answer": (
                "plt.subplot(行数, 列数, 序号) 可在同一画布中定位多个子图。"
                "感谢您的提问！"
            ),
        },
        {
            "question": "怎样设置图片标题？",
            "answer": (
                "可使用 plt.title() 设置标题，并通过参数调整字号和颜色。"
                "感谢您的提问！"
            ),
        },
    ]
    return SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=embedding,
        vectorstore_cls=Chroma,
        k=2,
        input_keys=["question"],
    )


example_selector = get_example_selector()


class AgentState(TypedDict):
    """LangGraph 状态：消息列表及本轮用户消息在列表中的起始位置。"""
    messages: Annotated[list, add_messages]
    turn_start: int
    step_count: int  # 当前对话轮次
    max_steps: int   # 最大限制轮次
    retrieved_docs: list[Document]# 用于存储 RAG Fusion 融合后的文档列表，供前端展示


def format_messages_for_summary(messages: list) -> str:
    """将消息对象转换为适合摘要模型处理的纯文本。"""
    formatted: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            formatted.append(f"用户：{message.content}")
        elif isinstance(message, AIMessage):
            if message.content:
                formatted.append(f"助手：{message.content}")
            if message.tool_calls:
                formatted.append(f"[助手请求调用工具：{message.tool_calls}]")
    return "\n".join(formatted)

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "请用中文简洁地总结以下对话要点，突出技术问题与答案。"),
    ("human", "{chat_history}"),
])
summary_chain = summary_prompt | text_llm | StrOutputParser()

def summarize_messages(
    messages: list,
    existing_summary: str | None = None,
) -> str:
    """用一次模型调用合并既有摘要与待压缩消息，控制 API 调用频率。"""
    new_history = format_messages_for_summary(messages)
    if not new_history.strip():
        return existing_summary or "暂无历史"

    chat_history = new_history
    if existing_summary:
        chat_history = f"已有摘要：\n{existing_summary}\n\n新增对话：\n{new_history}"
    return summary_chain.invoke({"chat_history": chat_history})


def prepare_effective_messages(messages: list, session_id: str) -> list:
    """将持久化摘要作为系统消息置于近期对话之前。"""
    cached_summary, _ = get_cached_summary(session_id)
    if not cached_summary:
        return messages
    return [
        SystemMessage(content=f"历史对话摘要：\n{cached_summary}"),
        *messages,
    ]


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """返回指定会话的 Redis 消息历史。"""
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix="rag_chat:history:",
    )


def compact_redis_history(session_id: str) -> None:
    """将较旧消息压缩到 Redis 摘要，仅保留最近的完整消息。"""
    history = get_session_history(session_id)
    messages = history.messages
    if len(messages) <= SUMMARY_THRESHOLD:
        return

    old_messages = messages[:-RECENT_KEEP_COUNT]
    recent_messages = messages[-RECENT_KEEP_COUNT:]
    cached_summary, compacted_count = get_cached_summary(session_id)
    new_summary = summarize_messages(old_messages, cached_summary)
    set_cached_summary(
        session_id,
        new_summary,
        compacted_count + len(old_messages),
    )

    history.clear()
    history.add_messages(recent_messages)


logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    log_handler = logging.FileHandler(
        Path(__file__).with_suffix(".log"),
        encoding="utf-8",
    )
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(log_handler)

def safe_parse_tool_result(tool_output) -> list[dict]:
    """安全地把 MCP 工具的返回结果解析为文档字典列表。

    支持：
      - ToolMessage 对象（提取 .content）
      - 字符串 → 尝试 JSON 解析
      - 列表 → 直接使用
      - 其他 → 返回空列表
    """
    # 1. 提取字符串内容
    if isinstance(tool_output, ToolMessage):
        content = tool_output.content
    elif isinstance(tool_output, str):
        content = tool_output
    elif isinstance(tool_output, list):
        # 直接就是 dict 列表
        return tool_output
    else:
        logger.warning(f"未知工具返回值类型: {type(tool_output)}，内容: {str(tool_output)[:200]}")
        return []

    # 2. 尝试将字符串解析为 JSON
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"工具返回值不是合法 JSON: {e}，原始内容: {content[:200]}")
        return []

    # 3. 确保是列表
    if not isinstance(data, list):
        logger.warning(f"工具返回值应为列表，实际为: {type(data)}")
        return []

    return data


def documents_from_tool_messages(messages: list) -> list[Document]:
    """从本轮搜索 ToolMessage 中恢复来源文档，供 Streamlit 侧栏展示。"""
    documents: list[Document] = []
    seen_content: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", None) != "search_matplotlib_knowledge":
            continue
        for item in safe_parse_tool_result(message):
            if not isinstance(item, dict) or "content" not in item:
                continue
            content = item["content"]
            if content in seen_content:
                continue
            seen_content.add(content)
            documents.append(
                Document(
                    page_content=content,
                    metadata=item.get("metadata", {}),
                )
            )
    return documents

def rag_fusion_retrieve(original_question: str) -> tuple[str, list[Document], list[str]]:
    # 1. 生成多角度查询（保持不变）
    system_prompt = (
        "你是一位信息检索专家。请针对用户的问题，从不同维度（直接语义、关键实体、核心概念）"
        "生成3个不同的搜索查询，以便在知识库中获取最全面的资料。"
        "每个查询单独占一行，不要有多余解释或编号。"
    )
    res = text_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": original_question}
    ])
    queries = [q.strip() for q in res.content.split('\n') if q.strip()][:3]
    logger.info(f"RAG Fusion 生成查询: {queries}")

    # 2. 多路检索并融合
    search_tool = next(t for t in tools if t.name == "search_matplotlib_knowledge")
    doc_scores = defaultdict(float)
    doc_map = {}
    K = 60

    for query in queries:
        try:
            tool_output = search_tool.invoke({"query": query})
        except Exception as e:
            logger.warning(f"调用 MCP 搜索工具失败: {e}")
            continue

        docs_data = safe_parse_tool_result(tool_output)
        if not docs_data:
            logger.warning(f"查询 '{query}' 未返回有效文档")
            continue

        for rank, item in enumerate(docs_data):
            # 确保 item 是字典且包含必要字段
            if not isinstance(item, dict) or "content" not in item:
                continue
            content = item["content"]
            if content not in doc_map:
                doc_map[content] = Document(
                    page_content=content,
                    metadata=item.get("metadata", {})
                )
            doc_scores[content] += 1 / (K + rank + 1)

    # 3. 排序并截取 top_k
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_k = 12
    fusion_docs = [doc_map[content] for content, _ in sorted_docs[:top_k]]

    # 4. 分离文本和图片
    text_contexts = []
    image_bases = []
    for doc in fusion_docs:
        doc_type = doc.metadata.get("type", "text")
        if doc_type in ("text", "table"):
            text_contexts.append(doc.page_content)
        elif doc_type == "image":
            image_bases.append(doc.page_content)

    fusion_context = "\n\n".join(text_contexts)
    logger.info(f"RAG Fusion 最终得到 {len(fusion_docs)} 个文档片段")

    return fusion_context, fusion_docs, image_bases

def agent_node(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    user_question = next(
        (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), ""
    )

    # ---------- 改为动态筛选工具 ----------
    relevant_tools = get_relevant_tools(user_question)          # 语义搜索最相关的 2 个工具
    llm_with_relevant_tools = text_llm.bind_tools(relevant_tools)   # 只绑定这些工具
    # ------------------------------------

    selected_examples = example_selector.select_examples({"question": user_question})
    example_text = "\n\n".join(
        f"用户：{ex['question']}\n助手：{ex['answer']}" for ex in selected_examples
    )

    user_role = config.get("configurable", {}).get("user_role", "beginner")
    session_id = config.get("configurable", {}).get("session_id", "default")

    # 构建历史记录
    history_obj = get_session_history(session_id)
    tool_history = []
    for msg in history_obj.messages[-3:]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_history.append(f"历史工具调用：{msg.tool_calls}")
    tool_history_str = "\n".join(tool_history) if tool_history else "暂无历史工具调用"

    role_guide = {
        "beginner": "请用通俗易懂的语言解释，简单地解释概念，用通俗易懂的比喻，尽量避免专业术语（行话）。",
        "expert": "请提供深入的技术细节，提供详细的技术响应、底层原理和进阶优化技巧，回答应清楚的体现逻辑。",
        "user": "请根据问题正常回答。",
    }

    system_prompt = f"""你是一位专业的 Matplotlib 教学助手。
当前用户水平：{role_guide.get(user_role, role_guide['user'])}
你可以参考以下相似问题的回答风格：
{example_text}

【之前已经尝试过的工具调用】：
{tool_history_str}

如果用户询问 Matplotlib 相关知识，请**必须使用** search_matplotlib_knowledge 工具查询课程讲义。
回答结束时请说“感谢您的提问！”。"""

    # 使用动态绑定的工具进行推理
    response = llm_with_relevant_tools.invoke([SystemMessage(content=system_prompt), *messages])

    # AIMessage.tool_calls 已由 LangChain 标准化为 ToolNode 所需的
    # name / args / id 结构。不要重新组装，否则容易丢失协议字段。
    if response.tool_calls:
        print("工具调用原始结构：", type(response.tool_calls[0]), response.tool_calls[0])
        return {
            "messages": [response],
            "step_count": state["step_count"] + 1,
            "retrieved_docs": state.get("retrieved_docs", []),
        }
    else:
        # 没有触发工具调用的纯对话
        retrieved_docs = documents_from_tool_messages(
            state["messages"][state["turn_start"]:]
        )
        return {
            "messages": [response],
            "step_count": state["step_count"] + 1,
            "retrieved_docs": retrieved_docs or state.get("retrieved_docs", []),
        }
def should_continue(state: AgentState):
    if state["step_count"] >= state["max_steps"]:
        return END
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "tools"
    return END


async def manual_tools_node(state: AgentState) -> dict:
    """不依赖 ToolNode 的对照实现：逐个校验、调用并封装工具结果。"""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    tools_by_name = {tool.name: tool for tool in tools}
    results: list[ToolMessage] = []

    for index, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else None
        tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
        tool_call_id = (
            tool_call.get("id") if isinstance(tool_call, dict) else None
        ) or f"manual_call_{index}"

        if not tool_name:
            content = "工具调用缺少 name 字段。"
            logger.error("无效工具调用：%r", tool_call)
        elif tool_name not in tools_by_name:
            content = f"未知工具：{tool_name}"
            logger.error("模型请求了未注册工具：%s", tool_name)
        else:
            try:
                output = await tools_by_name[tool_name].ainvoke(tool_args)
                content = (
                    output
                    if isinstance(output, str)
                    else json.dumps(output, ensure_ascii=False, default=str)
                )
            except Exception as exc:
                logger.exception("手动执行工具失败：%s", tool_name)
                content = f"工具 {tool_name} 执行失败：{exc}"

        results.append(
            ToolMessage(
                content=content,
                name=tool_name or "invalid_tool_call",
                tool_call_id=tool_call_id,
            )
        )

    return {"messages": results}


def build_agent_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", manual_tools_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    # 工具调用无论成功或失败，都返回给 Agent 重新评估
    workflow.add_edge("tools", "agent")
    return workflow.compile()


agent_app = build_agent_app()


def input_adapter(input_dict: dict, config: RunnableConfig) -> dict:
    """合并 Redis 历史和当前问题，并转换为 LangGraph 输入。"""
    session_id = config.get("configurable", {}).get(
        "session_id",
        "default",
    )
    current_message = HumanMessage(content=input_dict["question"])
    all_messages = [*input_dict["messages"], current_message]
    effective_messages = prepare_effective_messages(all_messages, session_id)
    return {
        "messages": effective_messages,
        "turn_start": len(effective_messages) - 1,
        "step_count": 0,  # 重置步骤计数
        "max_steps": 6,  # 设定允许的最大死循环次数（例如 5 次）
        "retrieved_docs": [],  # 【新增】初始化融合文档列表
    }


def output_adapter(state: AgentState) -> dict:
    """提取最终回答和 RAG Fusion 融合后的来源文档，供 UI 展示。"""
    # 提取最终答案
    answer = state["messages"][-1].content
    # 直接提取融合后的文档列表
    retrieved_docs = state.get("retrieved_docs", [])

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs
    }

base_agent_chain = (
    RunnableLambda(input_adapter)
    | agent_app
    | RunnableLambda(output_adapter)
)
rag_chain_with_memory = RunnableWithMessageHistory(
    base_agent_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="messages",
    output_messages_key="answer",
)


def initialize_session_state() -> None:
    """初始化仅用于前端渲染的 Streamlit 会话状态。"""
    st.session_state.setdefault("role_for_prompt", "beginner")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_retrieved_docs", [])


initialize_session_state()
st.title("Matplotlib 课程智能助手")

with st.sidebar:
    st.subheader("控制面板")
    if st.button("清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieved_docs = []
        get_session_history(SESSION_ID).clear()
        redis_client.delete(
            f"rag_chat:summary:{SESSION_ID}",
            f"rag_chat:summary_until:{SESSION_ID}",
        )
        st.rerun()

    st.subheader("角色设定")
    role_label = st.radio(
        "请选择回答难度",
        options=["新手", "专家", "通用"],
        index=0,
        key="user_role",
    )
    st.session_state.role_for_prompt = {
        "新手": "beginner",
        "专家": "expert",
        "通用": "user",
    }[role_label]

    st.subheader("最近检索的文档")
    if st.session_state.last_retrieved_docs:
        for index, document in enumerate(
                st.session_state.last_retrieved_docs,
                start=1,
        ):
            source_name = Path(
                str(document.metadata.get("source", "未知来源"))
            ).name
            page_number = int(document.metadata.get("page", 0)) + 1
            with st.expander(
                    f"片段 {index}（{source_name} 第 {page_number} 页）"
            ):
                content = document.page_content
                doc_type = document.metadata.get("type", "text")
                if doc_type == "image":
                    preview = "[图片内容]"
                else:
                    preview = content[:300] + "..." if len(content) > 300 else content
                st.write(preview)
    else:
        st.caption("暂无检索结果")

    st.subheader("对话历史")
    if st.session_state.messages:
        for user_message, _ in st.session_state.messages:
            preview = (
                user_message[:50] + "..."
                if len(user_message) > 50
                else user_message
            )
            st.caption(f"用户：{preview}")
    else:
        st.caption("暂无历史")

for user_message, assistant_message in st.session_state.messages:
    with st.chat_message("user"):
        st.write(user_message)
    with st.chat_message("assistant"):
        st.write(assistant_message)

if prompt := st.chat_input("请输入你的问题..."):
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            monitor = AgentMonitorHandler()
            try:
                # MCP adapter 提供的是异步工具，手写节点也必须通过 ainvoke 执行。
                result = run_coroutine_sync(rag_chain_with_memory.ainvoke(
                        {"question": prompt},
                        config={
                            "configurable": {
                                "session_id": SESSION_ID,
                                "user_role": (
                                    st.session_state.role_for_prompt
                                ),
                            },
                            "callbacks": [monitor],
                        },
                    ))
                answer = result["answer"]

                # 【核心修改点】不再使用 filter_document_by_filename 二次检索
                # 而是直接取 RAG Fusion 处理过的融合文档源
                st.session_state.last_retrieved_docs = result.get("retrieved_docs", [])

            except Exception as exc:
                logger.exception("处理 Streamlit 请求失败，question=%r", prompt)
                answer = f"处理请求时出错：{exc}"
                st.session_state.last_retrieved_docs = []
            st.write(answer)

    st.session_state.messages.append((prompt, answer))
    try:
        compact_redis_history(SESSION_ID)
    except Exception as exc:
        st.warning(f"对话已完成，但 Redis 历史压缩失败：{exc}")


# '''启动方法
# cd D:\pycharm\langchain\LangChain_project
# streamlit run ReAct_agent_loop.py
# '''
