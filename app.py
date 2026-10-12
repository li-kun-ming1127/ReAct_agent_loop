"""基于 LangGraph 与 MCP 的 Matplotlib 课程教学助手。"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

# Hugging Face 镜像必须在相关依赖导入前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import redis
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
from langgraph.prebuilt import ToolNode

from mcp_server import (
    EMBEDDING_MODEL,
    MCP_HOST,
    MCP_PATH,
    MCP_PORT,
    SECOND_CHAPTER_FILENAME,
    get_retriever,
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
    """加载用于示例选择的中文嵌入模型。"""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


embedding = get_embedding()

# 保留客户端检索器和过滤函数，仅用于展示 MCP 本轮检索到的来源文档。
# 智能体绑定和实际调用的工具全部来自 MCP 服务端。
retriever = get_retriever()


def filter_document_by_filename(
    query: str,
    use_global: bool = True,
):
    """在客户端复现来源过滤，结果只用于 Streamlit 文档展示。"""
    documents = retriever.invoke(query)
    if not use_global:
        documents = [
            document
            for document in documents
            if SECOND_CHAPTER_FILENAME
            in str(document.metadata.get("source", ""))
        ]
    return documents


load_dotenv(ENV_FILE_PATH)
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    st.error(f"未在环境文件中找到 DEEPSEEK_API_KEY：{ENV_FILE_PATH}")
    st.stop()

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.0,
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com",
    timeout=60,
    max_retries=2,
)


async def _discover_mcp_tools():
    """连接本机 MCP 服务，并将服务端工具转换为 LangChain 工具。"""
    client = MultiServerMCPClient(
        {
            "matplotlib_knowledge": {
                "url": MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        }
    )
    return await client.get_tools()


@st.cache_resource(show_spinner="正在连接本机 MCP 知识库服务...")
def get_mcp_tools():
    """动态发现 MCP 工具，并在 Streamlit 进程内缓存工具定义。"""
    discovered_tools = asyncio.run(_discover_mcp_tools())
    if not discovered_tools:
        raise RuntimeError("MCP 服务未返回任何可用工具。")
    return discovered_tools


try:
    tools = get_mcp_tools()
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

llm_with_tools = llm.bind_tools(tools)


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


summary_prompt = ChatPromptTemplate.from_template(
    "请将以下对话历史总结为一段简洁文字，保留用户意图、重要事实和已做决定。\n\n"
    "{chat_history}\n\n摘要："
)
summary_chain = summary_prompt | llm | StrOutputParser()


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


def get_dynamic_prompt(user_role: str, example_text: str) -> str:
    """根据用户熟练度生成当前轮次的系统提示词。"""
    role_instructions = {
        "beginner": (
            "用通俗语言解释概念，分步骤说明，尽量减少不必要的专业术语。"
        ),
        "expert": (
            "提供严谨的技术说明、底层原理和进阶优化建议，清晰呈现推理逻辑。"
        ),
        "user": "以清晰、准确且难度适中的方式回答。",
    }
    role_instruction = role_instructions.get(
        user_role,
        role_instructions["beginner"],
    )
    return f"""你是 Matplotlib 课程的专属教学助手。
{role_instruction}

用户询问 Matplotlib 专业知识时，必须先调用 search_matplotlib_knowledge
查询讲义，再依据工具结果回答。问候或闲聊可直接回答，不必调用工具。
回答应简洁、准确；知识库没有的信息不得编造。回答结尾使用“感谢您的提问！”。

参考回答范例：
{example_text}"""


def agent_node(state: AgentState, config: RunnableConfig):
    """调用 DeepSeek，判断是直接回答还是请求 MCP 工具。"""
    messages = state["messages"]
    user_question = next(
        (
            message.content
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        "",
    )
    selected_examples = example_selector.select_examples(
        {"question": user_question}
    )
    example_text = "\n\n".join(
        f"用户：{example['question']}\n助手：{example['answer']}"
        for example in selected_examples
    )

    user_role = config.get("configurable", {}).get(
        "user_role",
        "beginner",
    )
    system_prompt = get_dynamic_prompt(user_role, example_text)
    response = llm_with_tools.invoke(
        [SystemMessage(content=system_prompt), *messages]
    )
    return {"messages": [response]}


def should_continue(state: AgentState):
    """当模型产生工具调用时进入工具节点，否则结束本轮工作流。"""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


def build_agent_app():
    """构建 Agent -> MCP Tools -> Agent 的 LangGraph 循环。"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
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
    }


def output_adapter(state: AgentState) -> dict:
    """提取最终回答和本轮 MCP 检索参数，供 UI 展示来源文档。"""
    retrieval_queries: list[str] = []
    for message in state["messages"][state["turn_start"] :]:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            if tool_call.get("name") != "search_matplotlib_knowledge":
                continue
            query = tool_call.get("args", {}).get("query")
            if query:
                retrieval_queries.append(query)

    return {
        "answer": state["messages"][-1].content,
        "retrieval_queries": retrieval_queries,
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
                st.write(content[:300] + "..." if len(content) > 300 else content)
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
                result = asyncio.run(
                    rag_chain_with_memory.ainvoke(
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
                    )
                )
                answer = result["answer"]
                retrieval_queries = result["retrieval_queries"]
                st.session_state.last_retrieved_docs = (
                    filter_document_by_filename(retrieval_queries[-1])
                    if retrieval_queries
                    else []
                )
            except Exception as exc:
                answer = f"处理请求时出错：{exc}"
                st.session_state.last_retrieved_docs = []
            st.write(answer)

    st.session_state.messages.append((prompt, answer))
    try:
        compact_redis_history(SESSION_ID)
    except Exception as exc:
        st.warning(f"对话已完成，但 Redis 历史压缩失败：{exc}")
