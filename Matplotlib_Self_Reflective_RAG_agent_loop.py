"""基于 LangGraph 与 MCP 的 Matplotlib 课程教学助手 (自省 RAG 版本)。"""

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
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langgraph.graph import END, StateGraph

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


@st.cache_resource(show_spinner="正在连接本机 MCP 知识库服务...")
def get_mcp_tools():
    discovered_tools = asyncio.run(_discover_mcp_tools())
    if not discovered_tools:
        raise RuntimeError("MCP 服务未返回任何可用工具。")
    return discovered_tools


try:
    load_dotenv(ENV_FILE_PATH)
    llm = ChatOpenAI(
        model="deepseek-vl",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )
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


# ==========================================
# 此处开始重写：自省 RAG 的 State 与 节点函数
# ==========================================

class RagState(TypedDict):
    input: str
    search_result: list[Document]  # 【修改】改成保留 Document 对象（包含 page_content 和 metadata）
    assistant_result: str
    history: str
    retrieval_queries: list[str]
    user_role: str
    attempts: int


def retrieve_node(state: RagState):
    print("开始调用检索的方法...")
    query = state["input"]
    search_tool = next(tool for tool in tools if tool.name == "search_matplotlib_knowledge")
    docs = search_tool.invoke({"query": query})

    # 兼容 MCP 返回可能是 Document 对象或字典的情况
    search_result = []
    for doc in docs:
        if isinstance(doc, Document):
            search_result.append(doc)
        elif isinstance(doc, dict):
            # 字典转为 Document
            search_result.append(Document(page_content=doc.get("page_content") or doc.get("content") or str(doc), metadata=doc.get("metadata", {})))

    return {"search_result": search_result, "retrieval_queries": [query]}


def grade_node(state: RagState):
    print("开始调用评分的方法...")
    input_question = state["input"]
    search_result = state["search_result"]

    check_result = []
    for doc in search_result:
        # 【过滤】如果是图片，或者内容极长（图片 Base64 特征），直接跳过不进行相关性打分，减少不必要的调用
        if doc.metadata.get("type") == "image" or len(doc.page_content) > 5000:
            # 图片默认相关，直接添加进结果，传给大模型判断
            check_result.append(doc)
            continue

        # 纯文本/表格的评分逻辑
        system_prompt = f"你是一位数据评分员，你需要根据用户的问题和每个检索的结果进行判断，检索的结果是否与用户的问题相关。用户的问题：{input_question}，检索的结果：{doc.page_content}，相关的话则只输出 'yes'，不相关则只输出 'no'。"
        messages = [{"role": "system", "content": system_prompt}]
        res = llm.invoke(messages)
        if "yes" in res.content.lower():
            check_result.append(doc)

    return {"search_result": check_result}


def rewrite_node(state: RagState):
    print("开始调用重写问题的方法...")
    input_question = state["input"]
    system_prompt = f"你是一位心理专家，你主要负责理解用户的问题，把深层含义和浅层含义理解，并形成一个新问题。用户的问题：{input_question}"
    messages = [{"role": "system", "content": system_prompt}]
    res = llm.invoke(messages)
    # 重写后增加重试计数，防止死循环
    return {"input": res.content, "attempts": state.get("attempts", 0) + 1}



def check_search_result(state: RagState):
    # 如果过滤后没有相关文档，就去重写问题重新检索；否则直接生成
    if len(state["search_result"]) == 0:
        return "rewrite"
    return "generate"


def generate_node(state: RagState):
    print("开始调用大模型结果的方法...")
    input_question = state["input"]
    search_result = state["search_result"]
    history = state["history"]
    user_role = state["user_role"]

    # 【核心修改】根据不同数据类型，构造上下文
    context_parts = []
    for doc in search_result:
        doc_type = doc.metadata.get("type", "text")
        if doc_type == "image":
            # 多模态模型支持 markdown 格式直接解析 Base64 图片
            context_parts.append(f"![图片]({doc.page_content})")
        elif doc_type == "table":
            context_parts.append(f"【表格】\n{doc.page_content}")
        else:
            context_parts.append(f"【文本】\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    selected_examples = example_selector.select_examples({"question": input_question})
    example_text = "\n\n".join(f"用户：{ex['question']}\n助手：{ex['answer']}" for ex in selected_examples)

    role_guide = {
        "beginner": "请用通俗易懂的语言解释，多举生活中的例子。",
        "expert": "请提供深入的技术细节和代码优化建议。",
        "user": "请根据问题正常回答。",
    }

    system_prompt = f"""你是一位专业的 Matplotlib 教学助手。
当前用户水平：{role_guide.get(user_role, role_guide['user'])}
你可以参考以下相似问题的回答风格：
{example_text}

历史对话摘要：
{history}

当前用户问题：{input_question}
检索到的多模态课程资料（包含图片和表格）：
{context}

请充分结合上面的文本、表格和图片信息回答用户的问题。回答结束时请说“感谢您的提问！”。"""

    messages = [{"role": "system", "content": system_prompt}]
    res = llm.invoke(messages)
    return {"assistant_result": res.content}



def check_hallucinations(state: RagState):
    # 1. 如果重试超过2次，强制结束防止死循环
    if state.get("attempts", 0) >= 2:
        return "END"

    input_question = state["input"]
    search_result = state["search_result"]
    assistant_result = state["assistant_result"]

    system_prompt_1 = f"你是一位大模型工作者，你需要判断大模型的输出与检索的结果进行比对，判断大模型的回答是否存在幻觉，大模型的回答：{assistant_result}，检索的结果：{','.join(search_result)}。如果存在幻觉就输出 yes，不存在幻觉输出 no。"
    messages = [{"role": "system", "content": system_prompt_1}]
    res = llm.invoke(messages)

    if "yes" in res.content.lower():
        return "generate"  # 存在幻觉，重新生成

    system_prompt_2 = f"你是一位大模型工作者，你需要判断大模型的输出与用户的问题进行比对，判断大模型的回答是否能够解决用户的问题，大模型的回答：{assistant_result}，用户的问题：{input_question}。如果能解决输出 yes，不能解决输出 no。"
    messages = [{"role": "system", "content": system_prompt_2}]
    res = llm.invoke(messages)

    if "yes" in res.content.lower():
        return "END"  # 能解决，结束
    return "rewrite"  # 不能解决，重写问题


# ==========================================
# 构建 自省 RAG 图
# ==========================================

def build_self_reflective_rag():
    workflow = StateGraph(RagState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        check_search_result,
        {"rewrite": "rewrite", "generate": "generate"}
    )

    workflow.add_edge("rewrite", "retrieve")

    workflow.add_conditional_edges(
        "generate",
        check_hallucinations,
        {"generate": "generate", "END": END, "rewrite": "rewrite"}
    )

    return workflow.compile()


agent_app = build_self_reflective_rag()


# ==========================================
# 适配器（将 Redis 历史对齐到新的 RagState）
# ==========================================

def input_adapter(question: str, config: RunnableConfig) -> dict:
    session_id = config.get("configurable", {}).get("session_id", "default")
    user_role = config.get("configurable", {}).get("user_role", "beginner")

    # 获取历史
    history_obj = get_session_history(session_id)
    history_str = ""
    for msg in history_obj.messages:
        if isinstance(msg, HumanMessage):
            history_str += f"用户：{msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"助手：{msg.content}\n"

    return {
        "input": question,
        "history": history_str,
        "search_result": [],
        "assistant_result": "",
        "retrieval_queries": [],
        "user_role": user_role,
        "attempts": 0
    }


def output_adapter(state: RagState) -> dict:
    return {
        "answer": state["assistant_result"],
        "retrieval_queries": state["retrieval_queries"]
    }


# ==========================================
# Redis 历史工具函数（原封不动）
# ==========================================

def format_messages_for_summary(messages: list) -> str:
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
summary_chain = summary_prompt | llm | StrOutputParser()


def summarize_messages(messages: list, existing_summary: str | None = None) -> str:
    new_history = format_messages_for_summary(messages)
    if not new_history.strip():
        return existing_summary or "暂无历史"

    chat_history = new_history
    if existing_summary:
        chat_history = f"已有摘要：\n{existing_summary}\n\n新增对话：\n{new_history}"
    return summary_chain.invoke({"chat_history": chat_history})


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix="rag_chat:history:",
    )


def compact_redis_history(session_id: str) -> None:
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


def initialize_session_state() -> None:
    st.session_state.setdefault("role_for_prompt", "beginner")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_retrieved_docs", [])


# ==========================================
# Streamlit UI 主程序
# ==========================================

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
        for index, document in enumerate(st.session_state.last_retrieved_docs, start=1):
            source_name = Path(str(document.metadata.get("source", "未知来源"))).name
            page_number = int(document.metadata.get("page", 0)) + 1
            with st.expander(f"片段 {index}（{source_name} 第 {page_number} 页）"):
                doc_type = document.metadata.get("type", "text")
                content = document.page_content

                if doc_type == "image":
                    # 渲染 Base64 图片
                    st.image(content, caption="Multi-Vector 提取的原图", use_container_width=True)
                elif doc_type == "table":
                    # 渲染表格
                    st.markdown(content)
                else:
                    # 渲染文本
                    st.write(content[:300] + "..." if len(content) > 300 else content)
    else:
        st.caption("暂无检索结果")

    st.subheader("对话历史")
    if st.session_state.messages:
        for user_message, _ in st.session_state.messages:
            preview = user_message[:50] + "..." if len(user_message) > 50 else user_message
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
                # 1. 准备输入
                state_input = input_adapter(
                    prompt,
                    config={
                        "configurable": {
                            "session_id": SESSION_ID,
                            "user_role": st.session_state.role_for_prompt
                        }
                    }
                )

                # 2. 同步调用 LangGraph 图（使用 asyncio 跑）
                graph_result = asyncio.run(agent_app.ainvoke(state_input))

                # 3. 提取结果
                result = output_adapter(graph_result)
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