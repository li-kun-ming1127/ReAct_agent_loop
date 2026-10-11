"""基于 LangGraph 与 MCP 的 Matplotlib 课程教学助手。"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
import re
import secrets
import time
import json
import logging
import nest_asyncio
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, TypedDict
from langchain_chroma import Chroma

# Hugging Face 镜像必须在相关依赖导入前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import redis
import streamlit as st
from dotenv import load_dotenv

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage,ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
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
from langchain_core.documents import Document


from mcp_server import (
    KNOWLEDGE_BASE_ID,
    MCP_REQUIRED_SCOPE,
    MCP_TOKEN_AUDIENCE,
    MCP_HOST,
    MCP_PATH,
    MCP_PORT,
)

st.set_page_config(page_title="Matplotlib 教学助手", layout="wide")

ENV_FILE_PATH = Path(
    r"D:\pycharm\langchain\Tool_Settings\Deepseek_API_Key.env"
)
load_dotenv(ENV_FILE_PATH)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_TTL_SECONDS = max(300, int(os.getenv("REDIS_TTL_SECONDS", "604800")))
REDIS_KEY_PREFIX = "rag_chat:v2"
MCP_TOKEN_SECRET = os.getenv("MCP_TOKEN_SECRET", "")
MCP_TOKEN_TTL_SECONDS = max(300, int(os.getenv("MCP_TOKEN_TTL_SECONDS", "3600")))
APP_USER_TOKENS_JSON = os.getenv("APP_USER_TOKENS_JSON", "{}")
APP_USER_KB_ACCESS_JSON = os.getenv("APP_USER_KB_ACCESS_JSON", "{}")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")
# The API currently advertises a larger context window, but this application
# keeps a bounded working window for latency, cost, tool schemas, and multi-agent growth.
MODEL_CONTEXT_TOKENS = max(
    32768,
    int(os.getenv("MODEL_CONTEXT_TOKENS", "131072")),
)
OUTPUT_RESERVE_TOKENS = max(
    1024,
    int(os.getenv("OUTPUT_RESERVE_TOKENS", "8192")),
)
SAFETY_MARGIN_TOKENS = max(
    2048,
    int(os.getenv("CONTEXT_SAFETY_MARGIN_TOKENS", "8192")),
)
SYSTEM_RESERVE_TOKENS = max(
    2048,
    int(os.getenv("SYSTEM_RESERVE_TOKENS", "8192")),
)
MAX_CONTEXT_INPUT_TOKENS = max(
    8192,
    MODEL_CONTEXT_TOKENS
    - OUTPUT_RESERVE_TOKENS
    - SAFETY_MARGIN_TOKENS
    - SYSTEM_RESERVE_TOKENS,
)
CONTEXT_INPUT_BUDGET_TOKENS = min(
    MAX_CONTEXT_INPUT_TOKENS,
    max(
        8192,
        int(
            os.getenv(
                "CONTEXT_INPUT_BUDGET_TOKENS",
                str(MAX_CONTEXT_INPUT_TOKENS),
            )
        ),
    ),
)
HISTORY_TARGET_TOKENS = min(
    CONTEXT_INPUT_BUDGET_TOKENS,
    max(8192, int(os.getenv("HISTORY_TARGET_TOKENS", "65536"))),
)
SUMMARY_INPUT_TOKENS = max(
    2048,
    int(os.getenv("SUMMARY_INPUT_TOKENS", "12000")),
)
SUMMARY_OUTPUT_TOKENS = max(
    512,
    int(os.getenv("SUMMARY_OUTPUT_TOKENS", "2048")),
)
MAX_HISTORY_MESSAGES = max(
    12,
    int(os.getenv("MAX_HISTORY_MESSAGES", "80")),
)
MIN_RECENT_TURNS = max(
    1,
    int(os.getenv("MIN_RECENT_TURNS", "2")),
)
TOKEN_ESTIMATE_CHARS_PER_TOKEN = max(
    1.5,
    float(os.getenv("TOKEN_ESTIMATE_CHARS_PER_TOKEN", "3.0")),
)
MAX_TOOL_MESSAGE_TOKENS = max(
    1024,
    int(os.getenv("MAX_TOOL_MESSAGE_TOKENS", "12000")),
)
MCP_SERVER_URL = f"http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}"

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


@dataclass(frozen=True)
class SessionContext:
    user_id: str
    browser_session_id: str
    knowledge_base_id: str

    @property
    def redis_session_id(self) -> str:
        return (
            f"kb={self.knowledge_base_id}:user={self.user_id}:"
            f"session={self.browser_session_id}"
        )


def _load_json_object(raw_value: str, variable_name: str) -> dict:
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{variable_name} 不是合法 JSON。") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{variable_name} 必须是 JSON 对象。")
    return value


def load_user_token_hashes() -> dict[str, str]:
    """Load user -> SHA-256(access token) mappings from environment."""
    configured = _load_json_object(APP_USER_TOKENS_JSON, "APP_USER_TOKENS_JSON")
    result: dict[str, str] = {}
    for user_id, token_hash in configured.items():
        normalized_user = str(user_id).strip()
        normalized_hash = str(token_hash).strip().lower()
        if not IDENTIFIER_PATTERN.fullmatch(normalized_user):
            raise RuntimeError(f"非法用户 ID：{normalized_user!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", normalized_hash):
            raise RuntimeError(
                f"用户 {normalized_user!r} 的访问令牌必须保存为 SHA-256 十六进制摘要。"
            )
        result[normalized_user] = normalized_hash
    return result


def allowed_knowledge_bases(user_id: str) -> list[str]:
    """Return the knowledge bases an authenticated user may select."""
    access_map = _load_json_object(
        APP_USER_KB_ACCESS_JSON,
        "APP_USER_KB_ACCESS_JSON",
    )
    configured = access_map.get(
        user_id,
        [] if access_map else [KNOWLEDGE_BASE_ID],
    )
    if not isinstance(configured, list):
        raise RuntimeError(f"用户 {user_id!r} 的知识库权限必须是 JSON 数组。")
    return [
        knowledge_base_id
        for knowledge_base_id in map(str, configured)
        if knowledge_base_id == KNOWLEDGE_BASE_ID
    ]


def require_authentication() -> str:
    """Render a fail-closed token login before loading models or MCP tools."""
    if len(MCP_TOKEN_SECRET.encode("utf-8")) < 32:
        st.error("服务未配置有效的 `MCP_TOKEN_SECRET`，已拒绝启动未鉴权客户端。")
        st.stop()

    try:
        user_hashes = load_user_token_hashes()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    if not user_hashes:
        st.error("未配置 `APP_USER_TOKENS_JSON`，当前没有可登录用户。")
        st.stop()

    st.session_state.setdefault(
        "browser_session_id",
        secrets.token_urlsafe(24),
    )
    authenticated_user_id = st.session_state.get("authenticated_user_id")
    authenticated_token_hash = st.session_state.get("authenticated_token_hash")
    if (
        authenticated_user_id in user_hashes
        and hmac.compare_digest(
            str(authenticated_token_hash),
            user_hashes[str(authenticated_user_id)],
        )
    ):
        return str(authenticated_user_id)

    st.title("Matplotlib 课程智能助手")
    with st.form("login_form"):
        user_id = st.text_input("用户 ID")
        access_token = st.text_input("访问令牌", type="password")
        submitted = st.form_submit_button("登录", width="stretch")
    if submitted:
        normalized_user = user_id.strip()
        expected_hash = user_hashes.get(normalized_user, "0" * 64)
        actual_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
        if normalized_user in user_hashes and hmac.compare_digest(actual_hash, expected_hash):
            st.session_state.authenticated_user_id = normalized_user
            st.session_state.authenticated_token_hash = expected_hash
            st.session_state.messages = []
            st.session_state.last_retrieved_docs = []
            st.rerun()
        st.error("用户 ID 或访问令牌错误。")
    st.stop()


def current_session_context(user_id: str) -> SessionContext:
    allowed = allowed_knowledge_bases(user_id)
    if not allowed:
        raise PermissionError("当前用户没有任何知识库访问权限。")
    selected = st.session_state.get("knowledge_base_id", allowed[0])
    if selected not in allowed:
        selected = allowed[0]
        st.session_state.knowledge_base_id = selected
    return SessionContext(
        user_id=user_id,
        browser_session_id=str(st.session_state.browser_session_id),
        knowledge_base_id=str(selected),
    )


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def create_mcp_context_token(context: SessionContext, token_window: int) -> str:
    """Create a short-lived HMAC token; credentials never enter the LLM prompt."""
    expires_at = (token_window + 2) * MCP_TOKEN_TTL_SECONDS
    payload = {
        "sub": context.user_id,
        "sid": context.browser_session_id,
        "kb": context.knowledge_base_id,
        "aud": MCP_TOKEN_AUDIENCE,
        "scopes": [MCP_REQUIRED_SCOPE],
        "iat": token_window * MCP_TOKEN_TTL_SECONDS,
        "exp": expires_at,
    }
    payload_part = _base64url(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(
        MCP_TOKEN_SECRET.encode("utf-8"),
        payload_part.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_part}.{_base64url(signature)}"


def get_cached_summary(session_id: str) -> tuple[str | None, int]:
    """读取指定会话的摘要及累计压缩消息数。"""
    summary = redis_client.get(f"{REDIS_KEY_PREFIX}:summary:{session_id}")
    compacted_count = redis_client.get(
        f"{REDIS_KEY_PREFIX}:summary_until:{session_id}"
    )
    return summary, int(compacted_count or 0)


def set_cached_summary(
    session_id: str,
    summary: str,
    compacted_count: int,
) -> None:
    """将会话摘要和压缩进度写入 Redis。"""
    with redis_client.pipeline() as pipeline:
        pipeline.setex(
            f"{REDIS_KEY_PREFIX}:summary:{session_id}",
            REDIS_TTL_SECONDS,
            summary,
        )
        pipeline.setex(
            f"{REDIS_KEY_PREFIX}:summary_until:{session_id}",
            REDIS_TTL_SECONDS,
            str(compacted_count),
        )
        pipeline.execute()


def refresh_session_ttl(session_id: str) -> None:
    """Refresh TTL for all Redis records belonging to one isolated session."""
    keys = (
        f"{REDIS_KEY_PREFIX}:history:{session_id}",
        f"{REDIS_KEY_PREFIX}:summary:{session_id}",
        f"{REDIS_KEY_PREFIX}:summary_until:{session_id}",
    )
    with redis_client.pipeline() as pipeline:
        for key in keys:
            pipeline.expire(key, REDIS_TTL_SECONDS)
        pipeline.execute()


class AgentMonitorHandler(BaseCallbackHandler):
    """记录执行阶段、耗时，并累计本轮全部 LLM 调用的 token。"""

    def __init__(self) -> None:
        self.step_times: dict[Any, float] = {}
        self._token_lock = Lock()
        self._token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
        }

    @staticmethod
    def _normalize_token_usage(raw_usage: Any) -> dict[str, int]:
        """兼容 OpenAI/DeepSeek 与 LangChain usage_metadata 字段名。"""
        if not isinstance(raw_usage, dict):
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        def non_negative_int(*keys: str) -> int:
            for key in keys:
                value = raw_usage.get(key)
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    continue
            return 0

        prompt_tokens = non_negative_int("prompt_tokens", "input_tokens")
        completion_tokens = non_negative_int(
            "completion_tokens",
            "output_tokens",
        )
        total_tokens = non_negative_int("total_tokens")
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @classmethod
    def _extract_token_usage(cls, response: Any) -> dict[str, int]:
        """优先读取 LLMResult，缺失时回退到生成消息的元数据。"""
        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            for key in ("token_usage", "usage"):
                normalized = cls._normalize_token_usage(llm_output.get(key))
                if normalized["total_tokens"] > 0:
                    return normalized

        for generation_group in getattr(response, "generations", []) or []:
            for generation in generation_group or []:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                candidates = [getattr(message, "usage_metadata", None)]
                response_metadata = getattr(message, "response_metadata", None)
                if isinstance(response_metadata, dict):
                    candidates.extend(
                        [
                            response_metadata.get("token_usage"),
                            response_metadata.get("usage"),
                        ]
                    )
                for candidate in candidates:
                    normalized = cls._normalize_token_usage(candidate)
                    if normalized["total_tokens"] > 0:
                        return normalized
        return cls._normalize_token_usage({})

    def token_usage(self) -> dict[str, int]:
        """返回线程安全的本轮 token 累计快照。"""
        with self._token_lock:
            return dict(self._token_usage)

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
        usage = self._extract_token_usage(response)
        with self._token_lock:
            self._token_usage["prompt_tokens"] += usage["prompt_tokens"]
            self._token_usage["completion_tokens"] += usage["completion_tokens"]
            self._token_usage["total_tokens"] += usage["total_tokens"]
            self._token_usage["llm_calls"] += 1
            accumulated = dict(self._token_usage)
        logging.getLogger(__name__).info(
            "LLM 调用结束：本次=%s，本轮累计=%s",
            usage,
            accumulated,
        )
        print(f"LLM 调用结束：本次 {usage} | 本轮累计 {accumulated}")


def format_token_usage(usage: dict[str, int]) -> str:
    """生成侧栏中稳定、紧凑的本轮 token 汇总文本。"""
    if not usage:
        return "暂无用量记录"
    return (
        "调用 {llm_calls} 次 · 输入 {prompt_tokens} · 输出 "
        "{completion_tokens} · 合计 {total_tokens} tokens"
    ).format(**usage)


AUTHENTICATED_USER_ID = require_authentication()
try:
    SESSION_CONTEXT = current_session_context(AUTHENTICATED_USER_ID)
except (RuntimeError, PermissionError) as exc:
    st.error(str(exc))
    st.stop()



@st.cache_resource(show_spinner="正在加载中文嵌入模型...")
def get_embedding() -> HuggingFaceEmbeddings:
    # ✅ 改成你刚才用 Git 下载的本地路径（注意前面的字母 r 不能漏掉！）
    local_model_path = r"D:\pycharm\langchain\models\text2vec-base-chinese"
    return HuggingFaceEmbeddings(model_name=local_model_path)


embedding = get_embedding()

# 保留客户端检索器和过滤函数，仅用于展示 MCP 本轮检索到的来源文档。
# 智能体绑定和实际调用的工具全部来自 MCP 服务端。


nest_asyncio.apply()

async def _discover_mcp_tools(
    context: SessionContext,
    token_window: int,
):
    context_token = create_mcp_context_token(context, token_window)
    client = MultiServerMCPClient(
        {
            "matplotlib_knowledge": {
                "url": MCP_SERVER_URL,
                "transport": "streamable_http",
                "headers": {
                    "Authorization": f"Bearer {context_token}",
                },
            }
        }
    )
    return await client.get_tools()

def _sync_discover_mcp_tools(
    context: SessionContext,
    token_window: int,
):
    """同步封装异步函数，避免 Streamlit 和 asyncio 循环冲突"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _discover_mcp_tools(context, token_window)
        )
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
def get_mcp_tools(
    context: SessionContext,
    token_window: int,
):
    discovered_tools = _sync_discover_mcp_tools(context, token_window)
    if not discovered_tools:
        raise RuntimeError("MCP 服务未返回任何可用工具。")
    return discovered_tools


try:
    text_llm = ChatOpenAI(
        model="deepseek-v4-flash",  # 当前 DeepSeek 文本/工具调用模型
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
        max_tokens=OUTPUT_RESERVE_TOKENS,
    )
    tools = get_mcp_tools(
        SESSION_CONTEXT,
        int(time.time()) // MCP_TOKEN_TTL_SECONDS,
    )

except Exception as exc:
    st.error(
        "无法连接 Matplotlib MCP 服务。请先在另一个终端运行 "
        f"`python mcp_server.py`。服务地址：`{MCP_SERVER_URL}`\n\n"
        f"错误详情：{exc}"
    )
    st.stop()

required_tools = {
    "search_matplotlib_knowledge",
    "list_matplotlib_course_documents",
}
missing_tools = required_tools - {tool.name for tool in tools}
if missing_tools:
    st.error(f"MCP 服务缺少必要工具：{', '.join(sorted(missing_tools))}")
    st.stop()

class ToolRouter:
    """Select a small, safe MCP tool set using an isolated semantic index."""

    CORE_TOOL_NAMES = frozenset({"search_matplotlib_knowledge"})

    def __init__(self, available_tools: list, embeddings: HuggingFaceEmbeddings):
        self.tools = {tool.name: tool for tool in available_tools}
        self._all_names = tuple(sorted(self.tools))
        tool_documents = [
            Document(
                page_content=f"{tool.name}\n{tool.description or ''}",
                metadata={"tool_name": tool.name, "catalog": "agent_tool_router_v1"},
            )
            for tool in available_tools
        ]
        self.vectorstore = Chroma(
            collection_name="react_agent_tool_router_v1",
            embedding_function=embeddings,
        )
        if tool_documents and self.vectorstore._collection.count() == 0:
            self.vectorstore.add_documents(tool_documents)

    @property
    def all_names(self) -> tuple[str, ...]:
        return self._all_names

    def select_tool_names(self, query: str) -> list[str]:
        """Return valid tool names, retaining a core search fallback."""
        if not query.strip() or not self.tools:
            return list(self._all_names)
        try:
            matches = self.vectorstore.similarity_search(query, k=min(3, len(self.tools)))
            selected = {
                str(doc.metadata.get("tool_name"))
                for doc in matches
                if doc.metadata.get("tool_name") in self.tools
            }
        except Exception:
            logger.exception("ToolRouter 检索失败，回退到全量工具")
            return list(self._all_names)
        selected.update(name for name in self.CORE_TOOL_NAMES if name in self.tools)
        return [name for name in self._all_names if name in selected] or list(self._all_names)

    def resolve(self, names: list[str] | tuple[str, ...] | None) -> list:
        """Resolve state names and fail closed to all configured tools."""
        valid = [name for name in (names or ()) if name in self.tools]
        if not valid:
            valid = list(self._all_names)
        return [self.tools[name] for name in valid]


tool_router = ToolRouter(tools, embedding)
tools_by_name = tool_router.tools
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
    retrieved_docs: list[Document]  # 用于存储 RAG Fusion 融合后的文档列表，供前端展示
    selected_tool_names: list[str]


def truncate_text(text: str, max_tokens: int) -> str:
    """Conservatively truncate text while retaining both its beginning and end."""
    max_chars = max(256, int(max_tokens * TOKEN_ESTIMATE_CHARS_PER_TOKEN))
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return f"{text[:head]}\n...[内容已按 token 预算截断]...\n{text[-tail:]}"


def estimate_message_tokens(messages: list, include_tools: bool = False) -> int:
    """Estimate message and optional tool-schema tokens before an LLM call."""
    return count_tokens_approximately(
        messages,
        chars_per_token=TOKEN_ESTIMATE_CHARS_PER_TOKEN,
        tools=tools if include_tools else None,
    )


def split_into_turns(messages: list) -> list[list]:
    """Group a human message and all following AI/tool messages into one turn."""
    turns: list[list] = []
    current: list = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, HumanMessage) and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def flatten_turns(turns: list[list]) -> list:
    return [message for turn in turns for message in turn]


def truncate_oversized_tool_messages(turns: list[list]) -> list[list]:
    """Bound tool payloads without breaking tool-call/message relationships."""
    bounded_turns: list[list] = []
    for turn in turns:
        bounded_turn: list = []
        for message in turn:
            if (
                isinstance(message, ToolMessage)
                and estimate_message_tokens([message]) > MAX_TOOL_MESSAGE_TOKENS
            ):
                bounded_content = truncate_text(
                    str(message.content),
                    MAX_TOOL_MESSAGE_TOKENS,
                )
                message = message.model_copy(update={"content": bounded_content})
            bounded_turn.append(message)
        bounded_turns.append(bounded_turn)
    return bounded_turns


def format_messages_for_summary(messages: list) -> str:
    """将完整对话轮次转换为受 token 预算限制的摘要输入。"""
    formatted: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            formatted.append(f"用户：{message.content}")
        elif isinstance(message, AIMessage):
            if message.content:
                formatted.append(f"助手：{message.content}")
            if message.tool_calls:
                calls = [
                    {
                        "tool": call.get("name", "unknown"),
                        "query": str(call.get("args", {}).get("query", ""))[:300],
                    }
                    for call in message.tool_calls
                ]
                formatted.append(
                    f"[助手工具调用摘要：{json.dumps(calls, ensure_ascii=False)}]"
                )
        elif isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", None) or "tool"
            content = truncate_text(str(message.content), MAX_TOOL_MESSAGE_TOKENS)
            formatted.append(f"工具[{tool_name}]：{content}")
    return "\n".join(formatted)

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "请用中文简洁地总结以下对话要点，突出技术问题与答案。"),
    ("human", "{chat_history}"),
])
summary_chain = (
    summary_prompt
    | text_llm.bind(max_tokens=SUMMARY_OUTPUT_TOKENS)
    | StrOutputParser()
)

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
    chat_history = truncate_text(chat_history, SUMMARY_INPUT_TOKENS)
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
        key_prefix=f"{REDIS_KEY_PREFIX}:history:",
        ttl=REDIS_TTL_SECONDS,
    )


def compact_redis_history(session_id: str) -> None:
    """按 token 预算压缩旧轮次，并保留最近的完整对话轮次。"""
    history = get_session_history(session_id)
    messages = history.messages
    turns = split_into_turns(messages)
    if not turns:
        return

    evicted_turns: list[list] = []
    while len(turns) > MIN_RECENT_TURNS:
        candidate = flatten_turns(turns)
        over_tokens = estimate_message_tokens(candidate) > HISTORY_TARGET_TOKENS
        over_count = len(candidate) > MAX_HISTORY_MESSAGES
        if not over_tokens and not over_count:
            break
        evicted_turns.append(turns.pop(0))

    if not evicted_turns:
        return

    old_messages = flatten_turns(evicted_turns)
    recent_messages = flatten_turns(turns)
    cached_summary, compacted_count = get_cached_summary(session_id)
    new_summary = summarize_messages(old_messages, cached_summary)
    set_cached_summary(
        session_id,
        new_summary,
        compacted_count + len(old_messages),
    )

    history.clear()
    history.add_messages(recent_messages)


def build_model_context(
    dynamic_system_prompt: str,
    messages: list,
    session_id: str,
) -> list:
    """Build a bounded context while preserving system messages and whole turns."""
    historical_system = [
        message for message in messages if isinstance(message, SystemMessage)
    ]
    turns = split_into_turns(messages)
    if not turns:
        return [SystemMessage(content=dynamic_system_prompt), *historical_system]

    def candidate_messages() -> list:
        return [
            SystemMessage(content=dynamic_system_prompt),
            *historical_system,
            *flatten_turns(turns),
        ]

    evicted_turns: list[list] = []
    while len(turns) > MIN_RECENT_TURNS and estimate_message_tokens(
        candidate_messages(), include_tools=True
    ) > CONTEXT_INPUT_BUDGET_TOKENS:
        evicted_turns.append(turns.pop(0))

    # The hard budget wins over the preferred minimum; always retain the
    # current complete turn, but an unusually large previous turn may be evicted.
    while len(turns) > 1 and estimate_message_tokens(
        candidate_messages(), include_tools=True
    ) > CONTEXT_INPUT_BUDGET_TOKENS:
        evicted_turns.append(turns.pop(0))

    if evicted_turns:
        cached_summary, _ = get_cached_summary(session_id)
        new_summary = summarize_messages(
            flatten_turns(evicted_turns),
            cached_summary,
        )
        historical_system = [
            SystemMessage(content=f"历史对话摘要：\n{new_summary}")
        ]

    turns = truncate_oversized_tool_messages(turns)
    context = candidate_messages()
    estimated_tokens = estimate_message_tokens(context, include_tools=True)
    if estimated_tokens > CONTEXT_INPUT_BUDGET_TOKENS:
        raise RuntimeError(
            "当前 system、工具定义和最近完整轮次超过上下文预算："
            f"estimated={estimated_tokens}, budget={CONTEXT_INPUT_BUDGET_TOKENS}。"
            "请降低单次工具结果、RAG top_k 或 MAX_TOOL_MESSAGE_TOKENS。"
        )
    logger.info(
        "上下文预算：estimated=%d budget=%d turns=%d",
        estimated_tokens,
        CONTEXT_INPUT_BUDGET_TOKENS,
        len(turns),
    )
    return context


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
    # MCP adapter 0.3.x 会把服务端结果放在 artifact.structured_content.result，
    # 同时把消息正文包装成 [{"type": "text", "text": "..."}]。
    if isinstance(tool_output, ToolMessage):
        artifact = getattr(tool_output, "artifact", None) or {}
        structured_content = artifact.get("structured_content", {})
        if "result" in structured_content:
            content = structured_content["result"]
        else:
            content = tool_output.content
    elif isinstance(tool_output, str):
        content = tool_output
    elif isinstance(tool_output, list):
        content = tool_output
    else:
        logger.warning("未知工具返回值类型: %s，内容: %s", type(tool_output), str(tool_output)[:200])
        return []

    if isinstance(content, list):
        if all(isinstance(item, dict) and "content" in item for item in content):
            return content
        text_blocks = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content = "\n".join(block for block in text_blocks if block)

    if isinstance(content, dict) and "result" in content:
        content = content["result"]

    try:
        data = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("工具返回值不是合法 JSON: %s，原始内容: %s", e, str(content)[:200])
        return []

    # RAG Fusion returns generated queries together with the fused document list.
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        data = data["results"]

    if not isinstance(data, list):
        logger.warning("工具返回值应为列表，实际为: %s", type(data))
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
            content = str(item["content"])
            metadata = item.get("metadata", {})
            document_key = str(metadata.get("doc_id") or content)
            if document_key in seen_content:
                continue
            seen_content.add(document_key)
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )
    return documents

def _tool_cache_key(session_id: str, tool_name: str, query: str) -> str:
    digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
    return f"{REDIS_KEY_PREFIX}:tool-cache:{session_id}:{tool_name}:{digest}"


def get_cached_tool_result(session_id: str, tool_name: str, query: str) -> str | None:
    if tool_name != "search_matplotlib_knowledge" or not query.strip():
        return None
    try:
        return redis_client.get(_tool_cache_key(session_id, tool_name, query))
    except Exception:
        logger.warning("读取工具结果缓存失败", exc_info=True)
        return None


def serialize_tool_content(content: Any) -> str:
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)

def set_cached_tool_result(session_id: str, tool_name: str, query: str, result: str) -> None:
    if tool_name != "search_matplotlib_knowledge" or not query.strip() or not result:
        return
    try:
        redis_client.setex(
            _tool_cache_key(session_id, tool_name, query),
            REDIS_TTL_SECONDS,
            result,
        )
    except Exception:
        logger.warning("写入工具结果缓存失败", exc_info=True)


def structured_tool_history(
    messages: list,
    current_query: str,
    cache_available: bool,
    limit: int = 3,
) -> str:
    """Summarize recent tool usage and the retrieval decision without raw calls."""
    records: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            args = call.get("args") if isinstance(call, dict) else {}
            record = {
                "tool": str(call.get("name", "unknown")),
                "query": str(args.get("query", ""))[:300] if isinstance(args, dict) else "",
                "result_count": 0,
            }
            call_id = call.get("id") if isinstance(call, dict) else None
            for following in messages[index + 1 :]:
                if isinstance(following, ToolMessage) and following.tool_call_id == call_id:
                    record["result_count"] = len(safe_parse_tool_result(following))
                    break
            records.append(record)

    recent = records[-limit:]
    latest_query = recent[-1]["query"] if recent else ""
    same_as_previous = bool(
        current_query.strip()
        and latest_query
        and current_query.strip().casefold() == latest_query.strip().casefold()
    )
    if cache_available:
        recommended_action = "reuse_cached_result_if_search_is_needed"
    elif same_as_previous and recent[-1]["result_count"] > 0:
        recommended_action = "reuse_recent_evidence_or_search_if_context_is_missing"
    else:
        recommended_action = "search_if_the_question_requires_course_evidence"
    return json.dumps(
        {
            "current_query": current_query[:300],
            "same_as_previous": same_as_previous,
            "cache_available": cache_available,
            "recommended_action": recommended_action,
            "recent_calls": recent,
        },
        ensure_ascii=False,
    )

def build_agent_system_prompt(
    user_role: str,
    example_text: str,
    tool_history: str,
    has_tool_result: bool,
) -> str:
    role_guide = {
        "beginner": "请用通俗易懂的语言解释，简单地解释概念，用通俗易懂的比喻，尽量避免专业术语（行话）。",
        "expert": "请提供深入的技术细节，提供详细的技术响应、底层原理和进阶优化技巧，回答应清楚地体现逻辑。",
        "user": "请根据问题正常回答。",
    }
    current_turn_rule = (
        "当前轮已经存在工具结果，直接依据结果回答，不要再次调用工具。"
        if has_tool_result
        else "如需课程知识，调用与问题最相关的工具；search_matplotlib_knowledge 每轮最多调用一次。"
    )
    return f"""你是一位专业的 Matplotlib 教学助手。
当前用户水平：{role_guide.get(user_role, role_guide['user'])}
你可以参考以下相似问题的回答风格：
{example_text}

最近工具调用摘要（结构化记录，不是原始调用对象）：
{tool_history}

RAG Fusion 由 MCP 服务端完成：search_matplotlib_knowledge 会在服务端生成多角度查询并进行融合，客户端不要拆成多次调用。
工具使用规则：
- 用户询问课程包含哪些 PDF、章节或索引状态时，调用 list_matplotlib_course_documents。
- 用户询问具体 Matplotlib 知识、代码、表格或图片内容时，调用 search_matplotlib_knowledge。
- {current_turn_rule}
- 不得声称查看过工具结果中不存在的 PDF 内容；没有结果时明确说明检索为空。
回答结束时请说“感谢您的提问！”。"""


async def selected_tool_node(state: AgentState, config: RunnableConfig):
    """Run ToolNode with exactly the tools bound to the current Agent step."""
    selected = tool_router.resolve(state.get("selected_tool_names"))
    last_message = state["messages"][-1]
    session_id = config.get("configurable", {}).get("session_id", "default")
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls or []) == 1:
        call = last_message.tool_calls[0]
        tool_name = str(call.get("name", ""))
        args = call.get("args", {}) if isinstance(call, dict) else {}
        query = str(args.get("query", "")) if isinstance(args, dict) else ""
        cached = get_cached_tool_result(session_id, tool_name, query)
        if cached:
            return {
                "messages": [
                    ToolMessage(
                        content=cached,
                        tool_call_id=call["id"],
                        name=tool_name,
                    )
                ]
            }

    result = await ToolNode(selected).ainvoke(state, config)
    for message in result.get("messages", []):
        if not isinstance(message, ToolMessage) or message.name != "search_matplotlib_knowledge":
            continue
        query = next(
            (
                str(call.get("args", {}).get("query", ""))
                for call in getattr(last_message, "tool_calls", [])
                if call.get("name") == message.name
            ),
            "",
        )
        set_cached_tool_result(session_id, message.name, query, serialize_tool_content(message.content))
    return result
async def agent_node(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    user_question = next(
        (str(msg.content) for msg in reversed(messages) if isinstance(msg, HumanMessage)), ""
    )
    current_turn_messages = messages[state["turn_start"]:]
    has_tool_result = any(isinstance(message, ToolMessage) for message in current_turn_messages)
    selected_examples = example_selector.select_examples({"question": user_question})
    example_text = "\n\n".join(
        f"用户：{ex['question']}\n助手：{ex['answer']}" for ex in selected_examples
    )
    user_role = config.get("configurable", {}).get("user_role", "beginner")
    session_id = config.get("configurable", {}).get("session_id", "default")
    selected_names = state.get("selected_tool_names") or tool_router.select_tool_names(user_question)
    cached_result = get_cached_tool_result(
        session_id, "search_matplotlib_knowledge", user_question
    )
    tool_history = structured_tool_history(
        get_session_history(session_id).messages[-12:],
        current_query=user_question,
        cache_available=bool(cached_result),
    )
    system_prompt = build_agent_system_prompt(
        user_role=user_role,
        example_text=example_text,
        tool_history=tool_history,
        has_tool_result=has_tool_result,
    )
    selected_tools = tool_router.resolve(selected_names)
    model = text_llm if has_tool_result else text_llm.bind_tools(selected_tools)
    model_messages = build_model_context(system_prompt, messages, session_id)
    response = await model.ainvoke(model_messages)
    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
        "selected_tool_names": selected_names,
        "retrieved_docs": (
            documents_from_tool_messages(current_turn_messages)
            or state.get("retrieved_docs", [])
        ),
    }


def should_continue(state: AgentState):
    if state["step_count"] >= state["max_steps"]:
        return END
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "tools"
    return END


def tool_validator_node(state: AgentState):
    """校验工具调用结果，如果发生异常或参数错误，发回给 Agent 纠正。"""
    last_message = state["messages"][-1]

    # 如果有工具调用，我们在这里先做一个简化校验，也可以留空交给 LangChain 原生报错处理
    # 但是在 LangGraph 中，如果 ToolNode 运行失败，它会引发异常
    # 因此，更标准的安全做法是在调用工具前做校验，或者捕获异常
    return {"messages": state["messages"]}


def build_agent_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", selected_tool_node)
    # 预留一个专门处理错误修正的节点（此处可以作为异常捕获的一环）
    # 为简化，我们直接利用 LangGraph 内部机制：出现工具执行错误时，ToolNode 返回的结果是包含 Error 的 ToolMessage
    # 所以在 ToolNode 执行完回到 Agent 时，Agent 就能看到错误

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
        "max_steps": 6,  # 设定允许的最大死循环次数
        "retrieved_docs": [],
        "selected_tool_names": tool_router.select_tool_names(str(input_dict["question"])),
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
    st.session_state.setdefault("last_token_usage", {})
    context_key = SESSION_CONTEXT.redis_session_id
    if st.session_state.get("active_context_key") != context_key:
        st.session_state.messages = []
        st.session_state.last_retrieved_docs = []
        st.session_state.last_token_usage = {}
        st.session_state.active_context_key = context_key


initialize_session_state()
st.title("Matplotlib 课程智能助手")

with st.sidebar:
    st.subheader("控制面板")
    st.caption(f"用户：{SESSION_CONTEXT.user_id}")
    allowed_kbs = allowed_knowledge_bases(SESSION_CONTEXT.user_id)
    st.selectbox(
        "知识库",
        options=allowed_kbs,
        key="knowledge_base_id",
    )

    if st.button("新建会话", width="stretch"):
        st.session_state.browser_session_id = secrets.token_urlsafe(24)
        st.session_state.messages = []
        st.session_state.last_retrieved_docs = []
        st.session_state.last_token_usage = {}
        st.session_state.pop("active_context_key", None)
        st.rerun()

    if st.button("清空对话历史", width="stretch"):
        st.session_state.messages = []
        st.session_state.last_retrieved_docs = []
        st.session_state.last_token_usage = {}
        session_id = SESSION_CONTEXT.redis_session_id
        get_session_history(session_id).clear()
        redis_client.delete(
            f"{REDIS_KEY_PREFIX}:summary:{session_id}",
            f"{REDIS_KEY_PREFIX}:summary_until:{session_id}",
        )
        st.rerun()

    if st.button("退出登录", width="stretch"):
        st.session_state.pop("authenticated_user_id", None)
        st.session_state.pop("authenticated_token_hash", None)
        st.session_state.pop("active_context_key", None)
        st.session_state.browser_session_id = secrets.token_urlsafe(24)
        st.session_state.messages = []
        st.session_state.last_retrieved_docs = []
        st.session_state.last_token_usage = {}
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

    st.subheader("本轮模型用量")
    token_usage_placeholder = st.empty()
    token_usage_placeholder.caption(
        format_token_usage(st.session_state.last_token_usage)
    )

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
                    asset_path = document.metadata.get("asset_path")
                    if asset_path and Path(str(asset_path)).is_file():
                        st.image(str(asset_path), caption=f"{source_name} 第 {page_number} 页")
                    preview = content[:500] + "..." if len(content) > 500 else content
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

if prompt := st.chat_input("请输入你的问题...", submit_mode="disable"):
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            monitor = AgentMonitorHandler()
            try:
                # MCP adapter 提供的是异步工具；整张图必须使用 ainvoke，
                # ToolNode 才会选择 tool.ainvoke() 而不是同步 tool.invoke()。
                result = run_coroutine_sync(rag_chain_with_memory.ainvoke(
                        {"question": prompt},
                        config={
                            "configurable": {
                                "session_id": SESSION_CONTEXT.redis_session_id,
                                "user_id": SESSION_CONTEXT.user_id,
                                "knowledge_base_id": (
                                    SESSION_CONTEXT.knowledge_base_id
                                ),
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
            st.session_state.last_token_usage = monitor.token_usage()
            logger.info(
                "本轮模型 token 汇总：session=%s usage=%s",
                SESSION_CONTEXT.redis_session_id,
                st.session_state.last_token_usage,
            )
            token_usage_placeholder.caption(
                format_token_usage(st.session_state.last_token_usage)
            )
            st.write(answer)

    st.session_state.messages.append((prompt, answer))
    try:
        compact_redis_history(SESSION_CONTEXT.redis_session_id)
        refresh_session_ttl(SESSION_CONTEXT.redis_session_id)
    except Exception as exc:
        st.warning(f"对话已完成，但 Redis 历史压缩失败：{exc}")


# '''启动方法
# cd D:\pycharm\langchain\LangChain_project
# streamlit run ReAct_agent_loop_with_toolnode.py
# '''
