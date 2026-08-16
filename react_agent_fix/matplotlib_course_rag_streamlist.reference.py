import os
from concurrent.futures.thread import ThreadPoolExecutor
from operator import itemgetter

from langchain_core.chat_history import InMemoryChatMessageHistory

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import streamlit as st
st.set_page_config(page_title="Matplotlib 教学助手", layout="wide")

import redis
r = redis.Redis(host='localhost', port=6379, db=0)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pdf2image import convert_from_path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory,RunnableConfig
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
import pytesseract
from langchain_core.tools import tool
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.embeddings import DashScopeEmbeddings
import time
import asyncio


# 👇 把函数定义放在这里
def get_cached_summary(session_id: str):
    key = f"rag_chat:summary:{session_id}"
    until_key = f"rag_chat:summary_until:{session_id}"
    summary = r.get(key)
    until = r.get(until_key)
    if summary and until:
        return summary.decode(), int(until)
    return None, 0

def set_cached_summary(session_id: str, summary: str, until: int):
    key = f"rag_chat:summary:{session_id}"
    until_key = f"rag_chat:summary_until:{session_id}"
    r.set(key, summary)
    r.set(until_key, str(until))


class AgentMonitorHandler(BaseCallbackHandler):
    """监控 Agent 的运行过程，可用于日志或 UI 状态更新"""

    def __init__(self):
        self.step_times = {}

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = serialized.get("name", "Unknown")
        print(f"⛓️ 链开始: {name} | 输入: {inputs}")
        if kwargs is not None and kwargs.get("run_id"):
            self.step_times[kwargs.get("run_id")] = time.time()


    def on_chain_end(self, outputs, **kwargs):
        run_id = kwargs.get("run_id")
        if kwargs is None:
            return
        run_id = kwargs.get("run_id")
        elapsed = time.time() - self.step_times.pop(run_id, 0)
        print(f"⛓️ 链结束，耗时 {elapsed:.2f}s，输出: {outputs}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"🔧 工具开始: {serialized['name']} | 输入: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"🔧 工具结束，输出: {output}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"🤖 LLM 开始调用，提示数量: {len(prompts)}")

    def on_llm_end(self, response, **kwargs):
        token_usage = response.llm_output.get("token_usage", {})
        print(f"🤖 LLM 调用结束，token 用量: {token_usage}")


# 1. 加载中文嵌入模型（自动下载，首次可能需要等待）
#    text2vec-base-chinese 是专门为中文优化的轻量模型
@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

"""deepseek版本的embedding模型，运行较慢更精准"""
# def get_embedding():
#     # 将模型名称替换为 DeepSeek 的 Embedding 模型
#     return HuggingFaceEmbeddings(
#         model_name="deepseek-ai/DeepSeek-Embedding-v1.5",
#         model_kwargs={'device': 'cpu'}, # 如果有 GPU 可改为 'cuda'
#         encode_kwargs={'normalize_embeddings': True} # 开启归一化，更适合余弦相似度检索
#     )


embedding = get_embedding()

@st.cache_resource
def load_vectordb():
    # 加载 PDF
    pdf_file_paths = [
        r"D:\pycharm\langchain\data\第一回：Matplotlib初识.pdf",
        r"D:\pycharm\langchain\data\第二回：艺术画笔见乾坤.pdf",
        r"D:\pycharm\langchain\data\第三回：布局格式定方圆.pdf"
    ]

    tesseract_cmd = r"D:\OCR_loader\tesseract.exe"
    poppler_path = r"D:\poppler\poppler-26.02.0\Library\bin"
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    docs = []

    def is_scanned_pdf(pages, check_first_n=3, min_text_len=10):
        for page in pages[:check_first_n]:
            if len(page.page_content.strip()) > min_text_len:
                return False
        return True

    def ocr_scanned_pdf(pdf_path):
        print(f"🔍 扫描PDF，启动OCR：{Path(pdf_path).name}")
        images = convert_from_path(pdf_path, dpi=400, poppler_path=poppler_path)
        ocr_doc_list = []
        for page_idx, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            doc = Document(
                page_content=text.strip(),
                metadata={
                    "source": pdf_path,
                    "page": page_idx,
                    "page_label": str(page_idx + 1)
                }
            )
            ocr_doc_list.append(doc)
        return ocr_doc_list

    for pdf_path in pdf_file_paths:
        print(f"\n📄 正在处理文件：{Path(pdf_path).name}")
        try:
            loader = PyPDFLoader(pdf_path)
            pdf_pages = loader.load()
            if is_scanned_pdf(pdf_pages):
                processed_docs = ocr_scanned_pdf(pdf_path)
            else:
                print(f"✅ 文字PDF，直接读取文本")
                processed_docs = pdf_pages
            docs.extend(processed_docs)
        except Exception as e:
            print(f"❌ 文件 {pdf_path} 处理失败：{str(e)}")

    chunk_size = 2000
    chunk_overlap = 100
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    print("=" * 30 + " 递归分割器 (Recursive) " + "=" * 30)
    r_result = r_splitter.split_documents(docs)

    persist_directory = 'docs/chroma/matplotlib/'
    vectordb = Chroma.from_documents(
        documents=r_result,
        embedding=embedding,
        persist_directory=persist_directory
    )

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    return vectordb, retriever

vectordb, retriever = load_vectordb()



# 1. 读取 DeepSeek 配置
env_file_path = r"D:\pycharm\langchain\Tool_Settings\Deepseek_API_Key.env"
load_dotenv(env_file_path)
print("LangSmith Key 存在:", os.getenv("LANGCHAIN_API_KEY") is not None)
api_key = os.getenv("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.0,
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com",
    timeout=60
)




# 自定义：根据文件名过滤文档（替代SelfQueryRetriever元数据筛选）
# 参数 use_global=True 时，返回全局检索结果；use_global=False 时，只返回第二回的片段
def filter_document_by_filename(query: str, use_global: bool = True):
    docs = retriever.invoke(query)

    if not use_global:
        target_keyword = "第二回：艺术画笔见乾坤.pdf"
        docs = [d for d in docs if target_keyword in d.metadata["source"]]

    return docs

# ================= 封装知识库检索为标准工具 =================
@tool
def search_matplotlib_knowledge(query: str) -> str:
    """查询Matplotlib课程讲义的知识库，获取相关的教学内容和代码示例。"""
    docs = filter_document_by_filename(query)
    # 缓存到 session_state 以供 UI 显示
    st.session_state.last_tool_docs = docs
    if not docs:
        return "未检索到相关的课程内容。"
    return "\n\n".join([f"参考片段{i+1}：\n{d.page_content}" for i, d in enumerate(docs)])

tools = [search_matplotlib_knowledge]
llm_with_tools = llm.bind_tools(tools)


# ================= 进阶：将自查询检索器嵌入 RAG 链 =================

# ================= 语义相似度示例选择器（优化版） =================
# 1. 定义示例库：贴合 Matplotlib 教学场景，可自行增删

@st.cache_resource
def get_example_selector():
    examples = [
        {
            "question": "Matplotlib是什么？",
            "answer": "Matplotlib是Python中最常用的可视化库，主要用于绘制各类静态、动态、交互式的统计图表。感谢您的提问！"
        },
        {
            "question": "如何绘制折线图？",
            "answer": "绘制折线图可以使用plt.plot()函数，传入x轴和y轴数据后调用plt.show()即可展示。感谢您的提问！"
        },
        {
            "question": "subplot怎么用？",
            "answer": "subplot用于在一个画布中创建多个子图，语法为plt.subplot(行数, 列数, 子图序号)，按顺序定位子图位置。感谢您的提问！"
        },
        {
            "question": "怎么设置图片标题？",
            "answer": "可以使用plt.title()函数设置图表标题，支持传入字体大小、颜色等样式参数。感谢您的提问！"
        }
    ]

    # 2. 创建语义相似度选择器（融合优化点）
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=embedding,           # 复用你已有的中文嵌入模型
        vectorstore_cls=Chroma,         # 使用 Chroma 支持持久化（吸收你代码的优点）
        k=2,                            # 每次选出最相似的2个示例
        input_keys=["question"]         # 只按question字段匹配，提升精度与速度
    )
    return example_selector

example_selector = get_example_selector()

# 3. 创建对话式少样本提示模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    # 单条示例的格式：用户提问 → AI回答
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{question}"),
        ("ai", "{answer}")
    ])
)


# ================= LangGraph 智能体工作流 =================
# 1. 定义工作流状态：核心是消息列表，用 add_messages 实现自动追加
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 摘要提示词
summary_prompt = ChatPromptTemplate.from_template(
    "请将以下对话历史总结为一段简洁的文字，保留关键信息（如用户意图、重要事实、决策等）：\n\n"
    "{chat_history}\n\n"
    "摘要："
)

# 摘要链
summary_chain = summary_prompt | llm | StrOutputParser()

def summarize_messages(messages, max_summary_tokens=500):
    """对消息列表进行摘要，返回纯文本摘要"""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"用户：{msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                formatted.append(f"助手：{msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                formatted.append(f"[助手请求调用工具：{msg.tool_calls}]")
    history_text = "\n".join(formatted)
    if not history_text.strip():
        return "暂无历史"
    summary = summary_chain.invoke({"chat_history": history_text})
    return summary

# ===== 新增：摘要持久化参数与函数 =====
SUMMARY_THRESHOLD = 6      # 超过此条数才触发摘要
RECENT_KEEP_COUNT = 4      # 保留最近完整消息数

def prepare_effective_messages(messages, session_id):
    """利用 Redis 缓存的摘要，增量生成压缩后的消息列表"""
    n = len(messages)
    cached_summary, cached_until = get_cached_summary(session_id)

    if n > SUMMARY_THRESHOLD:
        new_until = n - RECENT_KEEP_COUNT
        if new_until <= 0:
            summary_text = ""
        elif cached_summary and new_until == cached_until:
            # 摘要已覆盖到最新，直接复用
            summary_text = cached_summary
        else:
            # 需要更新摘要（处理新增消息）
            old_start = cached_until
            new_msgs = messages[old_start:new_until]
            new_part_summary = summarize_messages(new_msgs)
            if cached_summary:
                # 简单拼接（也可再用 LLM 合并，但会增加一次调用）
                summary_text = cached_summary + "\n" + new_part_summary
            else:
                summary_text = new_part_summary
            # 将新摘要及覆盖位置写入 Redis
            set_cached_summary(session_id, summary_text, new_until)

        # 构造有效消息：摘要系统消息 + 最近几轮对话
        recent_messages = messages[-RECENT_KEEP_COUNT:]
        summary_system_msg = SystemMessage(content=f"历史对话摘要：\n{summary_text}")
        effective = [summary_system_msg] + recent_messages
    else:
        effective = messages  # 消息数未超阈值，不压缩

    return effective


def compact_redis_history(session_id: str):
    history = get_session_history(session_id)
    messages = history.messages

    if len(messages) < SUMMARY_THRESHOLD:
        return

    old_messages = messages[:-RECENT_KEEP_COUNT]
    recent_messages = messages[-RECENT_KEEP_COUNT:]

    cached_summary, _ = get_cached_summary(session_id)
    new_summary = summarize_messages(old_messages)

    if cached_summary:
        new_summary = summary_chain.invoke({
            "chat_history": (
                f"已有摘要：\n{cached_summary}\n\n"
                f"新增历史摘要：\n{new_summary}"
            )
        })

    set_cached_summary(session_id, new_summary, 0)

    history.clear()
    history.add_messages(recent_messages)

# 2. 智能体节点：核心大脑，调用大模型判断是否调用工具
# --- 新增一个动态生成提示词的小函数 ---
def get_dynamic_prompt(user_role: str, example_text: str) -> str:
    base_prompt = "你是Matplotlib课程的专属教学助手，你善于根据Matplotlib的相关知识进行讲解。"

    # 定义角色对应的补充指令
    role_suffix = ""
    if user_role == "expert":
        role_suffix = "提供详细的技术响应、底层原理和进阶优化技巧，回答应清楚的体现逻辑。"
    elif user_role == "beginner":
        role_suffix = "简单地解释概念，用通俗易懂的比喻，尽量避免专业术语（行话）。"

    # 组装完整的系统提示词
    prompt_content = f"""{base_prompt}
{role_suffix}

如果用户问Matplotlib专业知识，请调用search_matplotlib_knowledge工具查询讲义内容再作答；
如果是问候、闲聊类问题，直接回答即可，无需调用工具。
回答尽量简洁，在回答的最后说一句"感谢您的提问！"
禁止编造知识库中没有的信息。

参考回答范例：
{example_text}"""
    return prompt_content

def agent_node(state: AgentState,config: RunnableConfig):
    effective_messages = state["messages"]
    user_question = effective_messages[-1].content

    selected_examples = example_selector.select_examples({"question": user_question})
    example_text = "\n\n".join([
        f"用户：{ex['question']}\n助手：{ex['answer']}"
        for ex in selected_examples
    ])

    # 👇 核心修改：从配置对象中安全读取角色，不再依赖 st.session_state
    current_role = config.get("configurable", {}).get("user_role", "beginner")
    system_prompt = get_dynamic_prompt(current_role, example_text)

    full_messages = [SystemMessage(content=system_prompt)] + effective_messages

    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}

# 3. 工具执行节点：LangGraph 内置组件，自动执行所有模型要求的工具
tool_node = ToolNode(tools)


# 4. 条件路由：判断下一步走向
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # 模型返回了工具调用指令 → 去执行工具
    if last_message.tool_calls:
        return "tools"
    # 没有工具调用 → 直接输出答案，结束流程
    return END


# 5. 搭建工作流图
workflow = StateGraph(AgentState)
# 添加两个核心节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
# 设置入口：从智能体节点开始
workflow.set_entry_point("agent")
# 添加条件分支：agent 节点根据判断走不同路线
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
# 工具执行完，必须回到 agent 节点，让模型根据工具结果生成回答
workflow.add_edge("tools", "agent")

# 6. 编译工作流，生成可直接调用的智能体对象
agent_app = workflow.compile()

# 导入 Redis 对话历史存储类
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import RunnableLambda

# 获取指定会话的历史记录（Redis 版）
def get_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379/0",
        key_prefix="rag_chat:history:"
    )

# 输入适配：拼接历史消息 + 当前提问，转成 LangGraph 格式
def input_adapter(input_dict: dict, config: RunnableConfig):
    session_id = config.get("configurable", {}).get("session_id", "default")
    history_messages = input_dict["messages"]
    current_msg = HumanMessage(content=input_dict["question"])
    all_messages = history_messages + [current_msg]

    # 使用缓存摘要生成压缩后的消息列表
    effective_messages = prepare_effective_messages(all_messages, session_id)
    return {"messages": effective_messages}

# 输出适配：从最终状态提取纯文本回答
def output_adapter(state: dict):
    return state["messages"][-1].content

# 组装基础调用链
base_agent_chain = (
    RunnableLambda(input_adapter)
    | agent_app
    | RunnableLambda(output_adapter)
)

# 用 RunnableWithMessageHistory 包装，复用 Redis 记忆
rag_chain_with_memory = RunnableWithMessageHistory(
    base_agent_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="messages"
)


# # ========== 5. 带记忆的测试运行 ==========
# # 注意：调用时除了输入问题，还必须要传入 session_id 来区分是谁在聊
# question1 = "这门课会重点讲 Matplotlib 的画图功能吗？"
# print("\n🟢 第一问：")
# answer1 = rag_chain_with_memory.invoke(
#     {"question": question1},
#     config={"configurable": {"session_id": "user123"}}
# )
# print(answer1)
#
# print("\n🟢 第二问：")
# question2 = "我刚刚问的是什么？"  # 它会记得上一问的内容！
# answer2 = rag_chain_with_memory.invoke(
#     {"question": question2},
#     config={"configurable": {"session_id": "user123"}}
# )
# print(answer2)

# ================== Streamlit UI ==================
if "role_for_prompt" not in st.session_state:
    st.session_state.role_for_prompt = "beginner"  # 默认给个新手
st.title("🤖 Matplotlib 课程智能助手")

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []          # 存储对话历史（用于显示）
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []  # 最近一次检索的文档
if "last_tool_docs" not in st.session_state:
    st.session_state.last_tool_docs = []
# 定义常量
SESSION_ID = "user123"  # 固定会话 ID（生产环境中应动态生成）

# ---------- 侧边栏：工具面板 ----------
with st.sidebar:
    st.subheader("⚙️ 控制面板")

    # 清空历史按钮
    if st.button("🔄 清空对话历史"):
        st.session_state.messages = []
        history = get_session_history(SESSION_ID)
        history.clear()
        # 删除 Redis 中的摘要缓存
        r.delete(f"rag_chat:summary:{SESSION_ID}", f"rag_chat:summary_until:{SESSION_ID}")
        st.rerun()

    # 选择角色设定代码
    st.subheader("👤 角色设定")
    user_role = st.radio(
        "请选择你的身份偏好：",
        options=["新手 (beginner)", "专家 (expert)", "通用 (user)"],
        index=0,
        key="user_role"  # 存到 session_state
    )
    # 记录实际用于提示词的角色值
    if "新手" in user_role:
        st.session_state.role_for_prompt = "beginner"
    elif "专家" in user_role:
        st.session_state.role_for_prompt = "expert"
    else:
        st.session_state.role_for_prompt = "user"


    # 显示检索源文档
    st.subheader("📚 最近检索的文档")
    if st.session_state.last_retrieved_docs:
        for i, doc in enumerate(st.session_state.last_retrieved_docs):
            with st.expander(f"片段 {i+1}（{Path(doc.metadata['source']).name} 第{doc.metadata['page']+1}页）"):
                st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    else:
        st.caption("暂无检索结果")

    # 显示对话历史（文本形式）
    st.subheader("📜 对话历史")
    if st.session_state.messages:
        for user_msg, bot_msg in st.session_state.messages:
            st.caption(f"**用户**: {user_msg[:50]}...")
    else:
        st.caption("暂无历史")

# ---------- 主界面：聊天区域 ----------
# 显示已有消息
for user_msg, bot_msg in st.session_state.messages:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(bot_msg)

# 接收用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    st.chat_message("user").write(prompt)

    # 调用 RAG 链（带记忆）
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            monitor = AgentMonitorHandler()
            try:
                answer = rag_chain_with_memory.invoke(
                    {"question": prompt},
                    config={
                        "configurable": {
                            "session_id": SESSION_ID,
                            "user_role": st.session_state.role_for_prompt  # 👈 从 UI 获取并传入后台
                        },
                        "callbacks": [monitor]
                    }
                )
            except Exception as e:
                answer = f"❌ 出错了：{str(e)}"
            st.write(answer)

    # 保存对话历史（前端展示用）
    st.session_state.messages.append((prompt, answer))

    # 保存检索文档
    st.session_state.last_retrieved_docs = st.session_state.get("last_tool_docs", [])

    # 压缩 Redis 中的历史（只保留摘要 + 最近消息）
    compact_redis_history(SESSION_ID)


