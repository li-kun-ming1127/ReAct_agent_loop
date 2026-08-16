"""Matplotlib 课程知识库的 MCP 服务端 (支持表格与图片的多向量检索)"""

from __future__ import annotations

import base64
import io
import logging
import os
import uuid
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pdfplumber
import pytesseract
from PIL import Image
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp.server.fastmcp import FastMCP
from pdf2image import convert_from_path

# 【重要】为了将向量库和 Docstore 关联起来，需要导入 MultiVectorRetriever
from langchain_classic.retrievers import MultiVectorRetriever

LOGGER = logging.getLogger("matplotlib_mcp_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = BASE_DIR / "docs" / "chroma" / "matplotlib"
COLLECTION_NAME = "matplotlib_course"
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

# 目录路径配置
PDF_FILE_PATHS = (
    Path(r"D:\pycharm\langchain\data\第一回：Matplotlib初识.pdf"),
    Path(r"D:\pycharm\langchain\data\第二回：艺术画笔见真章.pdf"),
    Path(r"D:\pycharm\langchain\data\第三回：布局格式定方圆.pdf"),
)
TESSERACT_CMD = Path(r"D:\OCR_loader\tesseract.exe")
POPPLER_PATH = Path(r"D:\poppler\poppler-26.02.0\Library\bin")
SECOND_CHAPTER_FILENAME = "第二回：艺术画笔见真章.pdf"

MCP_HOST = "127.0.0.1"
MCP_PORT = 8000
MCP_PATH = "/mcp"

mcp = FastMCP("Matplotlib Course Knowledge", instructions="提供 Matplotlib 中文课程讲义的本地向量检索能力。",
              log_level="INFO", streamable_http_path=MCP_PATH)


@lru_cache(maxsize=1)
def get_embedding() -> HuggingFaceEmbeddings:
    LOGGER.info("正在加载嵌入模型：%s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def parse_pdf_multimodal(pdf_path: Path) -> list[dict]:
    """解析 PDF，分离出文本块、Markdown 表格、以及图片的 Base64 编码。"""
    elements = []

    # 1. 纯文本提取（用于 OCR 备选和搜索）
    pages = PyPDFLoader(str(pdf_path)).load()

    # 2. 使用 pdfplumber 提取表格和结构
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            # A. 提取纯文本块 (分块处理)
            text = page.extract_text() or ""
            if text.strip():
                # 使用原来的文本切分器切分文本块
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    if chunk.strip():
                        elements.append({
                            "type": "text",
                            "summary": chunk[:100] + "..." if len(chunk) > 100 else chunk,  # 生成的摘要
                            "content": chunk,  # 原始文本
                            "metadata": {"source": str(pdf_path), "page": page_idx, "type": "text"}
                        })

            # B. 提取表格，转换成 Markdown 格式
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 1:  # 有效表格
                    # 简单的表格转 Markdown 方法
                    markdown_table = "| " + " | ".join([str(cell) if cell else "" for cell in table[0]]) + " |\n"
                    markdown_table += "|" + "|".join([" --- "] * len(table[0])) + "|\n"
                    for row in table[1:]:
                        markdown_table += "| " + " | ".join([str(cell) if cell else "" for cell in row]) + " |\n"

                    elements.append({
                        "type": "table",
                        "summary": f"第 {page_idx + 1} 页包含关于 {table[0][0] if table[0][0] else '数据'} 的表格",
                        "content": markdown_table,
                        "metadata": {"source": str(pdf_path), "page": page_idx, "type": "table"}
                    })

            # C. 提取图片 (转为 Base64)
            # 注意：如果要精确定位图片坐标，需要 pdfplumber 的 Image 对象，但这里用整页切片作为简易方案。
            # 如果你想要更高精度的图片，可以结合 pdfminer 和 PIL 手动切片。
            # 这里我们使用 pdfplumber 把页面特定区域作为图片提取，简化为整页转存。
            if page.images:
                # 将当前页面转为图像（这块默认要配置 poppler 之前的路径）
                pil_images = convert_from_path(str(pdf_path), first_page=page_idx + 1, last_page=page_idx + 1,
                                               poppler_path=str(POPPLER_PATH))
                if pil_images:
                    # 转 Base64
                    buffered = io.BytesIO()
                    pil_images[0].save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    elements.append({
                        "type": "image",
                        "summary": f"第 {page_idx + 1} 页含有的插图",
                        "content": img_base64,  # 原始 Base64 图片
                        "metadata": {"source": str(pdf_path), "page": page_idx, "type": "image"}
                    })

    return elements


@lru_cache(maxsize=1)
def get_multi_vector_retriever() -> MultiVectorRetriever:
    """初始化或重建 MultiVectorRetriever 检索器"""
    embedding = get_embedding()
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # 1. 初始化向量库（保存摘要）
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    # 2. 初始化文档存储（保存原始文本、原始表格、原始 Base64 图片）
    # 你可以用 InMemoryStore（重启会丢），或者配置 RedisStore 实现长久保存
    docstore = InMemoryStore()

    # 3. 实例化 MultiVectorRetriever
    mv_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id"
    )

    # 4. 如果向量库中已经存在数据，则直接返回
    if vectorstore._collection.count() > 0:
        LOGGER.info("复用现有 Multi-Vector 向量库：%s", PERSIST_DIRECTORY)
        return mv_retriever

    # 5. 如果没有数据，进行全量解析和索引注入
    LOGGER.info("开始初次构建多向量检索库，预计需要几分钟...")
    for pdf_path in PDF_FILE_PATHS:
        LOGGER.info("正在解析：%s", pdf_path.name)
        elements = parse_pdf_multimodal(pdf_path)

        for el in elements:
            doc_id = str(uuid.uuid4())
            el["metadata"]["doc_id"] = doc_id

            # A. 生成摘要文档，存入 Chroma (Vectorstore)
            # 摘要的 Content 是 summary，但包含唯一的 doc_id 指向原始数据
            summary_doc = Document(page_content=el["summary"], metadata=el["metadata"])
            mv_retriever.vectorstore.add_documents([summary_doc])

            # B. 原始数据，存入 Docstore
            # 这里存的是纯文本、原始 Markdown 表格、或 Base64 图片字符串
            original_doc = Document(page_content=el["content"], metadata=el["metadata"])
            mv_retriever.docstore.mset([(doc_id, original_doc)])

    return mv_retriever


# 全局级检索器缓存
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_multi_vector_retriever()
    return _retriever


import json

@mcp.tool()
def search_matplotlib_knowledge(query: str) -> str:
    """..."""
    LOGGER.info("收到知识库检索请求：%s", query)
    retriever = get_retriever()
    documents = retriever.invoke(query)   # 返回 List[Document]

    if not documents:
        return json.dumps([])   # 空列表

    result = []
    for doc in documents:
        item = {
            "type": doc.metadata.get("type", "text"),
            "content": doc.page_content,
            "metadata": doc.metadata   # 包含 source, page, doc_id 等
        }
        result.append(item)

    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    LOGGER.info("MCP 服务启动：http://%s:%d%s", MCP_HOST, MCP_PORT, MCP_PATH)
    os.environ["MCP_HOST"] = MCP_HOST
    os.environ["MCP_PORT"] = str(MCP_PORT)

    # ✅ 增加异常捕获，防止 Windows 下按 Ctrl+C 关闭时报错
    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        LOGGER.info("收到停止信号，MCP 服务已正常关闭。")
    except Exception as e:
        LOGGER.error("MCP 服务运行中发生异常: %s", e)

# '''
# 启动方法
# cd D:\pycharm\langchain\LangChain_project
# python mcp_server.py
# '''