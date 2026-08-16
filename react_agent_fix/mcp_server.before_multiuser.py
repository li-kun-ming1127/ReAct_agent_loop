"""Matplotlib 私有 PDF 知识库 MCP 服务端。

索引采用 Multi-Vector Retriever：Chroma 保存摘要和文本化表示，
LocalFileStore 持久保存原始父文档。服务重启后两者仍能通过 doc_id 关联。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TypedDict

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pdfplumber
import pytesseract
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.retrievers.multi_vector import SearchType
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp.server.fastmcp import FastMCP
from pdf2image import convert_from_path
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE_PATH = Path(r"D:\pycharm\langchain\Tool_Settings\Deepseek_API_Key.env")

PDF_FILE_PATHS = (
    Path(r"D:\pycharm\langchain\data\第一回：Matplotlib初识.pdf"),
    Path(r"D:\pycharm\langchain\data\第二回：艺术画笔见乾坤.pdf"),
    Path(r"D:\pycharm\langchain\data\第三回：布局格式定方圆.pdf"),
)
SECOND_CHAPTER_FILENAME = "第二回：艺术画笔见乾坤.pdf"

TESSERACT_CMD = Path(r"D:\OCR_loader\tesseract.exe")
POPPLER_PATH = Path(r"D:\poppler\poppler-26.02.0\Library\bin")

INDEX_SCHEMA_VERSION = "multivector-v2"
INDEX_ROOT = BASE_DIR / "docs" / "multivector" / "matplotlib_v2"
CHROMA_DIRECTORY = INDEX_ROOT / "chroma"
DOCSTORE_DIRECTORY = INDEX_ROOT / "docstore"
ASSET_DIRECTORY = INDEX_ROOT / "assets"
MODEL_CACHE_DIRECTORY = INDEX_ROOT / "model_cache"
MANIFEST_PATH = INDEX_ROOT / "index_manifest.json"

EMBEDDING_MODEL = str(
    Path(r"D:\pycharm\langchain\models\text2vec-base-chinese")
)
TEXT_SUMMARY_MODEL = os.getenv("TEXT_SUMMARY_MODEL", "deepseek-v4-flash")
# DeepSeek 官方 V4 API 是纯文本模型。扫描图片如需视觉结构化，必须显式配置
# OpenAI-compatible 的 VISION_MODEL / VISION_MODEL_BASE_URL / VISION_API_KEY。
VISION_MODEL = os.getenv("VISION_MODEL", "")
INDEX_WORKERS = max(1, int(os.getenv("PDF_INDEX_WORKERS", "3")))
ENABLE_MODEL_ENRICHMENT = os.getenv(
    "ENABLE_MODEL_ENRICHMENT", "1"
).lower() not in {"0", "false", "no"}

MCP_HOST = "127.0.0.1"
MCP_PORT = 8000
MCP_PATH = "/mcp"

# RAG Fusion parameters are separate from index parameters.
FUSION_QUERY_COUNT = 3
FUSION_RESULTS_PER_QUERY = 8
FUSION_RRF_K = 60
FUSION_TOP_K = 8


def configure_logging() -> logging.Logger:
    """同时输出控制台日志和可轮转的持久日志。"""
    logger = logging.getLogger("matplotlib_mcp_server")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = BASE_DIR / "mcp_server.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


LOGGER = configure_logging()

mcp = FastMCP(
    "Matplotlib Course Knowledge",
    instructions="提供 Matplotlib 中文课程私有 PDF 的多模态检索能力。",
    log_level="INFO",
    streamable_http_path=MCP_PATH,
)


class RawElement(TypedDict):
    type: str
    raw_content: str
    metadata: dict[str, Any]


class IndexedElement(TypedDict):
    type: str
    parent_content: str
    summary: str
    search_text: str
    metadata: dict[str, Any]


def validate_pdf_sources() -> None:
    missing = [str(path) for path in PDF_FILE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下 PDF 不存在：\n" + "\n".join(missing))
    if TESSERACT_CMD.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_CMD)
    else:
        LOGGER.warning("未找到 Tesseract，扫描页和图片将无法执行 OCR：%s", TESSERACT_CMD)
    if not POPPLER_PATH.is_dir():
        LOGGER.warning("未找到 Poppler，扫描页和图片将无法渲染：%s", POPPLER_PATH)


def calculate_source_fingerprint() -> str:
    """PDF 或索引配置变化时自动切换到新的 collection/docstore。"""
    digest = hashlib.sha256()
    digest.update(INDEX_SCHEMA_VERSION.encode("utf-8"))
    digest.update(EMBEDDING_MODEL.encode("utf-8"))
    digest.update(TEXT_SUMMARY_MODEL.encode("utf-8"))
    digest.update(VISION_MODEL.encode("utf-8"))
    digest.update(str(ENABLE_MODEL_ENRICHMENT).encode("ascii"))
    for path in PDF_FILE_PATHS:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def stable_document_id(
    pdf_path: Path,
    page: int,
    element_type: str,
    element_index: int,
    content: str,
) -> str:
    raw = f"{pdf_path.resolve()}|{page}|{element_type}|{element_index}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def get_embedding() -> HuggingFaceEmbeddings:
    LOGGER.info("正在加载嵌入模型：%s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_text_summary_model() -> ChatOpenAI | None:
    if not ENABLE_MODEL_ENRICHMENT:
        LOGGER.info("已禁用远程模型增强，文本和表格使用本地回退。")
        return None
    load_dotenv(ENV_FILE_PATH)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        LOGGER.warning("未配置 DEEPSEEK_API_KEY，文本和表格使用本地摘要回退。")
        return None
    return ChatOpenAI(
        model=TEXT_SUMMARY_MODEL,
        api_key=api_key,
        base_url=os.getenv("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0,
        timeout=90,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_vision_model() -> ChatOpenAI | None:
    if not ENABLE_MODEL_ENRICHMENT:
        LOGGER.info("已禁用远程模型增强，图片使用 OCR 回退。")
        return None
    load_dotenv(ENV_FILE_PATH)
    api_key = os.getenv("VISION_API_KEY")
    base_url = os.getenv("VISION_MODEL_BASE_URL")
    if not api_key or not base_url or not VISION_MODEL:
        LOGGER.warning(
            "未完整配置 VISION_MODEL/VISION_MODEL_BASE_URL/VISION_API_KEY，"
            "图片使用 OCR 文本化回退，扫描表格不会结构化恢复。"
        )
        return None
    return ChatOpenAI(
        model=VISION_MODEL,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=120,
        max_retries=1,
    )


def cache_path(namespace: str, content: bytes) -> Path:
    digest = hashlib.sha256(content).hexdigest()
    directory = MODEL_CACHE_DIRECTORY / namespace
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest}.txt"


def cached_text_generation(namespace: str, prompt: str, fallback: str) -> str:
    path = cache_path(namespace, prompt.encode("utf-8"))
    if path.is_file():
        return path.read_text(encoding="utf-8")

    model = get_text_summary_model()
    result = fallback
    if model is not None:
        try:
            response = model.invoke(prompt)
            if isinstance(response.content, str) and response.content.strip():
                result = response.content.strip()
        except Exception:
            LOGGER.exception("%s 模型处理失败，使用本地回退。", namespace)

    path.write_text(result, encoding="utf-8")
    return result


def summarize_text(content: str) -> str:
    fallback = content[:500].strip()
    prompt = (
        "请为以下 Matplotlib 中文课程文本生成准确、可检索的摘要。"
        "保留函数名、参数、概念和示例用途，不要编造内容。\n\n"
        f"{content[:8000]}"
    )
    return cached_text_generation("text", prompt, fallback)


def textualize_table(markdown_table: str) -> str:
    fallback = re.sub(r"\s+", " ", markdown_table).strip()[:1000]
    prompt = (
        "请把以下 Matplotlib 课程表格转换为适合语义检索的中文说明。"
        "说明表头、各列含义、关键数据关系和技术结论，不要遗漏函数名或数值。\n\n"
        f"{markdown_table[:12000]}"
    )
    return cached_text_generation("table", prompt, fallback)


def ocr_image(image: Image.Image) -> str:
    if not TESSERACT_CMD.is_file():
        return ""
    try:
        return pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
    except Exception:
        LOGGER.exception("图片 OCR 失败。")
        return ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def analyze_image(image_path: Path) -> dict[str, Any]:
    """用视觉模型同时生成图片描述，并抽取扫描图片中的 Markdown 表格。"""
    image_bytes = image_path.read_bytes()
    path = cache_path("image", image_bytes)
    if path.is_file():
        cached = path.read_text(encoding="utf-8")
        parsed = parse_json_object(cached)
        if parsed is not None:
            return parsed
        return {"description": cached, "tables": []}

    with Image.open(io.BytesIO(image_bytes)) as image:
        ocr_text = ocr_image(image)
    fallback = (
        f"图片中的可识别文字：{ocr_text[:2000]}"
        if ocr_text
        else "该图片来自 Matplotlib 课程 PDF，视觉模型未能生成描述。"
    )

    result: dict[str, Any] = {"description": fallback, "tables": []}
    model = get_vision_model()
    if model is not None:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "分析这张 Matplotlib 课程图片，并且只返回 JSON 对象："
                        '{"description":"图片、代码、坐标轴、趋势和教学含义的中文描述",'
                        '"tables":["识别出的 Markdown 表格"]}。'
                        "没有表格时 tables 必须为空列表；不得编造看不清的数据。"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
        )
        try:
            response = model.invoke([message])
            if isinstance(response.content, str) and response.content.strip():
                parsed = parse_json_object(response.content)
                if parsed is None:
                    parsed = {"description": response.content.strip(), "tables": []}
                description = str(parsed.get("description", "")).strip() or fallback
                if ocr_text:
                    description += f"\n\nOCR 文字：{ocr_text[:2000]}"
                tables = parsed.get("tables", [])
                result = {
                    "description": description,
                    "tables": [str(table) for table in tables if str(table).strip()]
                    if isinstance(tables, list)
                    else [],
                }
        except Exception:
            LOGGER.exception("视觉模型处理失败，使用 OCR 回退：%s", image_path)

    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int = 180) -> Image.Image:
    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_index + 1,
        last_page=page_index + 1,
        poppler_path=str(POPPLER_PATH),
    )
    if not images:
        raise RuntimeError(f"PDF 第 {page_index + 1} 页渲染失败：{pdf_path}")
    return images[0]


def table_to_markdown(table: list[list[Any]]) -> str:
    width = max(len(row) for row in table)
    rows = [list(row) + [""] * (width - len(row)) for row in table]

    def clean(value: Any) -> str:
        return str(value or "").replace("\n", " ").replace("|", "\\|").strip()

    header = [clean(cell) for cell in rows[0]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(clean(cell) for cell in row) + " |" for row in rows[1:])
    return "\n".join(lines)


def extract_page_images(
    pdf_path: Path,
    page: Any,
    page_index: int,
    rendered_page: Image.Image,
) -> list[RawElement]:
    elements: list[RawElement] = []
    scale_x = rendered_page.width / float(page.width)
    scale_y = rendered_page.height / float(page.height)
    page_dir = ASSET_DIRECTORY / re.sub(r"[^\w.-]+", "_", pdf_path.stem)
    page_dir.mkdir(parents=True, exist_ok=True)
    seen_boxes: set[tuple[int, int, int, int]] = set()

    for image_index, image_info in enumerate(page.images):
        box = (
            max(0, int(float(image_info.get("x0", 0)) * scale_x)),
            max(0, int(float(image_info.get("top", 0)) * scale_y)),
            min(rendered_page.width, int(float(image_info.get("x1", page.width)) * scale_x)),
            min(rendered_page.height, int(float(image_info.get("bottom", page.height)) * scale_y)),
        )
        if box in seen_boxes or box[2] - box[0] < 80 or box[3] - box[1] < 80:
            continue
        seen_boxes.add(box)
        asset_path = page_dir / f"page_{page_index + 1:03d}_image_{image_index + 1:02d}.png"
        rendered_page.crop(box).save(asset_path, format="PNG")
        elements.append(
            {
                "type": "image",
                "raw_content": str(asset_path),
                "metadata": {
                    "source": str(pdf_path),
                    "page": page_index,
                    "page_label": str(page_index + 1),
                    "type": "image",
                    "asset_path": str(asset_path),
                    "extraction_method": "pdf_image_crop",
                },
            }
        )
    return elements


def parse_pdf_multimodal(pdf_path: Path) -> list[RawElement]:
    """逐页判断文本/扫描类型，并分别抽取文本、表格和图片。"""
    elements: list[RawElement] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )

    with pdfplumber.open(str(pdf_path)) as pdf:
        LOGGER.info("解析 PDF：%s（%d 页）", pdf_path.name, len(pdf.pages))
        for page_index, page in enumerate(pdf.pages):
            extracted_text = (page.extract_text() or "").strip()
            rendered_page: Image.Image | None = None
            extraction_method = "pdf_text"

            if len(extracted_text) < 30:
                extraction_method = "ocr"
                if POPPLER_PATH.is_dir() and TESSERACT_CMD.is_file():
                    rendered_page = render_pdf_page(pdf_path, page_index, dpi=300)
                    extracted_text = ocr_image(rendered_page)
                LOGGER.info(
                    "第 %d 页判定为扫描/低文本页，OCR 字符数=%d",
                    page_index + 1,
                    len(extracted_text),
                )

            for chunk in splitter.split_text(extracted_text):
                if not chunk.strip():
                    continue
                elements.append(
                    {
                        "type": "text",
                        "raw_content": chunk.strip(),
                        "metadata": {
                            "source": str(pdf_path),
                            "page": page_index,
                            "page_label": str(page_index + 1),
                            "type": "text",
                            "extraction_method": extraction_method,
                        },
                    }
                )

            for table in page.extract_tables() or []:
                if not table or len(table) < 2:
                    continue
                markdown = table_to_markdown(table)
                elements.append(
                    {
                        "type": "table",
                        "raw_content": markdown,
                        "metadata": {
                            "source": str(pdf_path),
                            "page": page_index,
                            "page_label": str(page_index + 1),
                            "type": "table",
                            "extraction_method": "pdfplumber_table",
                        },
                    }
                )

            if page.images and POPPLER_PATH.is_dir():
                rendered_page = rendered_page or render_pdf_page(pdf_path, page_index)
                elements.extend(
                    extract_page_images(pdf_path, page, page_index, rendered_page)
                )

    LOGGER.info(
        "PDF 抽取完成：%s，文本=%d，表格=%d，图片=%d",
        pdf_path.name,
        sum(item["type"] == "text" for item in elements),
        sum(item["type"] == "table" for item in elements),
        sum(item["type"] == "image" for item in elements),
    )
    return elements


def enrich_element(element: RawElement) -> list[IndexedElement]:
    element_type = element["type"]
    raw_content = element["raw_content"]
    metadata = dict(element["metadata"])

    if element_type == "text":
        parent_content = raw_content
        search_text = raw_content
        summary = summarize_text(raw_content)
    elif element_type == "table":
        parent_content = raw_content
        search_text = textualize_table(raw_content)
        summary = search_text[:1200]
        metadata["textualized_content"] = search_text
    elif element_type == "image":
        analysis = analyze_image(Path(raw_content))
        search_text = str(analysis.get("description", ""))
        parent_content = search_text
        summary = search_text[:1200]
    else:
        parent_content = raw_content
        search_text = raw_content
        summary = raw_content[:500]

    indexed: list[IndexedElement] = [{
        "type": element_type,
        "parent_content": parent_content,
        "summary": summary,
        "search_text": search_text,
        "metadata": metadata,
    }]

    if element_type == "image":
        for table_index, markdown_table in enumerate(analysis.get("tables", [])):
            table_text = textualize_table(str(markdown_table))
            table_metadata = dict(metadata)
            table_metadata.update(
                {
                    "type": "table",
                    "extraction_method": "vision_table",
                    "table_index": table_index,
                }
            )
            indexed.append(
                {
                    "type": "table",
                    "parent_content": str(markdown_table),
                    "summary": table_text[:1200],
                    "search_text": table_text,
                    "metadata": table_metadata,
                }
            )
    return indexed


def docstore_is_complete(vectorstore: Chroma, retriever: MultiVectorRetriever) -> bool:
    if vectorstore._collection.count() == 0:
        return False
    sample = vectorstore.get(limit=10, include=["metadatas"])
    ids = []
    for metadata in sample.get("metadatas") or []:
        doc_id = (metadata or {}).get("doc_id")
        if doc_id and doc_id not in ids:
            ids.append(doc_id)
    return bool(ids) and all(doc is not None for doc in retriever.docstore.mget(ids))


def write_manifest(fingerprint: str, collection_name: str, stats: dict[str, int]) -> None:
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "collection_name": collection_name,
        "embedding_model": EMBEDDING_MODEL,
        "text_summary_model": TEXT_SUMMARY_MODEL,
        "vision_model": VISION_MODEL,
        "model_enrichment_enabled": ENABLE_MODEL_ENRICHMENT,
        "pdf_files": [
            {
                "path": str(path),
                "name": path.name,
                "size": path.stat().st_size,
                "modified_ns": path.stat().st_mtime_ns,
            }
            for path in PDF_FILE_PATHS
        ],
        "stats": stats,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@lru_cache(maxsize=1)
def get_multi_vector_retriever() -> MultiVectorRetriever:
    validate_pdf_sources()
    fingerprint = calculate_source_fingerprint()
    collection_name = f"matplotlib_course_mv_{fingerprint[:12]}"
    docstore_path = DOCSTORE_DIRECTORY / fingerprint

    CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    docstore_path.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding(),
        persist_directory=str(CHROMA_DIRECTORY),
    )
    byte_store = LocalFileStore(docstore_path)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=byte_store,
        id_key="doc_id",
        search_type=SearchType.mmr,
        search_kwargs={"k": 8, "fetch_k": 32, "lambda_mult": 0.55},
    )

    if docstore_is_complete(vectorstore, retriever):
        LOGGER.info(
            "复用持久 Multi-Vector 索引：collection=%s，向量=%d",
            collection_name,
            vectorstore._collection.count(),
        )
        return retriever

    LOGGER.info("开始构建持久 Multi-Vector 索引：%s", collection_name)
    raw_elements: list[RawElement] = []
    for pdf_path in PDF_FILE_PATHS:
        raw_elements.extend(parse_pdf_multimodal(pdf_path))

    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as executor:
        enriched_groups = list(executor.map(enrich_element, raw_elements))
    indexed_elements = [
        element for group in enriched_groups for element in group
    ]

    parent_pairs: list[tuple[str, Document]] = []
    child_documents: list[Document] = []
    child_ids: list[str] = []
    stats = {"text": 0, "table": 0, "image": 0, "parents": 0, "vectors": 0}

    for element_index, element in enumerate(indexed_elements):
        metadata = dict(element["metadata"])
        source = Path(str(metadata["source"]))
        page = int(metadata.get("page", 0))
        doc_id = stable_document_id(
            source,
            page,
            element["type"],
            element_index,
            element["parent_content"],
        )
        metadata["doc_id"] = doc_id
        metadata["summary"] = element["summary"]
        parent_pairs.append(
            (
                doc_id,
                Document(page_content=element["parent_content"], metadata=metadata),
            )
        )

        representations = [
            ("summary", element["summary"]),
            ("content", element["search_text"]),
        ]
        seen_representations: set[str] = set()
        for representation, content in representations:
            normalized = content.strip()
            if not normalized or normalized in seen_representations:
                continue
            seen_representations.add(normalized)
            child_metadata = {
                "doc_id": doc_id,
                "source": str(source),
                "page": page,
                "type": element["type"],
                "representation": representation,
            }
            child_documents.append(
                Document(page_content=normalized[:8000], metadata=child_metadata)
            )
            child_ids.append(f"{doc_id}:{representation}")

        stats[element["type"]] = stats.get(element["type"], 0) + 1

    retriever.docstore.mset(parent_pairs)
    batch_size = 64
    for start in range(0, len(child_documents), batch_size):
        vectorstore.add_documents(
            child_documents[start : start + batch_size],
            ids=child_ids[start : start + batch_size],
        )

    stats["parents"] = len(parent_pairs)
    stats["vectors"] = len(child_documents)
    write_manifest(fingerprint, collection_name, stats)
    LOGGER.info("Multi-Vector 索引构建完成：%s", stats)
    return retriever


def read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.exception("读取索引清单失败：%s", MANIFEST_PATH)
        return {}


def _normalise_query(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def parse_fusion_queries(raw_content: str, original_query: str) -> list[str]:
    """Parse an LLM response and always return three unique usable queries."""
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    candidates: list[str] = []
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            candidates = [str(item).strip() for item in parsed]
        elif isinstance(parsed, dict):
            values = parsed.get("queries", parsed.get("rewrites", []))
            if isinstance(values, list):
                candidates = [str(item).strip() for item in values]
    except json.JSONDecodeError:
        candidates = [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            for line in cleaned.splitlines()
        ]

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normalise_query(candidate)
        if candidate and key not in seen:
            seen.add(key)
            unique.append(candidate)
        if len(unique) == FUSION_QUERY_COUNT:
            return unique

    # Deterministic fallback keeps retrieval available for malformed/API-free responses.
    fallbacks = [
        original_query.strip(),
        f"{original_query.strip()} Matplotlib 原理和关键概念",
        f"{original_query.strip()} Matplotlib 代码示例、参数和用法",
        f"{original_query.strip()} Matplotlib 常见问题和实践场景",
    ]
    for candidate in fallbacks:
        key = _normalise_query(candidate)
        if candidate and key not in seen:
            seen.add(key)
            unique.append(candidate)
        if len(unique) == FUSION_QUERY_COUNT:
            break
    return unique


@lru_cache(maxsize=256)
def generate_fusion_queries(original_query: str) -> tuple[str, ...]:
    """Ask the configured LLM for three semantic rewrites."""
    query = original_query.strip()
    if not query:
        return ("Matplotlib", "Matplotlib 教程基础", "Matplotlib 绘图用法")
    model = get_text_summary_model()
    if model is None:
        LOGGER.warning("RAG Fusion query generation unavailable; using deterministic rewrites")
        return tuple(parse_fusion_queries("[]", query))
    prompt = (
        "Generate exactly 3 different Chinese search queries for the user's question. "
        "Preserve Matplotlib entities, functions, and constraints; vary concept, "
        "implementation, and practical-example angles. Return JSON array only.\n"
        f"User question: {query}"
    )
    try:
        response = model.invoke(prompt)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        queries = parse_fusion_queries(raw, query)
        LOGGER.info("RAG Fusion generated %d queries for %r: %s", len(queries), query, queries)
        return tuple(queries)
    except Exception:
        LOGGER.exception("RAG Fusion query generation failed; using deterministic rewrites")
        return tuple(parse_fusion_queries("[]", query))


def reciprocal_rank_fusion(
    query_results: list[tuple[str, list[Document]]],
    rrf_k: int = FUSION_RRF_K,
    top_k: int = FUSION_TOP_K,
) -> list[tuple[Document, float, list[str], dict[str, int]]]:
    """Merge ranked lists by stable doc_id using score=sum(1/(k+rank))."""
    merged: dict[str, dict[str, Any]] = {}
    for query, documents in query_results:
        for rank, document in enumerate(documents, start=1):
            metadata = dict(document.metadata)
            key = str(metadata.get("doc_id") or hashlib.sha256(document.page_content.encode("utf-8")).hexdigest())
            entry = merged.setdefault(key, {"document": document, "score": 0.0, "queries": [], "ranks": {}})
            entry["score"] += 1.0 / (rrf_k + rank)
            entry["queries"].append(query)
            entry["ranks"][query] = rank
    ordered = sorted(merged.values(), key=lambda item: (-item["score"], min(item["ranks"].values())))
    return [
        (item["document"], item["score"], item["queries"], item["ranks"])
        for item in ordered[:top_k]
    ]


def rag_fusion_search(query: str) -> tuple[list[str], list[tuple[Document, float, list[str], dict[str, int]]]]:
    """Generate three queries, retrieve them concurrently, then apply RRF."""
    queries = list(generate_fusion_queries(query))[:FUSION_QUERY_COUNT]
    retriever = get_multi_vector_retriever()
    started = time.perf_counter()

    def retrieve(one_query: str) -> tuple[str, list[Document]]:
        documents = retriever.invoke(one_query)[:FUSION_RESULTS_PER_QUERY]
        LOGGER.info("RAG Fusion retrieval query=%r hits=%d", one_query, len(documents))
        return one_query, documents

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        query_results = list(executor.map(retrieve, queries))
    fused = reciprocal_rank_fusion(query_results)
    LOGGER.info(
        "RAG Fusion complete original=%r queries=%d fused=%d elapsed_ms=%.1f",
        query, len(queries), len(fused), (time.perf_counter() - started) * 1000,
    )
    return queries, fused


@mcp.tool()
def list_matplotlib_course_documents() -> str:
    """列出私有 Matplotlib 课程 PDF、索引版本和各模态内容数量。"""
    get_multi_vector_retriever()
    return json.dumps(read_manifest(), ensure_ascii=False)


@mcp.tool()
def search_matplotlib_knowledge(query: str) -> str:
    """检索私有 Matplotlib PDF，返回相关文本、表格或图片的文字化内容及来源。"""
    LOGGER.info("收到知识库检索请求：%s", query)
    queries, fused_documents = rag_fusion_search(query)
    result = []
    for rank, (document, score, matched_queries, query_ranks) in enumerate(fused_documents, start=1):
        metadata = dict(document.metadata)
        metadata["fusion_score"] = score
        metadata["fusion_rank"] = rank
        metadata["fusion_query_ranks"] = query_ranks
        result.append(
            {
                "type": metadata.get("type", "text"),
                "content": document.page_content,
                "summary": metadata.get("summary", ""),
                "metadata": metadata,
                "fusion": {
                    "score": score,
                    "rank": rank,
                    "matched_queries": matched_queries,
                    "query_ranks": query_ranks,
                },
            }
        )
    LOGGER.info("检索完成：query=%r，结果=%d", query, len(result))
    LOGGER.info("RAG Fusion response original=%r queries=%s results=%d", query, queries, len(result))
    return json.dumps({"queries": queries, "results": result}, ensure_ascii=False)


if __name__ == "__main__":
    LOGGER.info("正在预热私有 PDF Multi-Vector 索引...")
    try:
        get_multi_vector_retriever()
        LOGGER.info("MCP 服务启动：http://%s:%d%s", MCP_HOST, MCP_PORT, MCP_PATH)
        os.environ["MCP_HOST"] = MCP_HOST
        os.environ["MCP_PORT"] = str(MCP_PORT)
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        LOGGER.info("收到停止信号，MCP 服务已正常关闭。")
    except Exception:
        LOGGER.exception("MCP 服务启动或运行失败。")
        raise


