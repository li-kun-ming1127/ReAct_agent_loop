import math

from langchain_core.documents import Document

import mcp_server


def doc(doc_id: str) -> Document:
    return Document(page_content=doc_id, metadata={"doc_id": doc_id})


queries = mcp_server.parse_fusion_queries(
    '["subplot 如何布局", "子图网格参数", "多坐标轴排版示例"]',
    "subplot 怎么用",
)
assert len(queries) == 3
assert len(set(queries)) == 3

fallbacks = mcp_server.parse_fusion_queries("not-json", "subplot 怎么用")
assert len(fallbacks) == 3
assert len(set(fallbacks)) == 3

fused = mcp_server.reciprocal_rank_fusion(
    [
        ("q1", [doc("shared"), doc("only-q1")]),
        ("q2", [doc("only-q2"), doc("shared")]),
        ("q3", [doc("shared")]),
    ],
    rrf_k=60,
    top_k=3,
)
assert fused[0][0].metadata["doc_id"] == "shared"
assert math.isclose(fused[0][1], 2 / 61 + 1 / 62)
assert fused[0][2] == ["q1", "q2", "q3"]
assert fused[0][3] == {"q1": 1, "q2": 2, "q3": 1}

print("RAG Fusion parser and RRF tests passed")
