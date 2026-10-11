import ast
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

SOURCE = Path(__file__).parents[1] / "ReAct_agent_loop_with_toolnode.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
NAMES = {"ToolRouter", "safe_parse_tool_result", "structured_tool_history", "build_agent_system_prompt", "_tool_cache_key", "selected_tool_node"}
selected = [node for node in TREE.body if getattr(node, "name", None) in NAMES]
module = ast.Module(body=selected, type_ignores=[])
ast.fix_missing_locations(module)
namespace = {
    "Any": dict,
    "Document": SimpleNamespace,
    "Chroma": object,
    "HuggingFaceEmbeddings": object,
    "logger": SimpleNamespace(exception=lambda *args, **kwargs: None),
    "json": json,
    "hashlib": hashlib,
    "REDIS_KEY_PREFIX": "test-rag",
    "AIMessage": AIMessage,
    "ToolMessage": ToolMessage,
    "AgentState": dict,
    "RunnableConfig": dict,
    "str": str,
}
exec(compile(module, str(SOURCE), "exec"), namespace)

ToolRouter = namespace["ToolRouter"]
router = ToolRouter.__new__(ToolRouter)
router.tools = {"search_matplotlib_knowledge": object(), "list_matplotlib_course_documents": object()}
router._all_names = tuple(sorted(router.tools))
router.vectorstore = SimpleNamespace(similarity_search=lambda query, k: [SimpleNamespace(metadata={"tool_name": "list_matplotlib_course_documents"})])
assert router.select_tool_names("课程有哪些章节") == ["list_matplotlib_course_documents", "search_matplotlib_knowledge"]
assert len(router.resolve(["missing"])) == 2
assert router.resolve(["search_matplotlib_knowledge"]) == [router.tools["search_matplotlib_knowledge"]]

history = [
    HumanMessage(content="subplot 怎么布局"),
    AIMessage(content="", tool_calls=[{"name": "search_matplotlib_knowledge", "args": {"query": "subplot 怎么布局"}, "id": "call-1"}]),
    ToolMessage(content=json.dumps({"results": [{"content": "subplot 文档"}]}), tool_call_id="call-1", name="search_matplotlib_knowledge"),
]
summary = json.loads(namespace["structured_tool_history"](history, "subplot 怎么布局", True))
assert summary["same_as_previous"] is True
assert summary["cache_available"] is True
assert summary["recommended_action"] == "reuse_cached_result_if_search_is_needed"
assert summary["recent_calls"] == [{"tool": "search_matplotlib_knowledge", "query": "subplot 怎么布局", "result_count": 1}]
prompt = namespace["build_agent_system_prompt"]("beginner", "示例", json.dumps(summary, ensure_ascii=False), False)
assert "call-1" not in prompt
assert "subplot 怎么布局" in prompt
assert "RAG Fusion" in prompt

key = namespace["_tool_cache_key"]("session-a", "search_matplotlib_knowledge", " Subplot 怎么布局 ")
assert key == namespace["_tool_cache_key"]("session-a", "search_matplotlib_knowledge", "subplot 怎么布局")
assert key.startswith("test-rag:tool-cache:session-a:")

class FakeRouter:
    def __init__(self):
        self.resolved = None

    def resolve(self, names):
        self.resolved = list(names)
        return [f"tool:{name}" for name in names]


class FakeToolNode:
    calls = 0
    selected = None

    def __init__(self, selected):
        FakeToolNode.selected = selected

    async def ainvoke(self, state, config):
        FakeToolNode.calls += 1
        call = state["messages"][-1].tool_calls[0]
        return {"messages": [ToolMessage(content="fresh", tool_call_id=call["id"], name=call["name"])]}


router_for_node = FakeRouter()
namespace.update(
    {
        "tool_router": router_for_node,
        "ToolNode": FakeToolNode,
        "get_cached_tool_result": lambda session, tool, query: "cached" if query == "same" else None,
        "set_cached_tool_result": lambda *args: None,
        "serialize_tool_content": str,
    }
)
selected_tool_node = namespace["selected_tool_node"]
base_state = {
    "messages": [
        HumanMessage(content="same"),
        AIMessage(content="", tool_calls=[{"name": "search_matplotlib_knowledge", "args": {"query": "same"}, "id": "cache-call"}]),
    ],
    "selected_tool_names": ["search_matplotlib_knowledge"],
}
cached_output = asyncio.run(selected_tool_node(base_state, {"configurable": {"session_id": "s1"}}))
assert cached_output["messages"][0].content == "cached"
assert FakeToolNode.calls == 0
assert router_for_node.resolved == ["search_matplotlib_knowledge"]

miss_state = {
    "messages": [
        HumanMessage(content="new"),
        AIMessage(content="", tool_calls=[{"name": "search_matplotlib_knowledge", "args": {"query": "new"}, "id": "remote-call"}]),
    ],
    "selected_tool_names": ["search_matplotlib_knowledge"],
}
asyncio.run(selected_tool_node(miss_state, {"configurable": {"session_id": "s1"}}))
assert FakeToolNode.calls == 1
assert FakeToolNode.selected == ["tool:search_matplotlib_knowledge"]
print("Agent extension tests passed")