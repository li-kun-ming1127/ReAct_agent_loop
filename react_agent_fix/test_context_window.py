import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately


source_path = Path(__file__).with_name("ReAct_agent_loop_with_toolnode.py")
tree = ast.parse(source_path.read_text(encoding="utf-8"))
wanted = {
    "truncate_text",
    "estimate_message_tokens",
    "split_into_turns",
    "flatten_turns",
    "truncate_oversized_tool_messages",
    "build_model_context",
}
nodes = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in wanted
]
module = ast.Module(body=nodes, type_ignores=[])
ast.fix_missing_locations(module)
namespace = {
    "Any": Any,
    "AIMessage": AIMessage,
    "HumanMessage": HumanMessage,
    "SystemMessage": SystemMessage,
    "ToolMessage": ToolMessage,
    "count_tokens_approximately": count_tokens_approximately,
    "TOKEN_ESTIMATE_CHARS_PER_TOKEN": 3.0,
    "MAX_TOOL_MESSAGE_TOKENS": 120,
    "CONTEXT_INPUT_BUDGET_TOKENS": 420,
    "MIN_RECENT_TURNS": 2,
    "tools": [],
    "get_cached_summary": lambda _session_id: ("existing summary", 0),
    "summarize_messages": lambda _messages, existing: f"{existing}; merged old turns",
    "logger": SimpleNamespace(info=lambda *_args, **_kwargs: None),
}
exec(compile(module, str(source_path), "exec"), namespace)

split_into_turns = namespace["split_into_turns"]
flatten_turns = namespace["flatten_turns"]
estimate_message_tokens = namespace["estimate_message_tokens"]
truncate_oversized_tool_messages = namespace["truncate_oversized_tool_messages"]
build_model_context = namespace["build_model_context"]

messages = [
    SystemMessage(content="system"),
    HumanMessage(content="question 1"),
    AIMessage(content="answer 1", tool_calls=[]),
    HumanMessage(content="question 2"),
    AIMessage(
        content="call tool",
        tool_calls=[
            {"name": "search", "args": {}, "id": "1", "type": "tool_call"}
        ],
    ),
    ToolMessage(content="tool result", tool_call_id="1", name="search"),
    AIMessage(content="answer 2"),
]
turns = split_into_turns(messages)
assert len(turns) == 2
assert len(flatten_turns(turns)) == 6
assert isinstance(turns[1][2], ToolMessage)

large_tool = ToolMessage(
    content="x" * 10000,
    tool_call_id="keep-id",
    name="search",
)
bounded = truncate_oversized_tool_messages(
    [[HumanMessage(content="q"), large_tool]]
)
assert bounded[0][1].tool_call_id == "keep-id"
assert len(bounded[0][1].content) < len(large_tool.content)
assert estimate_message_tokens(bounded[0]) > 0

long_history = []
for turn_number in range(1, 5):
    long_history.extend(
        [
            HumanMessage(content=f"question {turn_number} " + "q" * 360),
            AIMessage(content=f"answer {turn_number} " + "a" * 360),
        ]
    )

context = build_model_context("primary system", long_history, "session-1")
assert isinstance(context[0], SystemMessage)
assert context[0].content == "primary system"
assert isinstance(context[1], SystemMessage)
assert "merged old turns" in context[1].content
assert not any(
    isinstance(message, HumanMessage) and message.content.startswith("question 1")
    for message in context
)
assert any(
    isinstance(message, HumanMessage) and message.content.startswith("question 4")
    for message in context
)
assert any(
    isinstance(message, AIMessage) and message.content.startswith("answer 4")
    for message in context
)

print("context window helper tests passed")
