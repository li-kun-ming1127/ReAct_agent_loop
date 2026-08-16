import ast
import logging
import time
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


source_path = Path(__file__).with_name("ReAct_agent_loop_with_toolnode.py")
tree = ast.parse(source_path.read_text(encoding="utf-8"))
class_node = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "AgentMonitorHandler"
)
module = ast.Module(body=[class_node], type_ignores=[])
ast.fix_missing_locations(module)
namespace = {
    "Any": Any,
    "BaseCallbackHandler": BaseCallbackHandler,
    "Lock": Lock,
    "logging": logging,
    "time": time,
}
exec(compile(module, str(source_path), "exec"), namespace)
handler = namespace["AgentMonitorHandler"]()

handler.on_llm_end(
    SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        },
        generations=[],
    )
)

message = SimpleNamespace(
    usage_metadata={
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    },
    response_metadata={},
)
handler.on_llm_end(
    SimpleNamespace(
        llm_output=None,
        generations=[[SimpleNamespace(message=message)]],
    )
)

message = SimpleNamespace(
    usage_metadata=None,
    response_metadata={
        "token_usage": {
            "prompt_tokens": "4",
            "completion_tokens": "2",
        }
    },
)
handler.on_llm_end(
    SimpleNamespace(
        llm_output={},
        generations=[[SimpleNamespace(message=message)]],
    )
)

assert handler.token_usage() == {
    "prompt_tokens": 21,
    "completion_tokens": 10,
    "total_tokens": 31,
    "llm_calls": 3,
}
print("AgentMonitorHandler token aggregation tests passed")
