from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import graph_tools
from .routing import extract_tool_call

ToolFn = Callable[..., dict[str, Any]]

_TOOL_REGISTRY: dict[str, ToolFn] = {
    "bfs": graph_tools.bfs,
    "dfs": graph_tools.dfs,
    "dijkstra_shortest_path": graph_tools.dijkstra_shortest_path,
    "maximum_flow": graph_tools.maximum_flow,
    "bipartite_maximum_matching": graph_tools.bipartite_maximum_matching,
    "topological_sort": graph_tools.topological_sort,
    "cycle_detection": graph_tools.cycle_detection,
    "minimum_spanning_tree": graph_tools.minimum_spanning_tree,
}


def execute_from_model_output(model_output: str) -> dict[str, Any]:
    """Execute a graph tool directly from model JSON output.

    Expected JSON shape in model output:
    {"tool_name": "...", "arguments": {...}}
    """

    tool_call = extract_tool_call(model_output)
    tool_fn = _TOOL_REGISTRY[tool_call.tool_name]
    result = tool_fn(**tool_call.arguments)

    return {
        "tool_name": tool_call.tool_name,
        "arguments": tool_call.arguments,
        "result": result,
    }
