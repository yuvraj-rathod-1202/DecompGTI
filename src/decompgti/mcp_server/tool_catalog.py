from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="bfs",
        description="Breadth-first traversal from a source node.",
        input_schema={"type": "object", "properties": {"edges": {}, "source": {}}, "required": ["edges", "source"]},
    ),
    ToolSpec(
        name="dfs",
        description="Depth-first traversal from a source node.",
        input_schema={"type": "object", "properties": {"edges": {}, "source": {}}, "required": ["edges", "source"]},
    ),
    ToolSpec(
        name="dijkstra_shortest_path",
        description="Shortest path with non-negative edge weights.",
        input_schema={"type": "object", "properties": {"edges": {}, "source": {}, "target": {}}, "required": ["edges", "source", "target"]},
    ),
    ToolSpec(
        name="maximum_flow",
        description="Maximum flow between source and target using a directed network.",
        input_schema={"type": "object", "properties": {"edges": {}, "source": {}, "target": {}}, "required": ["edges", "source", "target"]},
    ),
    ToolSpec(
        name="bipartite_maximum_matching",
        description="Maximum matching in a bipartite graph.",
        input_schema={"type": "object", "properties": {"edges": {}, "left_nodes": {}}, "required": ["edges"]},
    ),
    ToolSpec(
        name="topological_sort",
        description="Topological ordering for directed acyclic graph.",
        input_schema={"type": "object", "properties": {"edges": {}, "directed": {}}, "required": ["edges"]},
    ),
    ToolSpec(
        name="cycle_detection",
        description="Detect whether a graph contains a cycle.",
        input_schema={"type": "object", "properties": {"edges": {}, "directed": {}}, "required": ["edges"]},
    ),
    ToolSpec(
        name="minimum_spanning_tree",
        description="Minimum spanning tree with Prim or Kruskal.",
        input_schema={"type": "object", "properties": {"edges": {}, "algorithm": {}}, "required": ["edges"]},
    ),
]


def get_model_tool_prompt_block() -> str:
    """Return a compact prompt block describing tools and expected output format."""

    tools_text = "\n".join(f"- {spec.name}: {spec.description}" for spec in TOOL_SPECS)
    return (
        "Available tools:\n"
        f"{tools_text}\n\n"
        "Output EXACT JSON only in this shape:\n"
        '{"tool_name": "<name>", "arguments": { ... }}'
    )
