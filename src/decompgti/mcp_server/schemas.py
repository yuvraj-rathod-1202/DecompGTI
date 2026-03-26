from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GraphInput(BaseModel):
    """Graph encoded as edge list with optional weights and direction."""

    edges: list[list[Any]] = Field(
        description="Each edge is [u, v] or [u, v, weight].",
        min_length=1,
    )
    directed: bool = Field(default=False)


class PathQuery(GraphInput):
    source: Any
    target: Any


class TraversalQuery(GraphInput):
    source: Any


class FlowQuery(GraphInput):
    source: Any
    target: Any


class TopologicalSortQuery(GraphInput):
    directed: bool = Field(default=True)


class MatchingQuery(GraphInput):
    left_nodes: list[Any] = Field(default_factory=list)


class MstQuery(GraphInput):
    algorithm: Literal["kruskal", "prim"] = "kruskal"


class CycleQuery(GraphInput):
    directed: bool = Field(default=False)


class ToolCall(BaseModel):
    """Normalized tool call extracted from model output."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
