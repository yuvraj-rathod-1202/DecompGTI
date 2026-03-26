from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import graph_tools
from .routing import extract_tool_call, extract_tool_name


def build_server() -> FastMCP:
    mcp = FastMCP(name="DecompGTI-GraphTools")

    @mcp.tool()
    def bfs(edges: list[list], source: str | int, directed: bool = False) -> dict:
        return graph_tools.bfs(edges=edges, source=source, directed=directed)

    @mcp.tool()
    def dfs(edges: list[list], source: str | int, directed: bool = False) -> dict:
        return graph_tools.dfs(edges=edges, source=source, directed=directed)

    @mcp.tool()
    def dijkstra_shortest_path(
        edges: list[list], source: str | int, target: str | int, directed: bool = False
    ) -> dict:
        return graph_tools.dijkstra_shortest_path(
            edges=edges,
            source=source,
            target=target,
            directed=directed,
        )

    @mcp.tool()
    def maximum_flow(edges: list[list], source: str | int, target: str | int) -> dict:
        return graph_tools.maximum_flow(edges=edges, source=source, target=target)

    @mcp.tool()
    def bipartite_maximum_matching(edges: list[list], left_nodes: list | None = None) -> dict:
        return graph_tools.bipartite_maximum_matching(edges=edges, left_nodes=left_nodes)

    @mcp.tool()
    def topological_sort(edges: list[list]) -> dict:
        return graph_tools.topological_sort(edges=edges)

    @mcp.tool()
    def cycle_detection(edges: list[list], directed: bool = False) -> dict:
        return graph_tools.cycle_detection(edges=edges, directed=directed)

    @mcp.tool()
    def minimum_spanning_tree(edges: list[list], algorithm: str = "kruskal") -> dict:
        return graph_tools.minimum_spanning_tree(edges=edges, algorithm=algorithm)

    @mcp.tool()
    def extract_tool_name_from_model_output(model_output: str) -> dict:
        return {"tool_name": extract_tool_name(model_output)}

    @mcp.tool()
    def extract_tool_call_from_model_output(model_output: str) -> dict:
        parsed = extract_tool_call(model_output)
        return parsed.model_dump()

    return mcp


def main() -> None:
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
