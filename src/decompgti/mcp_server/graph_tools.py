from __future__ import annotations

from typing import Any

import networkx as nx


def _build_graph(edges: list[list[Any]], directed: bool) -> nx.Graph | nx.DiGraph:
    graph: nx.Graph | nx.DiGraph = nx.DiGraph() if directed else nx.Graph()
    for edge in edges:
        if len(edge) == 2:
            u, v = edge
            graph.add_edge(u, v, weight=1.0)
        elif len(edge) == 3:
            u, v, w = edge
            graph.add_edge(u, v, weight=float(w), capacity=float(w))
        else:
            raise ValueError(f"Edge must have length 2 or 3, got {edge}")
    return graph


def bfs(edges: list[list[Any]], source: Any, directed: bool = False) -> dict[str, Any]:
    graph = _build_graph(edges, directed=directed)
    order = list(nx.bfs_tree(graph, source, sort_neighbors=lambda nodes: sorted(list(nodes), key=lambda x: int(x))).nodes())
    return {"order": order}


def dfs(edges: list[list[Any]], source: Any, directed: bool = False) -> dict[str, Any]:
    graph = _build_graph(edges, directed=directed)
    order = list(nx.dfs_preorder_nodes(graph, source=source, sort_neighbors=lambda nodes: sorted(list(nodes), key=lambda x: int(x))))
    return {"order": order}


def check_connectivity(
    edges: list[list[Any]],
    source: Any = None,
    target: Any = None,
    directed: bool = False,
) -> dict[str, Any]:
    graph = _build_graph(edges, directed=directed)
    if source is not None and target is not None:
        connected = nx.has_path(graph, source, target)
    else:
        connected = nx.is_connected(graph) if not directed else nx.is_weakly_connected(graph)
    return {"connected": connected}


def dijkstra_shortest_path(
    edges: list[list[Any]],
    source: Any,
    target: Any,
    directed: bool = False,
) -> dict[str, Any]:
    graph = _build_graph(edges, directed=directed)
    path = nx.dijkstra_path(graph, source=source, target=target, weight="weight")
    distance = nx.dijkstra_path_length(graph, source=source, target=target, weight="weight")
    return {"path": path, "distance": distance}


def maximum_flow(edges: list[list[Any]], source: Any, target: Any) -> dict[str, Any]:
    graph = _build_graph(edges, directed=True)
    value, flow_dict = nx.maximum_flow(graph, source, target, capacity="capacity")
    return {"max_flow": value, "flow": flow_dict}


def bipartite_maximum_matching(
    edges: list[list[Any]],
    left_nodes: list[Any] | None = None,
) -> dict[str, Any]:
    graph = _build_graph(edges, directed=False)
    if left_nodes:
        matching = nx.bipartite.maximum_matching(graph, top_nodes=set(left_nodes))
    else:
        left_part, _ = nx.bipartite.sets(graph)
        matching = nx.bipartite.maximum_matching(graph, top_nodes=left_part)

    unique_pairs = []
    used = set()
    for u, v in matching.items():
        key = tuple(sorted((u, v), key=str))
        if key not in used:
            used.add(key)
            unique_pairs.append([u, v])
    return {"matching_size": len(unique_pairs), "matching": unique_pairs}


def topological_sort(edges: list[list[Any]]) -> dict[str, Any]:
    graph = _build_graph(edges, directed=True)
    order = list(nx.topological_sort(graph))
    return {"order": order}


def cycle_detection(edges: list[list[Any]], directed: bool = False) -> dict[str, Any]:
    graph = _build_graph(edges, directed=directed)
    has_cycle = not nx.is_directed_acyclic_graph(graph) if directed else len(nx.cycle_basis(graph)) > 0
    return {"has_cycle": has_cycle}


def minimum_spanning_tree(
    edges: list[list[Any]],
    algorithm: str = "kruskal",
) -> dict[str, Any]:
    graph = _build_graph(edges, directed=False)
    if algorithm not in {"kruskal", "prim"}:
        raise ValueError("algorithm must be one of: kruskal, prim")

    tree = nx.minimum_spanning_tree(graph, algorithm=algorithm, weight="weight")
    tree_edges = [[u, v, d.get("weight", 1.0)] for u, v, d in tree.edges(data=True)]
    total_weight = sum(d.get("weight", 1.0) for _, _, d in tree.edges(data=True))
    return {"edges": tree_edges, "total_weight": total_weight}
