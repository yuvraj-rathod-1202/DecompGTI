from decompgti.mcp_server import graph_tools


def test_bfs_order_from_source() -> None:
    edges = [[1, 2], [1, 3], [2, 4], [3, 5]]
    result = graph_tools.bfs(edges=edges, source=1)

    assert result["order"][0] == 1
    assert set(result["order"]) == {1, 2, 3, 4, 5}


def test_dijkstra_shortest_path_distance() -> None:
    edges = [["A", "B", 2], ["B", "C", 3], ["A", "C", 10]]
    result = graph_tools.dijkstra_shortest_path(edges=edges, source="A", target="C")

    assert result["path"] == ["A", "B", "C"]
    assert result["distance"] == 5.0


def test_maximum_flow_value() -> None:
    edges = [["S", "A", 3], ["S", "B", 2], ["A", "T", 2], ["B", "T", 3], ["A", "B", 1]]
    result = graph_tools.maximum_flow(edges=edges, source="S", target="T")

    assert result["max_flow"] == 5.0


def test_topological_sort_valid_order() -> None:
    edges = [["A", "B"], ["A", "C"], ["B", "D"], ["C", "D"]]
    result = graph_tools.topological_sort(edges=edges)
    order = result["order"]

    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


def test_cycle_detection_directed_true() -> None:
    edges = [[1, 2], [2, 3], [3, 1]]
    result = graph_tools.cycle_detection(edges=edges, directed=True)

    assert result["has_cycle"] is True


def test_minimum_spanning_tree_total_weight() -> None:
    edges = [[1, 2, 1], [2, 3, 2], [1, 3, 5], [3, 4, 1], [2, 4, 4]]
    result = graph_tools.minimum_spanning_tree(edges=edges, algorithm="kruskal")

    assert result["total_weight"] == 4.0
    assert len(result["edges"]) == 3
