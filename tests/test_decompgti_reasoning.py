from decompgti.data.decompgti_reasoning import build_decompgti_reasoning_records, build_graph_json, build_tool_arguments, infer_tool_name


def test_build_decompgti_reasoning_records_for_shortest_path() -> None:
    sample = {
        "task": "shortest_path",
        "graph": "[(<0>, <1>, weight:2), (<1>, <2>, weight:3)]",
        "graph_nl": "Node <0> is connected to node <1> (weight: 2).\nNode <1> is connected to node <2> (weight: 3).",
        "nodes": "[<0>, <1>, <2>]",
        "directed": False,
        "question": "Calculate the distance of the shortest path from node <0> to node <2>.",
        "id": 7,
    }

    graph_json = build_graph_json(sample)
    tool_name = infer_tool_name(sample)
    arguments = build_tool_arguments(sample)
    records = build_decompgti_reasoning_records(sample)

    assert graph_json["directed"] is False
    assert graph_json["adjacency"]["0"]
    assert tool_name == "dijkstra_shortest_path"
    assert arguments["source"] == 0
    assert arguments["target"] == 2
    assert len(records) == 3
    assert records[2]["output"].startswith('{"arguments":') or records[2]["output"].startswith('{"tool_name":')