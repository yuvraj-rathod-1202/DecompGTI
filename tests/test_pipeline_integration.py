from decompgti.mcp_server.pipeline import execute_from_model_output


def test_roundtrip_routing_to_dijkstra() -> None:
    model_output = (
        '{"tool_name":"dijkstra_shortest_path","arguments":{"edges":'
        '[["A","B",2],["B","C",3],["A","C",10]],"source":"A","target":"C"}}'
    )

    execution = execute_from_model_output(model_output)

    assert execution["tool_name"] == "dijkstra_shortest_path"
    assert execution["result"]["path"] == ["A", "B", "C"]
    assert execution["result"]["distance"] == 5.0


def test_roundtrip_routing_to_cycle_detection() -> None:
    model_output = (
        '{"tool_name":"cycle_detection","arguments":{"edges":'
        '[[1,2],[2,3],[3,1]],"directed":true}}'
    )

    execution = execute_from_model_output(model_output)

    assert execution["tool_name"] == "cycle_detection"
    assert execution["result"]["has_cycle"] is True
