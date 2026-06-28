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


def test_roundtrip_routing_to_connectivity() -> None:
    model_output = (
        '{"tool_name":"connectivity","arguments":{"edges":'
        '[[1,2],[2,3],[3,4]],"source":1,"target":4,"directed":false}}'
    )

    execution = execute_from_model_output(model_output)

    assert execution["tool_name"] == "connectivity"
    assert execution["result"]["connected"] is True
    assert execution["result"]["path"] == [1, 2, 3, 4]


def test_roundtrip_routing_to_connected_component() -> None:
    model_output = (
        '{"tool_name":"connected_component","arguments":{"edges":'
        '[[1,2],[2,3],[4,5]],"source":1,"directed":false}}'
    )

    execution = execute_from_model_output(model_output)

    assert execution["tool_name"] == "connected_component"
    assert execution["result"]["component"] == [1, 2, 3]
