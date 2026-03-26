import pytest

from decompgti.mcp_server.routing import (
    ToolRoutingError,
    build_tool_call_from_model_output,
    extract_tool_call,
    extract_tool_name,
)


def test_extract_tool_name_and_args_from_json_output() -> None:
    output = '{"tool_name":"dijkstra_shortest_path","arguments":{"edges":[["A","B",2]],"source":"A","target":"B"}}'
    tool_name = extract_tool_name(output)
    call = extract_tool_call(output)

    assert tool_name == "dijkstra_shortest_path"
    assert call.arguments["source"] == "A"


def test_extract_tool_name_from_fenced_json() -> None:
    output = """The best tool is below:\n```json\n{\"tool_name\":\"bfs\",\"arguments\":{\"edges\":[[1,2]],\"source\":1}}\n```"""
    assert extract_tool_name(output) == "bfs"


def test_build_tool_call_from_model_output() -> None:
    output = '{"tool_name":"maximum_flow","arguments":{"edges":[["S","A",3],["A","T",2]],"source":"S","target":"T"}}'
    tool_name, arguments = build_tool_call_from_model_output(output)

    assert tool_name == "maximum_flow"
    assert arguments["target"] == "T"


def test_extract_tool_call_raises_for_unsupported_tool() -> None:
    output = '{"tool_name":"pagerank","arguments":{"edges":[[1,2]]}}'

    with pytest.raises(ToolRoutingError, match="Unsupported tool"):
        extract_tool_call(output)


def test_extract_tool_call_raises_for_non_json_output() -> None:
    output = "Use bfs with source A and edges A-B"

    with pytest.raises(ToolRoutingError, match="No JSON object"):
        extract_tool_call(output)
