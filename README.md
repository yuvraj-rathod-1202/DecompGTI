# DecompGTI

Graph-tool reasoning project with an MCP server for deterministic execution.

## Setup

1. Sync dependencies:

```bash
uv sync
```

2. Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Run MCP Server

```bash
uv run decompgti-mcp-server
```

## MCP Tools Implemented

- `bfs`
- `dfs`
- `dijkstra_shortest_path`
- `maximum_flow`
- `bipartite_maximum_matching`
- `topological_sort`
- `cycle_detection`
- `minimum_spanning_tree`
- `extract_tool_name_from_model_output`
- `extract_tool_call_from_model_output`

## Model Routing Helpers

Routing utilities are in `src/decompgti/mcp_server/routing.py`:

- `extract_tool_name(model_output)`
- `extract_tool_call(model_output)`
- `build_tool_call_from_model_output(model_output)`

Tool metadata for prompting is in `src/decompgti/mcp_server/tool_catalog.py` via `get_model_tool_prompt_block()`.

## End-to-End Demo

Run a local roundtrip demo (model output -> parse -> execute tool):

```bash
uv run python scripts/demo_roundtrip.py
```

Execution helper used in tests and demo:

- `src/decompgti/mcp_server/pipeline.py` (`execute_from_model_output`)