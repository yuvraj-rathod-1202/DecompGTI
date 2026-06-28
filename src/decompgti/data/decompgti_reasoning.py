from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decompgti.mcp_server.tool_catalog import TOOL_SPECS, get_model_tool_prompt_block


DECOMPGTI_TASKS: tuple[str, ...] = (
    "shortest_path",
    "BFS",
    "DFS",
    "connectivity",
    "cycle",
    "topological_sort",
    "bipartite",
    "MST",
    "maximum_flow",
    "connected_component",
)

TASK_TO_TOOL: dict[str, str] = {
    "shortest_path": "dijkstra_shortest_path",
    "BFS": "bfs",
    "DFS": "dfs",
    "connectivity": "connectivity",
    "cycle": "cycle_detection",
    "topological_sort": "topological_sort",
    "bipartite": "bipartite_maximum_matching",
    "MST": "minimum_spanning_tree",
    "maximum_flow": "maximum_flow",
    "connected_component": "connected_component",
}


@dataclass(frozen=True)
class StageRecord:
    instruction: str
    input: str
    output: str


def _parse_node(token: str) -> int | str:
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        token = token[1:-1].strip()
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    return token


def _parse_weight(token: str) -> int | float | str:
    token = token.strip()
    if token.startswith("weight:"):
        token = token.split(":", 1)[1].strip()
    value = float(token)
    if value.is_integer():
        return int(value)
    return value


def parse_graph_edge_list(graph_text: str) -> list[list[Any]]:
    edges: list[list[Any]] = []
    for item in re.findall(r"\(([^()]*)\)", graph_text):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"Unsupported edge format: {item!r}")
        u = _parse_node(parts[0])
        v = _parse_node(parts[1])
        if len(parts) == 2:
            edges.append([u, v])
        else:
            edges.append([u, v, _parse_weight(parts[2])])
    return edges


def parse_graph_nodes(nodes_text: str | None, edges: list[list[Any]]) -> list[int | str]:
    if nodes_text:
        tokens = re.findall(r"<[^>]+>|[^,\s\[\]]+", nodes_text)
        return [_parse_node(token) for token in tokens if token]

    node_set: set[int | str] = set()
    for edge in edges:
        node_set.add(edge[0])
        node_set.add(edge[1])
    return sorted(node_set, key=str)


def _infer_directed(sample: dict[str, Any]) -> bool:
    directed = sample.get("directed", False)
    if isinstance(directed, bool):
        return directed
    return str(directed).strip().lower() in {"1", "true", "yes"}


def build_graph_json(sample: dict[str, Any]) -> dict[str, Any]:
    edges = parse_graph_edge_list(str(sample["graph"]))
    nodes = parse_graph_nodes(str(sample.get("nodes", "")), edges)
    directed = _infer_directed(sample)

    adjacency: dict[str, list[Any]] = {str(node): [] for node in nodes}
    weighted = any(len(edge) == 3 for edge in edges)

    for edge in edges:
        u, v = edge[0], edge[1]
        if weighted:
            weight = edge[2]
            adjacency.setdefault(str(u), []).append({"node": v, "weight": weight})
            if not directed:
                adjacency.setdefault(str(v), []).append({"node": u, "weight": weight})
        else:
            adjacency.setdefault(str(u), []).append(v)
            if not directed:
                adjacency.setdefault(str(v), []).append(u)

    return {"directed": directed, "nodes": nodes, "adjacency": adjacency}


def _parse_int_from_question(question: str, pattern: str) -> int:
    match = re.search(pattern, question)
    if match is None:
        raise ValueError(f"Could not parse integer with pattern {pattern!r} from question: {question!r}")
    return int(match.group(1))


def _parse_two_ints(question: str, pattern: str) -> tuple[int, int]:
    match = re.search(pattern, question)
    if match is None:
        raise ValueError(f"Could not parse pair with pattern {pattern!r} from question: {question!r}")
    return int(match.group(1)), int(match.group(2))


def infer_tool_name(sample: dict[str, Any]) -> str:
    task_name = str(sample["task"])
    if task_name not in TASK_TO_TOOL:
        raise KeyError(f"Unsupported task for DecompGTI training: {task_name}")
    return TASK_TO_TOOL[task_name]


def build_tool_arguments(sample: dict[str, Any]) -> dict[str, Any]:
    task_name = str(sample["task"])
    question = str(sample["question"])
    graph_edges = parse_graph_edge_list(str(sample["graph"]))
    directed = _infer_directed(sample)

    if task_name in {"BFS", "DFS", "connected_component"}:
        source = _parse_int_from_question(question, r"node <(\d+)>")
        arguments: dict[str, Any] = {"edges": graph_edges, "source": source}
        if task_name != "connected_component":
            arguments["directed"] = directed
        return arguments

    if task_name in {"shortest_path", "maximum_flow", "connectivity"}:
        source, target = _parse_two_ints(question, r"node <(\d+)>.*?node <(\d+)>")
        arguments = {"edges": graph_edges, "source": source, "target": target}
        if task_name in {"shortest_path", "connectivity"}:
            arguments["directed"] = directed
        return arguments

    if task_name == "cycle":
        return {"edges": graph_edges, "directed": directed}

    if task_name == "topological_sort":
        return {"edges": graph_edges}

    if task_name == "bipartite":
        left_count = int(sample.get("n1", 0))
        if left_count <= 0:
            match = re.search(r"Nodes set 1 contains:(.*?)\.\n", question, flags=re.S)
            if match is None:
                raise ValueError(f"Could not infer bipartite left side from sample: {sample!r}")
            left_nodes = [_parse_node(token) for token in re.findall(r"<(\d+)>", match.group(1))]
        else:
            left_nodes = list(range(left_count))
        return {"edges": graph_edges, "left_nodes": left_nodes}

    if task_name == "MST":
        return {"edges": graph_edges, "algorithm": "kruskal"}

    raise KeyError(f"Unsupported task for argument extraction: {task_name}")


def _build_stage_records(sample: dict[str, Any]) -> list[StageRecord]:
    graph_json = build_graph_json(sample)
    tool_name = infer_tool_name(sample)
    arguments = build_tool_arguments(sample)

    graph_json_text = json.dumps(graph_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    tool_call_text = json.dumps({"tool_name": tool_name, "arguments": arguments}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    routing_text = json.dumps({"tool_name": tool_name}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    graph_instruction = (
        "Stage 1: convert the graph description into strict JSON adjacency form. "
        "Return JSON only with keys directed, nodes, and adjacency."
    )
    routing_instruction = (
        "Stage 2: given the extracted graph JSON and the question, identify the correct MCP tool. "
        f"Return JSON only.\n\n{get_model_tool_prompt_block()}"
    )

    args_schema = next(spec for spec in TOOL_SPECS if spec.name == tool_name).input_schema
    schema_text = json.dumps(args_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    argument_instruction = (
        "Stage 3: given the extracted graph JSON and the question, emit the exact MCP tool call JSON. "
        f"Tool: {tool_name}. Schema: {schema_text}. Return JSON only."
    )

    graph_input = str(sample["graph_nl"])
    graph_context_input = f"Graph JSON:\n{graph_json_text}\n\nQuestion:\n{sample['question']}"

    return [
        StageRecord(graph_instruction, graph_input, graph_json_text),
        StageRecord(routing_instruction, graph_context_input, routing_text),
        StageRecord(argument_instruction, graph_context_input, tool_call_text),
    ]


def build_decompgti_reasoning_records(sample: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stage_index, stage_record in enumerate(_build_stage_records(sample)):
        records.append(
            {
                "instruction": stage_record.instruction,
                "input": stage_record.input,
                "output": stage_record.output,
                "id": int(sample.get("id", 0)) * 10 + stage_index,
            }
        )
    return records


def load_task_samples(task_csv_path: Path) -> list[dict[str, Any]]:
    with task_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def build_reasoning_records_from_csv(task_csv_path: Path) -> list[dict[str, Any]]:
    rows = load_task_samples(task_csv_path)
    records: list[dict[str, Any]] = []
    for row in rows:
        records.extend(build_decompgti_reasoning_records(row))
    return records


def build_reasoning_records_for_tasks(dataset_root: Path, tasks: tuple[str, ...] = DECOMPGTI_TASKS) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task_name in tasks:
        task_csv = dataset_root / task_name / f"{task_name}.csv"
        if not task_csv.exists():
            raise FileNotFoundError(f"Missing raw GraphInstruct dataset: {task_csv}")
        records.extend(build_reasoning_records_from_csv(task_csv))
    return records