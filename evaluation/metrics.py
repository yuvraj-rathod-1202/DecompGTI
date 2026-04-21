"""
DecompGTI Evaluation Metrics.

Implements the 5 core metrics from the project proposal:
  1. JSON Validity Rate
  2. Tool Identification Accuracy
  3. Parameter Extraction Score (Precision / Recall / F1)
  4. Adjacency Extraction Accuracy (Edge-level F1)
  5. Task Success Rate (end-to-end exact match)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Result container for a single sample evaluation."""
    sample_id: int = 0
    task_type: str = ""
    graph_size: str = ""  # mini, small, medium, large
    num_nodes: int = 0

    # Metric flags
    json_valid: bool = False
    tool_correct: bool = False
    params_precision: float = 0.0
    params_recall: float = 0.0
    adj_edge_f1: float = 0.0
    task_success: bool = False

    # Debug info
    predicted_tool: str = ""
    expected_tool: str = ""
    predicted_params: dict = field(default_factory=dict)
    expected_params: dict = field(default_factory=dict)
    error_message: str = ""


@dataclass
class AggregateMetrics:
    """Aggregated metrics over a test set."""
    total_samples: int = 0
    json_validity_rate: float = 0.0
    tool_identification_accuracy: float = 0.0
    parameter_precision: float = 0.0
    parameter_recall: float = 0.0
    parameter_f1: float = 0.0
    adjacency_extraction_f1: float = 0.0
    task_success_rate: float = 0.0

    # Breakdown by graph size
    by_size: dict[str, dict[str, float]] = field(default_factory=dict)
    # Breakdown by task type
    by_task: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "json_validity_rate": round(self.json_validity_rate * 100, 1),
            "tool_identification_accuracy": round(self.tool_identification_accuracy * 100, 1),
            "parameter_precision": round(self.parameter_precision * 100, 1),
            "parameter_recall": round(self.parameter_recall * 100, 1),
            "parameter_f1": round(self.parameter_f1 * 100, 1),
            "adjacency_extraction_f1": round(self.adjacency_extraction_f1 * 100, 1),
            "task_success_rate": round(self.task_success_rate * 100, 1),
            "by_size": self.by_size,
            "by_task": self.by_task,
        }

    def print_report(self):
        """Print a formatted report table."""
        print("\n" + "=" * 55)
        print("  DecompGTI Evaluation Report")
        print("=" * 55)
        print(f"  Total Samples:                  {self.total_samples}")
        print("-" * 55)
        print(f"  JSON Validity Rate:             {self.json_validity_rate * 100:.1f}%")
        print(f"  Tool Identification Accuracy:   {self.tool_identification_accuracy * 100:.1f}%")
        print(f"  Parameter Precision:            {self.parameter_precision * 100:.1f}%")
        print(f"  Parameter Recall:               {self.parameter_recall * 100:.1f}%")
        print(f"  Parameter F1:                   {self.parameter_f1 * 100:.1f}%")
        print(f"  Adjacency Extraction F1:        {self.adjacency_extraction_f1 * 100:.1f}%")
        print(f"  Task Success Rate:              {self.task_success_rate * 100:.1f}%")
        print("-" * 55)

        if self.by_size:
            print("\n  By Graph Size:")
            print(f"  {'Size':<12} {'JSON%':>7} {'Tool%':>7} {'Param F1':>9} {'Success%':>9}")
            for size in ['mini', 'small', 'medium', 'large']:
                if size in self.by_size:
                    s = self.by_size[size]
                    print(f"  {size:<12} {s.get('json_validity', 0):>6.1f}% {s.get('tool_accuracy', 0):>6.1f}% {s.get('param_f1', 0):>8.1f}% {s.get('task_success', 0):>8.1f}%")

        if self.by_task:
            print("\n  By Task Type:")
            print(f"  {'Task':<25} {'Tool%':>7} {'Success%':>9}")
            for task, s in sorted(self.by_task.items()):
                print(f"  {task:<25} {s.get('tool_accuracy', 0):>6.1f}% {s.get('task_success', 0):>8.1f}%")

        print("=" * 55)


# ── Metric 1: JSON Validity ─────────────────────────────────────

def check_json_validity(raw_output: str) -> tuple[bool, dict | None]:
    """Check if the model output contains a valid, parseable JSON object.
    
    Returns (is_valid, parsed_dict_or_None).
    """
    # Find the first complete JSON object via brace matching
    brace_depth = 0
    start = None
    for i, ch in enumerate(raw_output):
        if ch == "{":
            if start is None:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                json_str = raw_output[start:i + 1]
                try:
                    parsed = json.loads(json_str)
                    # Check it has the expected top-level keys
                    has_keys = all(k in parsed for k in [
                        "step1_graph_extraction", "step2_tool_name", "step3_tool_parameters"
                    ])
                    return has_keys, parsed if has_keys else None
                except json.JSONDecodeError:
                    start = None
                    continue
    return False, None


# ── Metric 2: Tool Identification Accuracy ──────────────────────

# Normalize tool names (model may output slightly different names)
TOOL_ALIASES = {
    "shortest_path": "shortest_path",
    "dijkstra_shortest_path": "shortest_path",
    "dijkstra": "shortest_path",
    "depth_first_search": "dfs",
    "dfs": "dfs",
    "breadth_first_search": "bfs",
    "bfs": "bfs",
    "check_connectivity": "connectivity",
    "connectivity": "connectivity",
    "topological_sort": "topological_sort",
    "topo_sort": "topological_sort",
    "detect_cycle": "cycle",
    "cycle_detection": "cycle",
    "cycle": "cycle",
    "check_bipartite": "bipartite",
    "bipartite": "bipartite",
    "minimum_spanning_tree": "mst",
    "mst": "mst",
    "maximum_flow": "maximum_flow",
    "max_flow": "maximum_flow",
    "find_connected_components": "connected_components",
    "connected_component": "connected_components",
    "connected_components": "connected_components",
}


def normalize_tool_name(name: str) -> str:
    """Normalize a tool name to a canonical form."""
    return TOOL_ALIASES.get(name.lower().strip(), name.lower().strip())


def check_tool_accuracy(predicted_tool: str, expected_tool: str) -> bool:
    """Check if predicted tool matches expected (after normalization)."""
    return normalize_tool_name(predicted_tool) == normalize_tool_name(expected_tool)


# ── Metric 3: Parameter Extraction Score ────────────────────────

def check_parameter_extraction(
    predicted_params: dict,
    expected_params: dict,
) -> tuple[float, float]:
    """Compute precision and recall for parameter extraction.
    
    Returns (precision, recall).
    """
    if not expected_params and not predicted_params:
        return 1.0, 1.0  # Both empty = perfect
    if not expected_params:
        return 0.0 if predicted_params else 1.0, 1.0
    if not predicted_params:
        return 1.0, 0.0

    # Check each expected parameter
    correct = 0
    for key, expected_val in expected_params.items():
        if key in predicted_params:
            pred_val = predicted_params[key]
            # Coerce types for comparison
            try:
                if str(pred_val) == str(expected_val):
                    correct += 1
                elif int(pred_val) == int(expected_val):
                    correct += 1
            except (ValueError, TypeError):
                pass

    precision = correct / len(predicted_params) if predicted_params else 0.0
    recall = correct / len(expected_params) if expected_params else 0.0

    return precision, recall


# ── Metric 4: Adjacency Extraction Accuracy ─────────────────────

def parse_adjacency_string(adj_str: str, directed: bool) -> set[tuple]:
    """Parse the model's adjacency list string into a set of (u, v, w) edges."""
    edges = set()
    node_pattern = re.compile(r"(\d+)\s*:\s*\[(.*?)\]")
    edge_pattern = re.compile(r"\((\d+),\s*weight\s*:\s*(\d+)\)")

    for node_match in node_pattern.finditer(adj_str):
        u = int(node_match.group(1))
        neighbors_str = node_match.group(2)

        for edge_match in edge_pattern.finditer(neighbors_str):
            v = int(edge_match.group(1))
            w = int(edge_match.group(2))

            if directed:
                edges.add((u, v, w))
            else:
                # Canonical form for undirected
                edges.add((min(u, v), max(u, v), w))

    return edges


def check_adjacency_extraction(
    predicted_adj: str,
    expected_adj: str,
    directed: bool = False,
) -> float:
    """Compute edge-level F1 between predicted and expected adjacency lists."""
    try:
        pred_edges = parse_adjacency_string(predicted_adj, directed)
        true_edges = parse_adjacency_string(expected_adj, directed)
    except Exception:
        return 0.0

    if not true_edges and not pred_edges:
        return 1.0
    if not true_edges or not pred_edges:
        return 0.0

    tp = len(pred_edges & true_edges)
    precision = tp / len(pred_edges) if pred_edges else 0.0
    recall = tp / len(true_edges) if true_edges else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Metric 5: Task Success Rate ─────────────────────────────────

def check_task_success(
    predicted_answer: Any,
    expected_answer: Any,
    task_type: str = "",
    expected_adj: str = "",
    directed: bool = False,
    tolerance: float = 1e-6,
) -> bool:
    """Check if the final pipeline answer matches the ground truth.
    
    For numeric answers, uses tolerance-based comparison.
    For dict/list answers, uses string comparison or dynamic validation.
    """
    if predicted_answer is None or expected_answer is None:
        return predicted_answer is None and expected_answer is None

    # Handle Topological Sort Dynamically
    if task_type == "topological_sort" and isinstance(predicted_answer, list) and isinstance(expected_answer, list):
        if len(predicted_answer) != len(expected_answer):
            return False
        if set(str(x) for x in predicted_answer) != set(str(x) for x in expected_answer):
            return False
        
        # Build edges directly from ground truth string to verify order
        try:
            edges = parse_adjacency_string(expected_adj, directed=True)
            pred_str_list = [str(x) for x in predicted_answer]
            for u, v, _ in edges:
                u_str, v_str = str(u), str(v)
                if u_str in pred_str_list and v_str in pred_str_list:
                    if pred_str_list.index(u_str) >= pred_str_list.index(v_str):
                        return False  # Topological constraint violated!
            return True
        except Exception:
            return False

    # Handle lists properly (e.g. DFS and BFS traversals)
    if isinstance(predicted_answer, list) and isinstance(expected_answer, list):
        if len(predicted_answer) != len(expected_answer):
            return False
        # Since we enforce tie-breaking in tool execution, valid traversals
        # from identical extractions will be strictly numerically sorted if identical.
        # But we still allow set equivalence as a fallback if networkx fluctuates.
        if len(predicted_answer) > 0 and str(predicted_answer[0]) != str(expected_answer[0]):
            return False
        return set(str(x) for x in predicted_answer) == set(str(x) for x in expected_answer)

    # Numeric comparison
    try:
        pred_num = float(predicted_answer)
        exp_num = float(expected_answer)
        return abs(pred_num - exp_num) < tolerance
    except (ValueError, TypeError):
        pass

    # String comparison
    return str(predicted_answer).strip() == str(expected_answer).strip()


# ── Full Evaluation Pipeline ────────────────────────────────────

def evaluate_single_sample(
    raw_model_output: str,
    expected_tool: str,
    expected_params: dict,
    expected_adj: str,
    expected_directed: bool,
    expected_answer: Any,
    actual_answer: Any | None = None,  # from running the tool
) -> EvalResult:
    """Evaluate a single model output against ground truth."""
    result = EvalResult()
    result.expected_tool = expected_tool
    result.expected_params = expected_params

    # 1. JSON Validity
    result.json_valid, parsed = check_json_validity(raw_model_output)
    if not result.json_valid or parsed is None:
        result.error_message = "Invalid JSON output"
        return result

    # 2. Tool Identification
    result.predicted_tool = parsed.get("step2_tool_name", "")
    result.tool_correct = check_tool_accuracy(result.predicted_tool, expected_tool)

    # 3. Parameter Extraction
    result.predicted_params = parsed.get("step3_tool_parameters", {})
    result.params_precision, result.params_recall = check_parameter_extraction(
        result.predicted_params, expected_params
    )

    # 4. Adjacency Extraction
    graph_info = parsed.get("step1_graph_extraction", {})
    pred_adj = graph_info.get("adjacency_list", "")
    result.adj_edge_f1 = check_adjacency_extraction(
        pred_adj, expected_adj, directed=expected_directed
    )

    # 5. Task Success (if we ran the tool)
    if actual_answer is not None:
        result.task_success = check_task_success(
            actual_answer, 
            expected_answer,
            task_type=expected_tool,
            expected_adj=expected_adj,
            directed=expected_directed
        )

    return result


def aggregate_results(results: list[EvalResult]) -> AggregateMetrics:
    """Aggregate individual eval results into summary metrics."""
    if not results:
        return AggregateMetrics()

    metrics = AggregateMetrics()
    metrics.total_samples = len(results)

    # Overall averages
    metrics.json_validity_rate = sum(r.json_valid for r in results) / len(results)
    metrics.tool_identification_accuracy = sum(r.tool_correct for r in results) / len(results)

    valid_results = [r for r in results if r.json_valid]
    if valid_results:
        metrics.parameter_precision = sum(r.params_precision for r in valid_results) / len(valid_results)
        metrics.parameter_recall = sum(r.params_recall for r in valid_results) / len(valid_results)
        if metrics.parameter_precision + metrics.parameter_recall > 0:
            metrics.parameter_f1 = (
                2 * metrics.parameter_precision * metrics.parameter_recall
                / (metrics.parameter_precision + metrics.parameter_recall)
            )
        metrics.adjacency_extraction_f1 = sum(r.adj_edge_f1 for r in valid_results) / len(valid_results)

    metrics.task_success_rate = sum(r.task_success for r in results) / len(results)

    # Breakdown by size
    sizes = set(r.graph_size for r in results if r.graph_size)
    for size in sizes:
        size_results = [r for r in results if r.graph_size == size]
        n = len(size_results)
        if n == 0:
            continue
        valid_in_size = [r for r in size_results if r.json_valid]
        metrics.by_size[size] = {
            "count": n,
            "json_validity": sum(r.json_valid for r in size_results) / n * 100,
            "tool_accuracy": sum(r.tool_correct for r in size_results) / n * 100,
            "param_f1": (
                _safe_f1(
                    sum(r.params_precision for r in valid_in_size) / len(valid_in_size) if valid_in_size else 0,
                    sum(r.params_recall for r in valid_in_size) / len(valid_in_size) if valid_in_size else 0,
                ) * 100
            ),
            "task_success": sum(r.task_success for r in size_results) / n * 100,
        }

    # Breakdown by task
    tasks = set(r.task_type for r in results if r.task_type)
    for task in tasks:
        task_results = [r for r in results if r.task_type == task]
        n = len(task_results)
        if n == 0:
            continue
        metrics.by_task[task] = {
            "count": n,
            "tool_accuracy": sum(r.tool_correct for r in task_results) / n * 100,
            "task_success": sum(r.task_success for r in task_results) / n * 100,
        }

    return metrics


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
