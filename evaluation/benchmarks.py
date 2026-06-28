"""
DecompGTI Benchmark Test Set Generator.

Generates test sets of varying graph sizes with known ground-truth answers.
Uses the existing GTG task modules to create graphs and questions,
then computes the correct answers via the MCP graph tools.

Usage:
    python -m evaluation.benchmarks --output data/ --samples-per-config 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── Add paths so we can import from both GTG and src ────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPHINSTRUCT_ROOT = PROJECT_ROOT / "GraphInstruct"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(GRAPHINSTRUCT_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "graph_tools",
    PROJECT_ROOT / "src" / "decompgti" / "mcp_server" / "graph_tools.py",
)
graph_tools = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(graph_tools)  # type: ignore


# ── Graph generation (standalone, no GTG dependency) ────────────
import networkx as nx


def generate_random_graph(
    num_nodes: int,
    directed: bool | None = None,
    weighted: bool = True,
    density: float = 0.3,
) -> nx.Graph | nx.DiGraph:
    """Generate a random graph with edge weights."""
    if directed is None:
        directed = random.random() < 0.5

    if directed:
        g = nx.gnp_random_graph(num_nodes, density, directed=True)
        g = nx.DiGraph(g)
    else:
        g = nx.gnp_random_graph(num_nodes, density)
        g = nx.Graph(g)

    # Remove isolated nodes by adding at least one edge
    for node in list(g.nodes()):
        if g.degree(node) == 0:
            target = random.choice([n for n in g.nodes() if n != node])
            g.add_edge(node, target)

    # Add weights
    if weighted:
        for u, v in g.edges():
            g[u][v]["weight"] = random.randint(1, 10)

    return g


def graph_to_natural_language(g: nx.Graph | nx.DiGraph) -> str:
    """Convert a NetworkX graph to the standard NL description format."""
    s = ""
    for u in g.nodes():
        neighbors = []
        for v in g.neighbors(u):
            w = g[u][v].get("weight", 1)
            neighbors.append((v, w))

        if not neighbors:
            continue
        elif len(neighbors) == 1:
            v, w = neighbors[0]
            s += f"Node {u} is connected to node {v} (weight: {w}).\n"
        else:
            parts = ", ".join(f"{v} (weight: {w})" for v, w in neighbors)
            s += f"Node {u} is connected to nodes {parts}.\n"

    return s.strip()


def graph_to_adj_string(g: nx.Graph | nx.DiGraph) -> str:
    """Convert graph to the adjacency list string format used in training."""
    parts = []
    for u in g.nodes():
        neighbors = []
        for v in g.neighbors(u):
            w = g[u][v].get("weight", 1)
            neighbors.append(f"({v}, weight:{w})")
        parts.append(f"{u}: [{', '.join(neighbors)}]")
    return "{" + ",\\n".join(parts) + "}"


# ── Task generators ─────────────────────────────────────────────

TASK_GENERATORS: dict[str, Any] = {}


def register_task(name: str):
    def decorator(fn):
        TASK_GENERATORS[name] = fn
        return fn
    return decorator


@register_task("shortest_path")
def gen_shortest_path(g: nx.Graph | nx.DiGraph) -> dict:
    nodes = list(g.nodes())
    source = random.choice(nodes)
    target = random.choice([n for n in nodes if n != source])

    question = f"Calculate the distance of the shortest path from node {source} to node {target}."
    tool = "shortest_path"
    params = {"source": source, "target": target}

    # Compute ground truth
    edges = [[u, v, g[u][v].get("weight", 1)] for u, v in g.edges()]
    try:
        result = graph_tools.dijkstra_shortest_path(
            edges=edges, source=source, target=target, directed=g.is_directed()
        )
        answer = result["distance"]
    except Exception:
        answer = None  # No path exists

    return {"question": question, "tool": tool, "params": params, "answer": answer}


@register_task("bfs")
def gen_bfs(g: nx.Graph | nx.DiGraph) -> dict:
    source = random.choice(list(g.nodes()))
    target = random.choice([n for n in g.nodes() if n != source])
    question = f"Perform a breadth-first search starting from node {source}."
    tool = "breadth_first_search"
    params = {"source": source}

    edges = [[u, v, g[u][v].get("weight", 1)] for u, v in g.edges()]
    try:
        result = graph_tools.bfs(edges=edges, source=source, directed=g.is_directed())
        answer = result["order"]
    except Exception:
        answer = None

    return {"question": question, "tool": tool, "params": params, "answer": answer}


@register_task("dfs")
def gen_dfs(g: nx.Graph | nx.DiGraph) -> dict:
    source = random.choice(list(g.nodes()))
    question = f"Perform a depth-first search starting from node {source}."
    tool = "depth_first_search"
    params = {"source": source}

    edges = [[u, v, g[u][v].get("weight", 1)] for u, v in g.edges()]
    try:
        result = graph_tools.dfs(edges=edges, source=source, directed=g.is_directed())
        answer = result["order"]
    except Exception:
        answer = None

    return {"question": question, "tool": tool, "params": params, "answer": answer}


@register_task("cycle")
def gen_cycle(g: nx.Graph | nx.DiGraph) -> dict:
    question = "Determine whether the graph contains a cycle."
    tool = "detect_cycle"
    params = {}

    edges = [[u, v, g[u][v].get("weight", 1)] for u, v in g.edges()]
    try:
        result = graph_tools.cycle_detection(edges=edges, directed=g.is_directed())
        answer = result["has_cycle"]
    except Exception:
        answer = None

    return {"question": question, "tool": tool, "params": params, "answer": answer}


@register_task("connectivity")
def gen_connectivity(g: nx.Graph | nx.DiGraph) -> dict:
    question = "Determine whether the graph is connected."
    tool = "check_connectivity"
    params = {}

    try:
        if g.is_directed():
            answer = nx.is_strongly_connected(g)
        else:
            answer = nx.is_connected(g)
    except Exception:
        answer = None

    return {"question": question, "tool": tool, "params": params, "answer": answer}


@register_task("topological_sort")
def gen_topo_sort(g: nx.Graph | nx.DiGraph) -> dict:
    # Need a DAG for this
    question = "Find a topological ordering of the graph."
    tool = "topological_sort"
    params = {}

    edges = [[u, v, g[u][v].get("weight", 1)] for u, v in g.edges()]
    try:
        result = graph_tools.topological_sort(edges=edges)
        answer = result["order"]
    except Exception:
        answer = None

    return {"question": question, "tool": tool, "params": params, "answer": answer}


@register_task("mst")
def gen_mst(g: nx.Graph | nx.DiGraph) -> dict:
    question = "Find the minimum spanning tree of the graph."
    tool = "minimum_spanning_tree"
    params = {}

    edges = [[u, v, g[u][v].get("weight", 1)] for u, v in g.edges()]
    try:
        result = graph_tools.minimum_spanning_tree(edges=edges)
        answer = result["total_weight"]
    except Exception:
        answer = None

    return {"question": question, "tool": tool, "params": params, "answer": answer}


@register_task("maximum_flow")
def gen_max_flow(g: nx.Graph | nx.DiGraph) -> dict:
    nodes = list(g.nodes())
    source = random.choice(nodes)
    target = random.choice([n for n in nodes if n != source])

    question = f"Calculate the maximum flow from node {source} to node {target}."
    tool = "maximum_flow"
    params = {"source": source, "target": target}

    edges = [[u, v, g[u][v].get("weight", 1)] for u, v in g.edges()]
    try:
        result = graph_tools.maximum_flow(edges=edges, source=source, target=target)
        answer = result["max_flow"]
    except Exception:
        answer = None

    return {"question": question, "tool": tool, "params": params, "answer": answer}


# ── Size configs ─────────────────────────────────────────────────

SIZE_CONFIGS = {
    "mini":   {"num_nodes_range": (5, 7),   "density": 0.4},
    "small":  {"num_nodes_range": (8, 15),  "density": 0.3},
    "medium": {"num_nodes_range": (16, 25), "density": 0.25},
    "large":  {"num_nodes_range": (26, 50), "density": 0.15},
}


# ── Main generation ─────────────────────────────────────────────

def generate_test_set(
    size: str = "small",
    tasks: list[str] | None = None,
    samples_per_task: int = 12,
    seed: int = 42,
) -> list[dict]:
    """Generate a test set with ground-truth answers.
    
    Args:
        size: Graph size category (mini, small, medium, large)
        tasks: Which tasks to include (None = all)
        samples_per_task: Number of samples per task
        seed: Random seed for reproducibility
        
    Returns:
        List of test samples, each containing:
        - graph_nl: natural language description
        - graph_adj: adjacency list string
        - question: the question text
        - directed: whether the graph is directed
        - num_nodes: number of nodes
        - task_type: task name
        - graph_size: size category
        - expected_tool: correct tool name
        - expected_params: correct parameters
        - expected_answer: correct answer
    """
    random.seed(seed)
    np.random.seed(seed)

    if tasks is None:
        tasks = list(TASK_GENERATORS.keys())

    config = SIZE_CONFIGS[size]
    test_samples = []
    sample_id = 0

    for task_name in tasks:
        generator = TASK_GENERATORS[task_name]
        generated = 0
        attempts = 0
        max_attempts = samples_per_task * 10

        while generated < samples_per_task and attempts < max_attempts:
            attempts += 1

            num_nodes = random.randint(*config["num_nodes_range"])

            # Special cases
            if task_name == "topological_sort":
                directed = True
            elif task_name == "mst":
                directed = False
            elif task_name == "maximum_flow":
                directed = True
            else:
                directed = random.random() < 0.5

            g = generate_random_graph(
                num_nodes=num_nodes,
                directed=directed,
                density=config["density"],
            )

            # For topological sort, ensure DAG
            if task_name == "topological_sort":
                if not nx.is_directed_acyclic_graph(g):
                    continue

            task_data = generator(g)

            # Skip samples where the answer couldn't be computed
            if task_data["answer"] is None:
                continue

            sample = {
                "sample_id": sample_id,
                "graph_nl": graph_to_natural_language(g),
                "graph_adj": graph_to_adj_string(g),
                "question": task_data["question"],
                "directed": directed,
                "num_nodes": num_nodes,
                "task_type": task_name,
                "graph_size": size,
                "expected_tool": task_data["tool"],
                "expected_params": task_data["params"],
                "expected_answer": task_data["answer"],
            }

            test_samples.append(sample)
            sample_id += 1
            generated += 1

    return test_samples


def main():
    parser = argparse.ArgumentParser(description="Generate DecompGTI test sets")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--samples-per-task", type=int, default=12, help="Samples per task per size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sizes", nargs="+", default=["mini", "small", "medium", "large"],
        help="Which size categories to generate"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        print(f"\n{'='*50}")
        print(f"Generating {size} test set...")
        print(f"{'='*50}")

        test_set = generate_test_set(
            size=size,
            samples_per_task=args.samples_per_task,
            seed=args.seed,
        )

        output_file = output_dir / f"test_set_{size}.json"
        with open(output_file, "w") as f:
            json.dump(test_set, f, indent=2, default=str)

        # Print summary
        task_counts = {}
        for s in test_set:
            task_counts[s["task_type"]] = task_counts.get(s["task_type"], 0) + 1

        print(f"  Total samples: {len(test_set)}")
        for task, count in sorted(task_counts.items()):
            print(f"    {task}: {count}")
        print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    main()
