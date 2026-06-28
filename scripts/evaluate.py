"""
DecompGTI Evaluation Runner.

Runs the full evaluation pipeline on one or more test sets with a single
model load. Shows tqdm progress bars for every step.

Usage:
    # ALL test sets at once:
    python scripts/evaluate.py \
        --base-model Qwen/Qwen2.5-7B \
        --adapter GraphInstruct/LLaMAFactory/saves/Qwen2.5-7B/lora/train_3_epochs_fix \
        --test-set data/test_set_mini.json data/test_set_small.json \
                   data/test_set_medium.json data/test_set_large.json \
        --output-dir evaluation/results/qwen7b

    # Single test set:
    python scripts/evaluate.py \
        --base-model Qwen/Qwen2.5-7B \
        --adapter GraphInstruct/LLaMAFactory/saves/Qwen2.5-7B/lora/train_3_epochs_fix \
        --test-set data/test_set_small.json \
        --output-dir evaluation/results/qwen7b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import importlib.util
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    EvalResult,
    aggregate_results,
    check_json_validity,
    check_tool_accuracy,
    check_parameter_extraction,
    check_adjacency_extraction,
    check_task_success,
)

_spec = importlib.util.spec_from_file_location(
    "graph_tools",
    PROJECT_ROOT / "src" / "decompgti" / "mcp_server" / "graph_tools.py",
)
graph_tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(graph_tools)

# ── Tool execution ──────────────────────────────────────────────

TOOL_DISPATCH = {
    "shortest_path": "dijkstra_shortest_path",
    "dijkstra_shortest_path": "dijkstra_shortest_path",
    "depth_first_search": "dfs", "dfs": "dfs",
    "breadth_first_search": "bfs", "bfs": "bfs",
    "detect_cycle": "cycle_detection",
    "cycle_detection": "cycle_detection", "cycle": "cycle_detection",
    "check_connectivity": "check_connectivity", "connectivity": "check_connectivity",
    "topological_sort": "topological_sort",
    "check_bipartite": None, "bipartite": None,
    "minimum_spanning_tree": "minimum_spanning_tree",
    "mst": "minimum_spanning_tree",
    "maximum_flow": "maximum_flow", "max_flow": "maximum_flow",
    "find_connected_components": None,
    "connected_components": None, "connected_component": None,
}

TOOL_FUNCTIONS = {
    "dijkstra_shortest_path": graph_tools.dijkstra_shortest_path,
    "bfs": graph_tools.bfs,
    "dfs": graph_tools.dfs,
    "maximum_flow": graph_tools.maximum_flow,
    "topological_sort": graph_tools.topological_sort,
    "cycle_detection": graph_tools.cycle_detection,
    "minimum_spanning_tree": graph_tools.minimum_spanning_tree,
    "check_connectivity": graph_tools.check_connectivity,
}

SAMPLE_LIMIT = 100

def parse_adjacency_to_edges(adj_str, directed):
    edges, seen = [], set()
    node_pat = re.compile(r"(\d+)\s*:\s*\[(.*?)\]")
    edge_pat = re.compile(r"\((\d+),\s*weight\s*:\s*(\d+)\)")
    for nm in node_pat.finditer(adj_str):
        u = int(nm.group(1))
        for em in edge_pat.finditer(nm.group(2)):
            v, w = int(em.group(1)), int(em.group(2))
            if directed:
                edges.append([u, v, w])
            else:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    edges.append([u, v, w])
    return edges


def execute_tool(parsed_output):
    tool_name = parsed_output.get("step2_tool_name", "")
    params = parsed_output.get("step3_tool_parameters", {})
    graph_info = parsed_output.get("step1_graph_extraction", {})
    adj_str = graph_info.get("adjacency_list", "")
    directed = graph_info.get("directed", False)

    dispatch_name = TOOL_DISPATCH.get(tool_name.lower())
    if dispatch_name is None or dispatch_name not in TOOL_FUNCTIONS:
        return None

    tool_fn = TOOL_FUNCTIONS[dispatch_name]
    edges = parse_adjacency_to_edges(adj_str, directed)

    try:
        kwargs = {"edges": edges}
        if dispatch_name == "dijkstra_shortest_path":
            kwargs.update(source=params.get("source"), target=params.get("target"), directed=directed)
        elif dispatch_name in ("bfs", "dfs"):
            kwargs.update(source=params.get("source"), directed=directed)
        elif dispatch_name == "maximum_flow":
            kwargs.update(source=params.get("source"), target=params.get("target"))
        elif dispatch_name == "cycle_detection":
            kwargs["directed"] = directed
        elif dispatch_name == "check_connectivity":
            kwargs.update(source=params.get("source"), target=params.get("target"), directed=directed)

        result = tool_fn(**kwargs)
        for key in ("distance", "max_flow", "has_cycle", "connected", "total_weight", "order"):
            if key in result:
                return result[key]
        return result
    except Exception:
        return None


# ── Model loading ───────────────────────────────────────────────

def load_model(base_model, adapter_path, device="auto"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading base model from {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map=device, dtype=torch.bfloat16, trust_remote_code=True
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("Model loaded!\n")
    return model, tokenizer


# ── Prompt building ─────────────────────────────────────────────
# Training used LLaMAFactory Alpaca format with "default" template.
# Dataset fields: instruction, input, output, system
# Template: "Human: {instruction}\n{input}<eos>\nAssistant:"

INSTRUCTION = (
    "You are a graph reasoning agent. Given a graph description and a question, "
    "perform these three steps:\n"
    "1. Extract the graph structure as an adjacency list.\n"
    "2. Identify the correct graph algorithm tool to use.\n"
    "3. Extract the parameters required by that tool.\n"
    "When multiple nodes can be visited next in a traversal, always visit the node with the lowest numerical ID first.\n"
    "Output your answer as a single valid JSON object."
)


def build_prompt(tokenizer, user_input):
    """Build prompt matching LLaMAFactory default template exactly."""
    eos = tokenizer.eos_token
    # Match the training format: Human: {instruction}\n{input}<eos>\nAssistant:
    prompt = f"Human: {INSTRUCTION}\n{user_input}{eos}\nAssistant:"
    return prompt


def run_model_inference(model, tokenizer, question):
    """Run greedy inference — no sampling, no NaN risk."""
    import torch

    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False,            # greedy — deterministic, no NaN
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Evaluation loop ────────────────────────────────────────────

def run_evaluation(test_set, model=None, tokenizer=None, predictions=None, verbose=True):
    from tqdm import tqdm

    results = []
    all_predictions = predictions or []

    desc = "Evaluating"
    iterator = tqdm(test_set, desc=desc, disable=not verbose, ncols=100)
    for i, sample in enumerate(iterator):
        # Get model output
        if predictions and i < len(predictions):
            raw_output = predictions[i].get("raw_output", "")
        elif model is not None:
            input_text = sample["graph_nl"] + "\n\nQuestion: " + sample["question"]
            t0 = time.time()
            raw_output = run_model_inference(model, tokenizer, input_text)
            elapsed = time.time() - t0
            all_predictions.append({
                "sample_id": sample.get("sample_id", i),
                "raw_output": raw_output,
                "inference_time_s": round(elapsed, 2),
            })
            iterator.set_postfix({"last_t": f"{elapsed:.1f}s"})
        else:
            raise ValueError("Must provide either (model, tokenizer) or predictions")

        # Parse and evaluate
        result = EvalResult()
        result.sample_id = sample.get("sample_id", i)
        result.task_type = sample.get("task_type", "")
        result.graph_size = sample.get("graph_size", "")
        result.num_nodes = sample.get("num_nodes", 0)

        result.json_valid, parsed = check_json_validity(raw_output)

        if result.json_valid and parsed:
            result.predicted_tool = parsed.get("step2_tool_name", "")
            result.expected_tool = sample.get("expected_tool", "")
            result.tool_correct = check_tool_accuracy(result.predicted_tool, result.expected_tool)

            result.predicted_params = parsed.get("step3_tool_parameters", {})
            result.expected_params = sample.get("expected_params", {})
            result.params_precision, result.params_recall = check_parameter_extraction(
                result.predicted_params, result.expected_params
            )

            graph_info = parsed.get("step1_graph_extraction", {})
            pred_adj = graph_info.get("adjacency_list", "")
            expected_adj = sample.get("graph_adj", "")
            directed = sample.get("directed", False)
            result.adj_edge_f1 = check_adjacency_extraction(pred_adj, expected_adj, directed=directed)

            try:
                actual_answer = execute_tool(parsed)
                expected_answer = sample.get("expected_answer")
                result.task_success = check_task_success(actual_answer, expected_answer)
            except Exception:
                result.task_success = False
        else:
            result.error_message = "Invalid JSON"

        results.append(result)

    return results, all_predictions


# ── Main ────────────────────────────────────────────────────────

def evaluate_single_test_set(test_set_path, model, tokenizer, output_dir, base_model_name, adapter_name):
    """Evaluate one test set, print report, save results."""
    test_set_path = Path(test_set_path)
    with open(test_set_path) as f:
        test_set = json.load(f)
        test_set = test_set[:SAMPLE_LIMIT] if SAMPLE_LIMIT else test_set

    set_name = test_set_path.stem.replace("test_set_", "")
    print(f"\n{'='*60}")
    print(f"  Evaluating: {set_name} ({len(test_set)} samples)")
    print(f"{'='*60}")

    t0 = time.time()
    results, predictions = run_evaluation(test_set, model=model, tokenizer=tokenizer)
    elapsed = time.time() - t0

    metrics = aggregate_results(results)
    metrics.print_report()
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(test_set):.2f}s/sample)")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        results_file = out / f"{set_name}_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "config": {
                    "base_model": base_model_name,
                    "adapter": adapter_name,
                    "test_set": str(test_set_path),
                    "total_time_s": round(elapsed, 1),
                    "samples": len(test_set),
                },
                "metrics": metrics.to_dict(),
            }, f, indent=2)
        print(f"  Results: {results_file}")

        pred_file = out / f"{set_name}_predictions.json"
        with open(pred_file, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"  Predictions: {pred_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="DecompGTI Evaluation Runner")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--test-set", type=str, nargs="+", required=True,
                        help="One or more test set JSON files")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Pre-saved predictions file (single test-set only)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (one file per test set)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Load model once
    model, tokenizer = None, None
    if args.predictions:
        # predictions mode only supports single test set
        if len(args.test_set) != 1:
            print("ERROR: --predictions only supports a single --test-set")
            sys.exit(1)
        with open(args.predictions) as f:
            predictions = json.load(f)
        with open(args.test_set[0]) as f:
            test_set = json.load(f)
        print(f"Loaded {len(predictions)} predictions, {len(test_set)} test samples")
        results, _ = run_evaluation(test_set, predictions=predictions)
        metrics = aggregate_results(results)
        metrics.print_report()
        return

    if not args.adapter:
        print("ERROR: Must provide --adapter for live inference or --predictions")
        sys.exit(1)

    model, tokenizer = load_model(args.base_model, args.adapter, args.device)

    print(f"\nWill evaluate {len(args.test_set)} test set(s):")
    for ts in args.test_set:
        with open(ts) as f:
            n = len(json.load(f))
        print(f"  - {Path(ts).name}: {n} samples")

    all_metrics = {}
    total_start = time.time()

    for ts_path in args.test_set:
        metrics = evaluate_single_test_set(
            ts_path, model, tokenizer, args.output_dir,
            args.base_model, args.adapter,
        )
        set_name = Path(ts_path).stem.replace("test_set_", "")
        all_metrics[set_name] = metrics

    total_elapsed = time.time() - total_start

    # Print combined summary
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print(f"  COMBINED SUMMARY ({len(all_metrics)} test sets)")
        print(f"{'='*60}")
        header = f"  {'Set':<10} {'N':>5} {'JSON%':>7} {'Tool%':>7} {'PF1':>7} {'Adj':>7} {'Succ%':>7}"
        print(header)
        print(f"  {'-'*52}")
        for name, m in all_metrics.items():
            print(f"  {name:<10} {m.total_samples:>5} "
                  f"{m.json_validity_rate*100:>6.1f}% "
                  f"{m.tool_identification_accuracy*100:>6.1f}% "
                  f"{m.parameter_f1*100:>6.1f}% "
                  f"{m.adjacency_extraction_f1*100:>6.1f}% "
                  f"{m.task_success_rate*100:>6.1f}%")
        print(f"  {'-'*52}")
        print(f"  Total time: {total_elapsed:.0f}s")
        print(f"{'='*60}")

    if args.output_dir:
        summary_file = Path(args.output_dir) / "summary.json"
        summary = {name: m.to_dict() for name, m in all_metrics.items()}
        summary["_total_time_s"] = round(total_elapsed, 1)
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
