"""
End-to-end inference script for DecompGTI.

Pipeline:
  1. Load the fine-tuned Qwen2.5-7B + LoRA adapter
  2. Send a graph reasoning question to the model
  3. Parse the model's 3-step JSON decomposition output
  4. Convert the adjacency list → edge list format
  5. Execute the correct graph tool via the MCP server
  6. Print the final mathematical answer
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

# ── Model loading ────────────────────────────────────────────────
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Graph tools (your existing MCP server code) ─────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from decompgti.mcp_server import graph_tools  # noqa: E402

# ── Tool name mapping ───────────────────────────────────────────
# The model outputs short names like "shortest_path", but your
# graph_tools.py uses "dijkstra_shortest_path". This map bridges them.
TOOL_NAME_MAP: dict[str, str] = {
    "shortest_path": "dijkstra_shortest_path",
    "dijkstra_shortest_path": "dijkstra_shortest_path",
    "bfs": "bfs",
    "dfs": "dfs",
    "maximum_flow": "maximum_flow",
    "bipartite_maximum_matching": "bipartite_maximum_matching",
    "topological_sort": "topological_sort",
    "cycle_detection": "cycle_detection",
    "minimum_spanning_tree": "minimum_spanning_tree",
}

TOOL_REGISTRY = {
    "bfs": graph_tools.bfs,
    "dfs": graph_tools.dfs,
    "dijkstra_shortest_path": graph_tools.dijkstra_shortest_path,
    "maximum_flow": graph_tools.maximum_flow,
    "bipartite_maximum_matching": graph_tools.bipartite_maximum_matching,
    "topological_sort": graph_tools.topological_sort,
    "cycle_detection": graph_tools.cycle_detection,
    "minimum_spanning_tree": graph_tools.minimum_spanning_tree,
}

SYSTEM_PROMPT = (
    "You are a graph reasoning agent. Given a graph description and a question, "
    "perform these three steps:\n"
    "1. Extract the graph structure as an adjacency list.\n"
    "2. Identify the correct graph algorithm tool to use.\n"
    "3. Extract the parameters required by that tool.\n"
    "Output your answer as a single valid JSON object."
)


# ── Adjacency list parser ───────────────────────────────────────
def parse_adjacency_list(adj_str: str, directed: bool) -> list[list[Any]]:
    """Convert the model's adjacency-list string into [[u, v, w], ...] edge list.

    The model outputs strings like:
      {0: [(3, weight:7), (5, weight:1)], 1: [(3, weight:8)]}

    This function parses that into:
      [[0, 3, 7], [0, 5, 1], [1, 3, 8]]
    """
    edges: list[list[Any]] = []
    seen: set[tuple[int, int]] = set()

    # Match each "node: [(neighbor, weight:W), ...]" block
    node_pattern = re.compile(r"(\d+)\s*:\s*\[(.*?)\]")
    edge_pattern = re.compile(r"\((\d+),\s*weight\s*:\s*(\d+)\)")

    for node_match in node_pattern.finditer(adj_str):
        u = int(node_match.group(1))
        neighbors_str = node_match.group(2)

        for edge_match in edge_pattern.finditer(neighbors_str):
            v = int(edge_match.group(1))
            w = int(edge_match.group(2))

            if directed:
                edges.append([u, v, w])
            else:
                # For undirected graphs, avoid duplicate edges
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    edges.append([u, v, w])

    return edges


# ── Model output parser ─────────────────────────────────────────
def parse_model_output(raw_text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from the model's raw output.

    The model sometimes repeats the JSON block multiple times
    (stop-token mismatch). This function grabs only the first one.
    """
    # Find the first complete JSON object
    brace_depth = 0
    start = None
    for i, ch in enumerate(raw_text):
        if ch == "{":
            if start is None:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                json_str = raw_text[start : i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    start = None  # try next block
                    continue

    raise ValueError(f"Could not extract valid JSON from model output:\n{raw_text[:500]}")


# ── Model loading ────────────────────────────────────────────────
def load_model(base_model: str, adapter_path: str, device: str = "auto"):
    """Load the base Qwen model and merge the LoRA adapter."""
    print(f"[1/3] Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"[2/3] Loading base model from {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    print(f"[3/3] Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("✅ Model loaded successfully!\n")
    return model, tokenizer


# ── Inference ────────────────────────────────────────────────────
def run_inference(model, tokenizer, question: str) -> str:
    """Send a graph question to the model and return raw text output."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=16384,
        temperature=0.1,
        top_p=0.7,
        do_sample=True,
        repetition_penalty=1.1,
    )

    # Decode only the newly generated tokens (skip the input prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Parameter Validation Fallback ───────────────────────────────
def validate_and_fix_params(tool_name: str, tool_params: dict, question: str) -> dict:
    """Ensure required parameters are present; extract from question if missing."""
    import re
    
    REQUIRED_PARAMS = {
        "dijkstra_shortest_path": ["source", "target"],
        "bfs": ["source"],
        "dfs": ["source"],
        "maximum_flow": ["source", "target"],
        "bipartite_maximum_matching": [],
        "topological_sort": [],
        "cycle_detection": [],
        "minimum_spanning_tree": [],
    }
    
    required = REQUIRED_PARAMS.get(tool_name, [])
    missing = [p for p in required if p not in tool_params]
    
    if not missing:
        return tool_params  # All good
    
    print(f"\n⚠️  Missing parameters {missing} — extracting from question...")
    
    # Simple regex-based extraction from the question text
    fixed_params = dict(tool_params)
    
    # Look for patterns like "from node X", "source X", "node X to node Y"
    source_match = re.search(
        r'(?:from|source|start(?:ing)?\s+(?:node|vertex)?)\s+(\d+)', 
        question, re.IGNORECASE
    )
    target_match = re.search(
        r'(?:to|target|destination|end(?:ing)?\s+(?:node|vertex)?)\s+(\d+)', 
        question, re.IGNORECASE
    )
    
    # Fallback: grab the last two numbers mentioned in the question
    all_numbers = re.findall(r'\b(\d+)\b', question)
    
    if "source" in missing:
        if source_match:
            fixed_params["source"] = int(source_match.group(1))
        elif len(all_numbers) >= 2:
            fixed_params["source"] = int(all_numbers[-2])
            print(f"   ⚠️  Guessed source={fixed_params['source']} from question numbers")
    
    if "target" in missing:
        if target_match:
            fixed_params["target"] = int(target_match.group(1))
        elif len(all_numbers) >= 1:
            fixed_params["target"] = int(all_numbers[-1])
            print(f"   ⚠️  Guessed target={fixed_params['target']} from question numbers")
    
    # Final check
    still_missing = [p for p in required if p not in fixed_params]
    if still_missing:
        raise ValueError(
            f"Could not resolve required parameters {still_missing} "
            f"for tool '{tool_name}'. Please include them explicitly in the question."
        )
    
    print(f"   ✅ Fixed params: {fixed_params}")
    return fixed_params


# ── End-to-end pipeline ─────────────────────────────────────────
def solve_graph_question(model, tokenizer, question: str) -> dict[str, Any]:
    """Full pipeline: question → model → parse → execute tool → answer."""

    print("=" * 60)
    print("📝 QUESTION:")
    print(question)
    print("=" * 60)

    # Step 1: Get model's decomposition
    print("\n🤖 Running model inference...")
    raw_output = run_inference(model, tokenizer, question)
    print(f"\n📤 RAW MODEL OUTPUT:\n{raw_output[:500]}")

    # Step 2: Parse the JSON
    parsed = parse_model_output(raw_output)
    print(f"\n📋 PARSED DECOMPOSITION:")
    print(json.dumps(parsed, indent=2))

    # Step 3: Extract the three steps
    adj_list_str = parsed["step1_graph_extraction"]["adjacency_list"]
    directed = parsed["step1_graph_extraction"].get("directed", False)
    model_tool_name = parsed["step2_tool_name"]
    tool_params = parsed["step3_tool_parameters"]

    # Step 4: Convert adjacency list → edge list
    edges = parse_adjacency_list(adj_list_str, directed=directed)
    print(f"\n🔗 CONVERTED EDGE LIST ({len(edges)} edges):")
    for e in edges[:10]:  # Show first 10
        print(f"   {e}")
    if len(edges) > 10:
        print(f"   ... and {len(edges) - 10} more")

    # Step 5: Map tool name
    actual_tool_name = TOOL_NAME_MAP.get(model_tool_name, model_tool_name)
    if actual_tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool '{model_tool_name}' (mapped to '{actual_tool_name}'). "
            f"Available: {list(TOOL_REGISTRY.keys())}"
        )

    print(f"\n🔧 TOOL: {model_tool_name} → {actual_tool_name}")

    # ✅ ADD THIS: validate and fix missing parameters
    tool_params = validate_and_fix_params(actual_tool_name, tool_params, question)
    print(f"   PARAMS: {tool_params}")

    # Step 6: Execute the tool
    tool_fn = TOOL_REGISTRY[actual_tool_name]
    tool_args = {"edges": edges, "directed": directed, **tool_params}
    
    try:
        result = tool_fn(**tool_args)
    except Exception as e:
        print(f"\n❌ Tool Execution Failed: {e}")
        result = {"error": str(e)}

    print(f"\n✅ FINAL ANSWER:")
    print(json.dumps(result, indent=2))
    print("=" * 60)

    return {
        "question": question,
        "model_decomposition": parsed,
        "tool_executed": actual_tool_name,
        "result": result,
    }


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DecompGTI End-to-End Inference")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter",
        default="/home/ramji.purwar/DecompGTI/DecompGTI/GraphInstruct/LLaMAFactory/saves/Qwen2.5-7B/lora/train_3_epochs_fix",
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map (auto, cpu, cuda:0, etc.)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="A graph reasoning question to solve. If not provided, uses a demo question.",
    )
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.base_model, args.adapter, args.device)

    # Use provided question or ask interactively
    if args.question:
        question = args.question
    else:
        print("\n" + "="*60)
        print("Please enter your graph description and question below.")
        print("(Press Ctrl+D on a new line to finish and submit):")
        print("="*60 + "\n")
        
        question = sys.stdin.read().strip()
        
        if not question:
            print("No question provided. Exiting.")
            sys.exit(0)

    solve_graph_question(model, tokenizer, question)


if __name__ == "__main__":
    main()
