"""
DecompGTI Web Demo.

A Gradio-based interactive demo that shows the full DecompGTI pipeline:
  1. User enters a graph description + question
  2. Model decomposes it into structured JSON
  3. MCP tool executes the algorithm
  4. Results are displayed step-by-step

Usage:
    pip install gradio
    python web/app.py

    # With a specific model:
    python web/app.py --config configs/qwen2.5_7b.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "graph_tools",
    PROJECT_ROOT / "src" / "decompgti" / "mcp_server" / "graph_tools.py",
)
graph_tools = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(graph_tools)  # type: ignore

# ── Example prompts ─────────────────────────────────────────────

EXAMPLES = [
    # Mini graph — shortest path
    {
        "name": "🔹 Mini Graph — Shortest Path (5 nodes)",
        "input": (
            "Node 0 is connected to nodes 1 (weight: 3), 2 (weight: 7).\n"
            "Node 1 is connected to nodes 0 (weight: 3), 3 (weight: 2), 2 (weight: 1).\n"
            "Node 2 is connected to nodes 0 (weight: 7), 1 (weight: 1), 4 (weight: 5).\n"
            "Node 3 is connected to nodes 1 (weight: 2), 4 (weight: 4).\n"
            "Node 4 is connected to nodes 2 (weight: 5), 3 (weight: 4).\n\n"
            "Question: Calculate the distance of the shortest path from node 0 to node 4."
        ),
    },
    # Medium graph — cycle detection
    {
        "name": "🔹 Medium Graph — Cycle Detection (8 nodes)",
        "input": (
            "Node 0 is connected to nodes 1 (weight: 5), 3 (weight: 2).\n"
            "Node 1 is connected to nodes 0 (weight: 5), 2 (weight: 4).\n"
            "Node 2 is connected to nodes 1 (weight: 4), 3 (weight: 6).\n"
            "Node 3 is connected to nodes 0 (weight: 2), 2 (weight: 6), 4 (weight: 3).\n"
            "Node 4 is connected to nodes 3 (weight: 3), 5 (weight: 7).\n"
            "Node 5 is connected to nodes 4 (weight: 7), 6 (weight: 1).\n"
            "Node 6 is connected to nodes 5 (weight: 1), 7 (weight: 8).\n"
            "Node 7 is connected to node 6 (weight: 8).\n\n"
            "Question: Determine whether the graph contains a cycle."
        ),
    },
    # Directed graph — maximum flow
    {
        "name": "🔹 Directed Graph — Maximum Flow (6 nodes)",
        "input": (
            "Node 0 is connected to nodes 1 (weight: 10), 2 (weight: 8).\n"
            "Node 1 is connected to nodes 3 (weight: 5), 2 (weight: 2).\n"
            "Node 2 is connected to nodes 4 (weight: 10).\n"
            "Node 3 is connected to node 5 (weight: 7).\n"
            "Node 4 is connected to nodes 3 (weight: 8), 5 (weight: 10).\n\n"
            "Question: Calculate the maximum flow from node 0 to node 5."
        ),
    },
]


# ── Tool execution ──────────────────────────────────────────────

TOOL_DISPATCH = {
    "shortest_path": "dijkstra_shortest_path",
    "dijkstra_shortest_path": "dijkstra_shortest_path",
    "depth_first_search": "dfs",
    "dfs": "dfs",
    "breadth_first_search": "bfs",
    "bfs": "bfs",
    "detect_cycle": "cycle_detection",
    "cycle_detection": "cycle_detection",
    "cycle": "cycle_detection",
    "topological_sort": "topological_sort",
    "minimum_spanning_tree": "minimum_spanning_tree",
    "mst": "minimum_spanning_tree",
    "maximum_flow": "maximum_flow",
    "max_flow": "maximum_flow",
    "check_connectivity": "check_connectivity",
    "connectivity": "check_connectivity",
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


# ── FIX 1: Correct adjacency list parser ────────────────────────
# The old version split on commas first, which destroyed "(4, weight:7)"
# into ["(4", " weight:7)"] and caused weights to be misread as node IDs.
# This version (copied from evaluate.py) uses a proper two-regex approach:
#   - node_pat matches each "N: [...]" block
#   - edge_pat matches "(v, weight:w)" tuples within that block
def parse_adjacency_to_edges(adj_str: str, directed: bool) -> list[list]:
    edges: list[list] = []
    seen: set[tuple[int, int]] = set()
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


def parse_model_output(raw_text: str) -> dict | None:
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
                try:
                    return json.loads(raw_text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def execute_parsed_tool(parsed: dict) -> dict:
    """Execute tool and return result dict."""
    tool_name = parsed.get("step2_tool_name", "")
    params = parsed.get("step3_tool_parameters", {})
    graph_info = parsed.get("step1_graph_extraction", {})
    adj_str = graph_info.get("adjacency_list", "")
    directed = graph_info.get("directed", False)

    dispatch_name = TOOL_DISPATCH.get(tool_name.lower())
    if not dispatch_name or dispatch_name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {tool_name}"}

    tool_fn = TOOL_FUNCTIONS[dispatch_name]
    edges = parse_adjacency_to_edges(adj_str, directed)

    # FIX 2: maximum_flow must NOT receive directed= kwarg.
    # graph_tools.maximum_flow always builds a DiGraph internally.
    # Passing directed= causes a TypeError.
    kwargs: dict = {"edges": edges}
    if dispatch_name == "dijkstra_shortest_path":
        kwargs.update(source=params.get("source"), target=params.get("target"), directed=directed)
    elif dispatch_name in ("bfs", "dfs"):
        kwargs.update(source=params.get("source"), directed=directed)
    elif dispatch_name == "maximum_flow":
        # No directed= here — graph_tools.maximum_flow doesn't accept it
        kwargs.update(source=params.get("source"), target=params.get("target"))
    elif dispatch_name == "cycle_detection":
        kwargs["directed"] = directed
    elif dispatch_name == "check_connectivity":
        kwargs.update(source=params.get("source"), target=params.get("target"), directed=directed)
    # topological_sort and minimum_spanning_tree only need edges

    try:
        return tool_fn(**kwargs)
    except Exception as e:
        return {"error": str(e)}


# ── Gradio app ──────────────────────────────────────────────────

def create_app(config_path: str | None = None):
    """Create the Gradio app."""
    import gradio as gr

    # Load config
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    model_name = config.get("model", {}).get("name", "DecompGTI")

    # Model loading (lazy)
    _model_cache: dict = {}

    def get_model():
        if "model" not in _model_cache:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            mc = config.get("model", {})
            base = mc.get("base", "Qwen/Qwen2.5-7B")
            adapter = mc.get("adapter", "")

            if not adapter:
                raise ValueError("No adapter path in config. Set model.adapter in your YAML.")

            adapter_path = Path(adapter)
            if not adapter_path.is_absolute():
                adapter_path = PROJECT_ROOT / adapter_path

            tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                base,
                device_map=mc.get("device_map", "auto"),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()

            _model_cache["model"] = model
            _model_cache["tokenizer"] = tokenizer

        return _model_cache["model"], _model_cache["tokenizer"]

    # ── FIX 3: Correct prompt format ────────────────────────────
    # The model was fine-tuned with LLaMAFactory's "default" (Alpaca) template.
    # That template is:  Human: {instruction}\n{input}<eos>\nAssistant:
    # There is NO "System:" prefix — the old app.py added one which is wrong.
    # The instruction field from training data is used verbatim here.
    # Source of truth: evaluate.py's build_prompt() and INSTRUCTION constant.
    INSTRUCTION = (
        "You are a graph reasoning agent. Given a graph description and a question, "
        "perform these three steps:\n"
        "1. Extract the graph structure as an adjacency list.\n"
        "2. Identify the correct graph algorithm tool to use.\n"
        "3. Extract the parameters required by that tool.\n"
        "When multiple nodes can be visited next in a traversal, always visit the "
        "node with the lowest numerical ID first.\n"
        "Output your answer as a single valid JSON object."
    )

    def build_prompt(tokenizer, user_input: str) -> str:
        eos = tokenizer.eos_token or "<|endoftext|>"
        return f"Human: {INSTRUCTION}\n{user_input}{eos}\nAssistant:"

    def run_inference(question: str) -> str:
        model, tokenizer = get_model()
        ic = config.get("inference", {})

        prompt = build_prompt(tokenizer, question)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        # FIX 4: Use greedy decoding (do_sample=False) — matches evaluate.py.
        # Sampling with temperature can produce NaN logits on long sequences and
        # causes non-deterministic, sometimes garbled JSON.  Greedy is stable.
        # If you need sampling, at minimum set repetition_penalty=1.1 (from YAML).
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=ic.get("max_new_tokens", 8192),
                do_sample=False,               # greedy — deterministic, no NaN
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    def process_query(user_input: str, progress=gr.Progress()):
        """Main pipeline: input → model → parse → tool → results."""
        if not user_input.strip():
            return "❌ Please enter a graph description and question.", "", "", "", ""

        progress(0.1, desc="🤖 Running model inference...")

        try:
            start = time.time()
            raw_output = run_inference(user_input)
            inference_time = time.time() - start
        except Exception as e:
            return f"❌ Model error: {e}", "", "", "", ""

        progress(0.5, desc="📋 Parsing JSON output...")

        parsed = parse_model_output(raw_output)
        if parsed is None:
            return (
                "❌ Model output was not valid JSON.",
                f"```\n{raw_output[:1000]}\n```",
                "", "", ""
            )

        # Format step-by-step outputs
        graph_info = parsed.get("step1_graph_extraction", {})
        adj_list = graph_info.get("adjacency_list", "N/A")
        directed = graph_info.get("directed", False)
        tool_name = parsed.get("step2_tool_name", "N/A")
        params = parsed.get("step3_tool_parameters", {})

        step1_output = f"**Adjacency List:**\n```\n{adj_list}\n```\n**Directed:** {directed}"
        step2_output = f"**Selected Tool:** `{tool_name}`"
        step3_output = f"**Parameters:**\n```json\n{json.dumps(params, indent=2)}\n```"

        progress(0.8, desc="🔧 Executing graph tool...")

        result = execute_parsed_tool(parsed)
        result_str = (
            f"```json\n{json.dumps(result, indent=2)}\n```\n\n"
            f"⏱️ Inference time: {inference_time:.2f}s"
        )

        status = (
            "✅ Pipeline completed successfully!"
            if "error" not in result
            else f"⚠️ Tool error: {result['error']}"
        )

        return status, step1_output, step2_output, step3_output, result_str

    # Build the UI
    with gr.Blocks(
        title="DecompGTI — Graph Reasoning Agent",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
        ),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #6b7280; margin-bottom: 1.5em; }
        .step-box { border-left: 3px solid #6366f1; padding-left: 12px; }
        """
    ) as app:
        gr.Markdown(
            f"# 🧠 DecompGTI — Graph Reasoning Agent\n"
            f"### Decomposed Graph Tool Instruction for Small LLMs\n"
            f"**Active Model:** `{model_name}`",
            elem_classes=["main-title"]
        )
        gr.Markdown(
            "Enter a graph description in natural language and a question. "
            "The model will decompose it into a structured tool call, "
            "and the MCP server will execute the graph algorithm.",
            elem_classes=["subtitle"]
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_box = gr.Textbox(
                    label="📝 Graph Description & Question",
                    placeholder=(
                        "Node 0 is connected to nodes 1 (weight: 3), 2 (weight: 7).\n"
                        "...\n\n"
                        "Question: Calculate the shortest path from node 0 to node 4."
                    ),
                    lines=12,
                )
                with gr.Row():
                    submit_btn = gr.Button("🚀 Solve", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

                gr.Markdown("### 💡 Try an Example")
                for ex in EXAMPLES:
                    gr.Button(ex["name"], size="sm").click(
                        fn=lambda x=ex["input"]: x,
                        outputs=input_box,
                    )

            with gr.Column(scale=1):
                status_box = gr.Markdown(label="Status", value="*Waiting for input...*")

                with gr.Accordion("Step 1: Graph Extraction", open=True):
                    step1_box = gr.Markdown(elem_classes=["step-box"])

                with gr.Accordion("Step 2: Tool Identification", open=True):
                    step2_box = gr.Markdown(elem_classes=["step-box"])

                with gr.Accordion("Step 3: Parameter Extraction", open=True):
                    step3_box = gr.Markdown(elem_classes=["step-box"])

                with gr.Accordion("✅ Final Result", open=True):
                    result_box = gr.Markdown(elem_classes=["step-box"])

        submit_btn.click(
            fn=process_query,
            inputs=input_box,
            outputs=[status_box, step1_box, step2_box, step3_box, result_box],
        )

        clear_btn.click(
            fn=lambda: ("", "*Waiting for input...*", "", "", "", ""),
            outputs=[input_box, status_box, step1_box, step2_box, step3_box, result_box],
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="DecompGTI Web Demo")
    parser.add_argument("--config", type=str, default="configs/qwen2.5_7b.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    app = create_app(str(config_path))
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()