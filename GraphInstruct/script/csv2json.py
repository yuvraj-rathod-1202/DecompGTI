"""
Converts generated CSV datasets into LLaMAFactory Alpaca-format JSON.
Reads from: C:\DecompGTI\DecompGTI\data\{mini,small,medium}\{task}\{task}-int_id.csv
Writes to:  GraphInstruct\LLaMAFactory\data\decompgti_3step.json
Also registers the dataset in dataset_info.json.
"""
import os
import re
import json
import pandas as pd
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent                          # GraphInstruct/
DATA_ROOT = PROJECT_ROOT.parent / "data"                  # DecompGTI/data/
LLAMA_DATA_DIR = PROJECT_ROOT / "LLaMAFactory" / "data"   # GraphInstruct/LLaMAFactory/data/

TASKS = [
    'shortest_path', 'DFS', 'BFS', 'connectivity', 'topological_sort',
    'cycle', 'bipartite', 'MST', 'maximum_flow', 'connected_component'
]
SPLITS = ['mini', 'small', 'medium']

# Maps task names to the MCP tool names the model should learn to call
TOOL_NAME_MAP = {
    'shortest_path': 'shortest_path',
    'DFS': 'depth_first_search',
    'BFS': 'breadth_first_search',
    'connectivity': 'check_connectivity',
    'topological_sort': 'topological_sort',
    'cycle': 'detect_cycle',
    'bipartite': 'check_bipartite',
    'MST': 'minimum_spanning_tree',
    'maximum_flow': 'maximum_flow',
    'connected_component': 'find_connected_components'
}


def extract_parameters(task_name, question_text):
    """Extract tool parameters from the natural language question."""
    params = {}
    q = question_text.lower() if isinstance(question_text, str) else ""

    # Tasks with source and target: "from node X to node Y"
    if task_name in ['shortest_path', 'DFS', 'BFS', 'maximum_flow']:
        match = re.search(r"from node (\w+) to node (\w+)", q)
        if match:
            params['source'] = match.group(1)
            params['target'] = match.group(2)
            # Try to convert to int if possible
            try:
                params['source'] = int(params['source'])
                params['target'] = int(params['target'])
            except ValueError:
                pass

    # Cycle detection: "starting from node X"
    elif task_name == 'cycle':
        match = re.search(r"start(?:ing)? from node (\w+)", q)
        if match:
            try:
                params['source'] = int(match.group(1))
            except ValueError:
                params['source'] = match.group(1)

    # No specific parameters needed for: connectivity, topological_sort, bipartite, MST, connected_component
    return params


def main():
    LLAMA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_data = []
    stats = {}

    for split in SPLITS:
        stats[split] = 0
        for task in TASKS:
            csv_path = DATA_ROOT / split / task / f"{task}-int_id.csv"
            if not csv_path.exists():
                print(f"  WARNING: Missing {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            count = 0
            for _, row in df.iterrows():
                adj_str = str(row.get('graph_adj', ''))
                graph_nl = str(row.get('graph_nl', ''))
                question = str(row.get('question', ''))
                directed = bool(row.get('directed', False))
                tool_name = TOOL_NAME_MAP.get(task, task)
                params = extract_parameters(task, question)

                # Build the 3-step decomposed target output
                target_output = json.dumps({
                    "step1_graph_extraction": {
                        "adjacency_list": adj_str,
                        "directed": directed
                    },
                    "step2_tool_name": tool_name,
                    "step3_tool_parameters": params
                }, ensure_ascii=False)

                all_data.append({
                    "instruction": (
                        "You are a graph reasoning agent. Given a graph description and a question, "
                        "perform these three steps:\n"
                        "1. Extract the graph structure as an adjacency list.\n"
                        "2. Identify the correct graph algorithm tool to use.\n"
                        "3. Extract the parameters required by that tool.\n"
                        "Output your answer as a single valid JSON object."
                    ),
                    "input": graph_nl.strip() + "\n\nQuestion: " + question.strip(),
                    "output": target_output,
                    "system": (
                        "You are DecompGTI, an intelligent agent that decomposes graph reasoning problems "
                        "into structured tool calls for an MCP server. Always output strictly valid JSON."
                    )
                })
                count += 1

            stats[split] += count
            print(f"  {split}/{task}: {count} samples")

    # Write the training JSON
    output_json = LLAMA_DATA_DIR / "graph_reasoning.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Total samples: {len(all_data)}")
    for split, count in stats.items():
        print(f"  {split}: {count}")
    print(f"Saved to: {output_json}")

    # Register in dataset_info.json
    info_path = LLAMA_DATA_DIR / "dataset_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    info["graph_reasoning"] = {
        "file_name": "graph_reasoning.json"
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Registered 'decompgti_3step' in {info_path}")


if __name__ == "__main__":
    main()
