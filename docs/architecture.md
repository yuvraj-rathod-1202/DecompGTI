# DecompGTI Architecture

## System Overview

DecompGTI augments small LLMs (≤8B parameters) with **Decomposed Graph Tool Instruction**.
Instead of asking the LLM to solve graph problems directly, we fine-tune it to produce a
structured 3-step decomposition that an external MCP tool server can execute deterministically.

```
┌──────────────────────────────────────────────────────────────────┐
│                    User Query (Natural Language)                  │
│  "Node 0 is connected to nodes 1 (weight: 3), 2 (weight: 7)..." │
│  "Calculate the shortest path from node 0 to node 4."            │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│              Fine-Tuned Small LLM (Qwen2.5-7B + LoRA)           │
│                                                                   │
│  Outputs structured JSON:                                         │
│  {                                                                │
│    "step1_graph_extraction": { adj_list, directed },              │
│    "step2_tool_name": "shortest_path",                            │
│    "step3_tool_parameters": { source: 0, target: 4 }             │
│  }                                                                │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     JSON Parser + Validator                       │
│  - Extracts first valid JSON object from model output            │
│  - Converts adjacency list string → edge list                    │
│  - Validates parameters against tool schema                      │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│              MCP Graph Tool Server (NetworkX)                     │
│                                                                   │
│  Tools: dijkstra_shortest_path, bfs, dfs, maximum_flow,          │
│         topological_sort, cycle_detection, minimum_spanning_tree, │
│         bipartite_maximum_matching                                │
│                                                                   │
│  ✅ Deterministic execution — guaranteed correct answers          │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Final Answer                               │
│  { "path": [0, 1, 2, 4], "distance": 9 }                        │
└──────────────────────────────────────────────────────────────────┘
```

## Key Insight

The LLM **never solves** the graph problem. It only:
1. **Parses** the natural language into a structured graph representation
2. **Classifies** the task into the correct algorithm
3. **Extracts** the parameters

All computation is offloaded to deterministic algorithms. This means:
- ✅ **100% correct answers** (if the decomposition is correct)
- ✅ **Scales to any graph size** (the tool handles computation)
- ✅ **Works on small models** (7B, 2B) that can't do math internally

## File Structure

```
src/decompgti/mcp_server/
├── graph_tools.py    # 8 graph algorithms (NetworkX wrappers)
├── pipeline.py       # execute_from_model_output()
├── routing.py        # Model output → tool call mapping
├── schemas.py        # Pydantic validation models
├── server.py         # MCP server entry point
└── tool_catalog.py   # Tool metadata for prompting

scripts/
├── inference_e2e.py        # Full pipeline inference
├── evaluate.py             # Benchmarking script
├── compare_baselines.py    # GPT-4o / Claude comparison
└── demo_roundtrip.py       # Tool execution demo

evaluation/
├── metrics.py        # 5 evaluation metrics
└── benchmarks.py     # Test set generation

web/
└── app.py            # Gradio web demo

configs/
├── qwen2.5_7b.yaml   # Fixed inference params for 7B model
├── qwen2.5_1.5b.yaml # Fixed inference params for 1.5B model
└── llama3_8b.yaml     # Fixed inference params for LLaMA-3
```
