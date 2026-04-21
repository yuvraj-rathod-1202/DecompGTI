# 🧠 DecompGTI

**Augmenting Small LLMs with Decomposed Graph Tool Instruction**

DecompGTI enables small language models (≤8B parameters) to achieve near-perfect accuracy on graph reasoning tasks by decomposing problems into structured tool calls executed by a deterministic MCP server.

> **Key Insight:** The LLM never solves graph problems directly. It only *parses, classifies, and extracts* — all computation is offloaded to verified graph algorithms.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Query (Natural Language)                               │
│  "Node 0 connects to Node 1 (weight: 3)..."                │
│  "Find the shortest path from node 0 to node 4."           │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  Fine-Tuned LLM (Qwen2.5-7B / 1.5B + LoRA)                │
│  Step 1: Extract graph → adjacency list                     │
│  Step 2: Identify tool → "shortest_path"                    │
│  Step 3: Extract params → {source: 0, target: 4}           │
│  Output: Single valid JSON object                            │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  MCP Graph Tool Server (NetworkX)                            │
│  Deterministic execution → guaranteed correct answers       │
│  Tools: Dijkstra, BFS, DFS, MaxFlow, MST, Topo Sort, ...   │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
           Final Answer ✅
```

## 📊 Results

| Model | Params | JSON Valid | Tool Accuracy | Param F1 | Task Success |
|-------|--------|-----------|---------------|----------|-------------|
| GPT-4o (zero-shot) | ~200B | — | — | — | — |
| Qwen2.5-7B (base) | 7B | — | — | — | — |
| **DecompGTI-7B** | 7B | — | — | — | — |
| **DecompGTI-LLaMA-8B** | 8B | — | — | — | — |
| **DecompGTI-1.5B** | 1.5B | — | — | — | — |

> *Run `python scripts/evaluate.py` to fill in these numbers. See [docs/evaluation.md](docs/evaluation.md).*

## 🚀 Quick Start

### Setup

```bash
# Clone and install
git clone https://github.com/deepbuha/DecompGTI.git
cd DecompGTI
pip install -e .
```

### Run the Web Demo

```bash
pip install gradio pyyaml
python web/app.py --config configs/qwen2.5_7b.yaml
# Open http://localhost:7860
```

### Run Evaluation

```bash
# Generate test sets
python -m evaluation.benchmarks --output data/ --samples-per-task 50

# Evaluate a model
python scripts/evaluate.py \
    --base-model Qwen/Qwen2.5-7B \
    --adapter path/to/lora/adapter \
    --test-set data/test_set_small.json \
    --output evaluation/results/qwen7b_small.json

# Compare with baselines
python scripts/compare_baselines.py \
    --test-set data/test_set_small.json \
    --baseline base_qwen \
    --output evaluation/results/baseline.json
```

### Run End-to-End Inference

```bash
python scripts/inference_e2e.py \
    --base-model Qwen/Qwen2.5-7B \
    --adapter path/to/lora/adapter \
    --question "Node 0 is connected to nodes 1 (weight: 3), 2 (weight: 7). ..."
```

## 📁 Project Structure

```
DecompGTI/
├── src/decompgti/mcp_server/   # MCP tool server (8 graph algorithms)
├── scripts/                     # Inference, evaluation, comparison scripts
├── evaluation/                  # Metrics (5 metrics) + test set generator
├── web/                         # Gradio web demo
├── configs/                     # Fixed inference params per model
├── data/                        # Generated test sets
├── docs/                        # Architecture, training, evaluation docs
├── training/                    # Training config reference (slim)
├── tests/                       # Unit tests
└── proposal/                    # Project proposal (LaTeX + PDF)
```

## 🔧 MCP Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `dijkstra_shortest_path` | source, target | Shortest weighted path |
| `bfs` | source | Breadth-first search traversal |
| `dfs` | source | Depth-first search traversal |
| `maximum_flow` | source, target | Maximum network flow |
| `minimum_spanning_tree` | — | Kruskal's/Prim's MST |
| `topological_sort` | — | DAG topological ordering |
| `cycle_detection` | — | Cycle existence check |
| `bipartite_maximum_matching` | — | Bipartite matching |

## 📈 Evaluation Metrics

1. **JSON Validity Rate** — Can the model produce parseable JSON?
2. **Tool Identification Accuracy** — Does it pick the right algorithm?
3. **Parameter Extraction F1** — Are source/target/params correct?
4. **Adjacency Extraction F1** — Is the graph structure correctly extracted?
5. **Task Success Rate** — Does the full pipeline give the correct answer?

See [docs/evaluation.md](docs/evaluation.md) for methodology details.

## 📚 Documentation

- [Architecture](docs/architecture.md) — Pipeline design and file reference
- [Training](docs/training.md) — How models were trained (hyperparameters, data stats)
- [Evaluation](docs/evaluation.md) — Metrics, test sets, and how to run benchmarks

## 📄 Citation

```bibtex
@article{graphinstruct,
    title={GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability},
    author={Luo, Zihan and Song, Xiran and Huang, Hong and Lian, Jianxun and Zhang, Chenhao and Jiang, Jinqi and Xie, Xing},
    journal={CoRR},
    volume={abs/2403.04483},
    year={2024}
}

@inproceedings{zheng-etal-2024-llamafactory,
    title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
    author={Zheng, Yaowei and Zhang, Richong and Zhang, Junhao and Ye, Yanhan and Luo, Zheyan and Feng, Zhangchi and Ma, Yongqiang},
    booktitle={Proceedings of the 62nd ACL (System Demonstrations)},
    year={2024}
}
```

## 👥 Team

Arpna Gupta, Buha Deep, Rathod Yuvraj, Solanki Viraj — IIT Gandhinagar

## 📝 License

MIT License — see [LICENSE](LICENSE)