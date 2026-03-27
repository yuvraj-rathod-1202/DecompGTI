# Data Generation Guide

This document describes how to generate graph-task datasets for the DecompGTI project using the GraphInstruct framework.

## Overview

Data generation uses the **GraphInstruct** dataset generation pipeline to create training and evaluation data for 10 core graph algorithms covered in the project proposal. The generated datasets are used for:

- **Fine-tuning** small LLMs (< 8B parameters) to extract graph structures and route tool calls
- **Evaluation** of tool identification accuracy and parameter extraction correctness
- **Benchmarking** against baseline models across varying graph sizes

## Included Graph Tasks

The data generation pipeline produces datasets for the following algorithms (as specified in the project proposal):

| Algorithm | Category | Use Case |
|-----------|----------|----------|
| **Shortest Path (Dijkstra)** | Traversal & Paths | Find optimal routes in weighted graphs |
| **Breadth-First Search (BFS)** | Traversal & Paths | Level-based graph exploration |
| **Depth-First Search (DFS)** | Traversal & Paths | Exhaustive graph traversal |
| **Connectivity** | Structural Analysis | Check global graph connectivity |
| **Cycle Detection** | Structural Analysis | Identify loops in directed graphs |
| **Topological Sort** | Structural Analysis | Order nodes in DAGs respecting dependencies |
| **Bipartite Matching** | Structural Analysis | Maximum matching in bipartite graphs |
| **Minimum Spanning Tree (MST)** | Optimization | Find minimum-weight spanning trees |
| **Maximum Flow** | Network Flow | Route maximum flow through a network |
| **Connected Component** | Structural Analysis | Identify connected subgraphs |

## Quick Start

### Prerequisites

All dependencies are installed via the root `pyproject.toml`:

```bash
uv sync
```

This installs:
- `networkx>=3.4.2` (graph algorithms)
- `pandas>=2.1.3` (data handling)
- `numpy>=1.21.5` (numerical operations)
- `PyYAML>=6.0` (configuration files)
- `tqdm>=4.63.0` (progress tracking)

### Generate Data

1. **Update the generation script path:**

   Edit [GraphInstruct/script/run_all_generation.sh](GraphInstruct/script/run_all_generation.sh) and set `project_root` to your local path:

   ```bash
   project_root=/path/to/your/DecompGTI/GraphInstruct
   ```

2. **Run the generation script:**

   ```bash
   bash GraphInstruct/script/run_all_generation.sh
   ```

   This generates two dataset sizes:
   - **mini**: 5-7 nodes, 300 samples per task (600 samples total)

   - **small**: 8-15 nodes, 300 samples per task (600 samples total)

3. **Output location:**

   Generated datasets are saved to:
   ```
   GraphInstruct/data/dataset/mini/      # mini dataset
   GraphInstruct/data/dataset/small/    # small dataset
   ```

## Dataset Structure

Each task generates a CSV file with the following format:

```
id,input,output
0,"Graph: {adjacency list in natural language}","Expected output for the algorithm"
1,"Graph: ...","..."
...
```

### Example Output

For shortest path on a mini graph:
```
id,input,output
12,"Nodes: 1,2,3,4,5. Edges: 1->2(5), 2->3(2), 1->4(1), 4->3(4)","Shortest path from 1 to 3: 1->4->3 (cost: 5)"
```

## Customization

### Modify Graph Sizes

Edit [GraphInstruct/script/run_all_generation.sh](GraphInstruct/script/run_all_generation.sh):

```bash
# For mini dataset
num_nodes_range="(5,7)"      # Change to (X,Y)
num_sample=300               # Change sample count

# For small dataset
num_nodes_range="(8,15)"     # Change to (X,Y)
num_sample=300               # Change sample count
```

### Add/Remove Tasks

The script currently generates only the 10 proposed tasks. To modify which tasks are generated, uncomment/comment lines in the script:

```bash
bash $script_root/shortest_path.sh ...  # Uncommented: will run
# bash $script_root/page_rank.sh ...    # Commented: will not run
```

Excluded tasks (commented out) include: `page_rank`, `degree`, `common_neighbor`, `jaccard`, `edge`, `neighbor`, `predecessor`, `clustering_coefficient`, `diameter`, `euler_path`, `hamiltonian_path`.

## Usage in Training Pipeline

Once generated, datasets are used with **LLaMAFactory** for fine-tuning:

1. Reformat generated CSVs into LLaMA instruction format
2. Place reformatted data in `GraphInstruct/LLaMAFactory/data/reasoning/`
3. Configure training via `examples/train_reasoning/llama3_lora_sft.yaml`
4. Run training: `bash GraphInstruct/LLaMAFactory/run.sh`

See [GraphInstruct/README.md](GraphInstruct/README.md) for detailed training instructions.

## Evaluation

Generated datasets can be evaluated using:

```bash
bash GraphInstruct/script/run_all_evaluation.sh
```

Evaluation requires:
- A CSV file with model outputs (columns: `id`, `output`)
- The original generated dataset for reference

## References

- **GraphInstruct Paper**: https://arxiv.org/abs/2403.04483
- **GraphInstruct Repository**: https://github.com/CGCL-codes/GraphInstruct
- **NetworkX Documentation**: https://networkx.org/

## Troubleshooting

**Issue**: Script fails with "directory not found"
- **Solution**: Verify `project_root` path in the script is correct

**Issue**: Generation is slow
- **Solution**: Reduce `num_sample` or `num_nodes_range` for testing

**Issue**: Out of memory errors
- **Solution**: Run generation on smaller subsets or increase system RAM
