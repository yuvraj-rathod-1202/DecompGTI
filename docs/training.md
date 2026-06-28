# Training Documentation

## How the Models Were Trained

### Data Generation Pipeline

Training data was generated using the GraphInstruct framework (GTG module):

1. **Graph Generation**: Random graphs (Erdős–Rényi, Barabási–Albert, Watts–Strogatz) of varying sizes
2. **Task Generation**: 10 graph reasoning tasks with natural language questions
3. **CSV Output**: Raw data saved as CSVs with graph descriptions, questions, and answers
4. **JSON Conversion**: `csv2json.py` converts CSVs to Alpaca-format JSON for LLaMAFactory

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total samples | 15,000 |
| Tasks | 10 (shortest_path, BFS, DFS, connectivity, topological_sort, cycle, bipartite, MST, maximum_flow, connected_components) |
| Graph sizes | Mini (5-7 nodes), Small (8-15), Medium (16-25) |
| Samples per task per size | 500 |
| Format | Alpaca (instruction, input, output, system) |
| Node labeling | Integer IDs |
| Graph types | Mixed directed/undirected, weighted |

### Training Configuration

All models were trained using **LLaMAFactory** with the following settings:

| Parameter | Value |
|-----------|-------|
| Fine-tuning method | LoRA |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Epochs | 3 |
| Batch size | 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 |
| Learning rate | 5e-5 |
| LR scheduler | Cosine |
| Optimizer | AdamW |
| Precision | bf16 |
| Cutoff length | 2048 |
| Max samples | 100,000 |

### Models Trained

| Model | Base | Parameters | Training Status |
|-------|------|-----------|-----------------|
| DecompGTI-7B | Qwen/Qwen2.5-7B | 7B | ✅ Trained |
| DecompGTI-LLaMA-8B | meta-llama/Meta-Llama-3-8B-Instruct | 8B | ✅ Trained |
| DecompGTI-1.5B | Qwen/Qwen2.5-1.5B | 1.5B | 🔜 Training |

### Reproducing Training

To reproduce training, you need:
1. Install [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory)
2. Place the training data at `data/graph_reasoning.json`
3. Use the config from `training/training_config.yaml`

```bash
# Inside LLaMAFactory directory:
llamafactory-cli train training_config.yaml
```
