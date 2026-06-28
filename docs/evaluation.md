# Evaluation Methodology

## Metrics

DecompGTI is evaluated on 5 metrics that decompose the pipeline's correctness:

### 1. JSON Validity Rate
**What:** Percentage of model outputs that are parseable as valid JSON with the correct schema.

**How:** Extract the first complete JSON object from the raw model output. Check that it contains the three required keys: `step1_graph_extraction`, `step2_tool_name`, `step3_tool_parameters`.

### 2. Tool Identification Accuracy
**What:** Exact match rate for routing to the correct graph algorithm.

**How:** Compare the model's `step2_tool_name` against the ground truth after normalizing aliases (e.g., "dijkstra" → "shortest_path").

### 3. Parameter Extraction Score
**What:** Precision, Recall, and F1 for the extracted tool parameters.

**How:** Compare each key-value pair in `step3_tool_parameters` against ground truth. Handles type coercion (string "3" matches integer 3).

### 4. Adjacency Extraction F1
**What:** Edge-level F1 score comparing the extracted graph structure to the true graph.

**How:** Parse both predicted and expected adjacency list strings into sets of (u, v, weight) tuples. Compute precision, recall, and F1 at the edge level.

### 5. Task Success Rate
**What:** End-to-end exact match accuracy — does the full pipeline produce the correct final answer?

**How:** Execute the MCP tool using the model's extracted graph, tool, and parameters. Compare the tool's output against the pre-computed ground truth answer.

## Test Set Design

Test sets are generated programmatically with known ground-truth answers:

| Size Category | Nodes | Purpose |
|---------------|-------|---------|
| Mini | 5-7 | Baseline competency |
| Small | 8-15 | Standard difficulty |
| Medium | 16-25 | Context window stress |
| Large | 26-50 | Scalability evaluation |

Each test set contains 50 samples per task per size, covering all 8 supported graph algorithms.

## Running the Evaluation

```bash
# Step 1: Generate test sets
python -m evaluation.benchmarks --output data/ --samples-per-task 50

# Step 2: Evaluate a model
python scripts/evaluate.py \
    --base-model Qwen/Qwen2.5-7B \
    --adapter path/to/lora/adapter \
    --test-set data/test_set_small.json \
    --output evaluation/results/qwen7b_small.json

# Step 3: Compare with baselines
python scripts/compare_baselines.py \
    --test-set data/test_set_small.json \
    --baseline base_qwen \
    --output evaluation/results/baseline_base_qwen.json
```
