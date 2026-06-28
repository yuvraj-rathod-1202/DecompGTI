# Training Reference

This folder contains **slim reference copies** of the training configurations and data format.
The actual training infrastructure (LLaMAFactory, GTG data generator) lives in `GraphInstruct/` 
and is NOT included in the GitHub repository due to size (11 GB+).

## Files

| File | Purpose |
|------|---------|
| `training_config.yaml` | Exact LLaMAFactory config used for training (copied from saves/) |
| `csv2json.py` | Script that converts generated CSVs → Alpaca-format JSON (copied from GraphInstruct/script/) |
| `data_format_example.json` | 5 example training samples showing the exact data format |

## How to Reproduce Training

### Prerequisites
- [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory) installed
- CUDA-compatible GPU (tested on NVIDIA A100/V100)
- The full training dataset (`graph_reasoning.json`, ~30 MB, 15K samples)

### Steps

1. **Generate training data** (if you have the GTG pipeline):
   ```bash
   cd GraphInstruct
   python script/generation.py   # Generates CSVs
   python script/csv2json.py     # Converts to Alpaca JSON
   ```

2. **Place data in LLaMAFactory**:
   ```bash
   cp graph_reasoning.json /path/to/LLaMAFactory/data/
   ```

3. **Train**:
   ```bash
   cd /path/to/LLaMAFactory
   llamafactory-cli train /path/to/training_config.yaml
   ```

### Key Hyperparameters

| Setting | Value | Rationale |
|---------|-------|-----------|
| LoRA rank 8 | Sufficient for structured output tasks |
| LR 5e-5 | Standard for LoRA SFT |
| 3 epochs | Enough for 15K samples without overfitting |
| Batch 2 × accum 8 = 16 | Balances memory and convergence |
| Cutoff 2048 | Covers even medium graph descriptions |
