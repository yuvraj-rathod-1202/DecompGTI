#!/bin/bash
# Script to run Groq LLaMA-3-70B baseline evaluation locally on CPU over API

# 1. Activate the python environment
source /home/ramji.purwar/miniconda3/bin/activate llama_env

# 2. Install the openai pip package quietly if missing (Groq uses the OpenAI wrapper)
pip install openai -q

# 3. Export the API key
export GROQ_API_KEY=""

# 4. Run the baseline comparison on the mini dataset
echo "Starting Groq LLaMA-3-70B baseline evaluation..."
python scripts/compare_baselines.py \
    --test-set data/test_set_mini.json \
    --baseline groq_llama70b \
    --output evaluation/results/baseline_groq_mini.json
