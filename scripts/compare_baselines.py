"""
DecompGTI Baseline Comparison.

Compares DecompGTI fine-tuned models against:
  1. Base Qwen2.5-7B (no fine-tuning, zero-shot)
  2. GPT-4o (via OpenAI API)
  3. Claude Sonnet (via Anthropic API)

Usage:
    # Compare against base model (no API key needed):
    python scripts/compare_baselines.py \
        --test-set data/test_set_small.json \
        --baseline base_qwen \
        --output evaluation/results/baseline_base_qwen.json

    # Compare against GPT-4o (needs OPENAI_API_KEY env var):
    python scripts/compare_baselines.py \
        --test-set data/test_set_small.json \
        --baseline gpt4o \
        --output evaluation/results/baseline_gpt4o.json

    # Compare against Claude (needs ANTHROPIC_API_KEY env var):
    python scripts/compare_baselines.py \
        --test-set data/test_set_small.json \
        --baseline claude \
        --output evaluation/results/baseline_claude.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    EvalResult,
    aggregate_results,
    check_json_validity,
    check_tool_accuracy,
    check_parameter_extraction,
    check_adjacency_extraction,
)
from scripts.evaluate import execute_tool, check_task_success

SYSTEM_PROMPT = (
    "You are a graph reasoning agent. Given a graph description and a question, "
    "perform these three steps:\n"
    "1. Extract the graph structure as an adjacency list.\n"
    "2. Identify the correct graph algorithm tool to use.\n"
    "3. Extract the parameters required by that tool.\n"
    "When multiple nodes can be visited next in a traversal, always visit the node with the lowest numerical ID first.\n"
    "Output your answer as a single valid JSON object with keys: "
    "step1_graph_extraction, step2_tool_name, step3_tool_parameters.\n\n"
    "The adjacency list should be formatted as a string like: "
    '"{0: [(1, weight:3), (2, weight:7)], 1: [(0, weight:3)]}"\n'
    "step1_graph_extraction should have keys: adjacency_list (string), directed (boolean)."
)


# ── Baseline: Base Qwen (no fine-tuning) ────────────────────────

def run_base_qwen(test_set: list[dict], model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> list[dict]:
    """Run base Qwen model without any fine-tuning."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    model.eval()

    predictions = []
    for i, sample in enumerate(test_set):
        print(f"  Base model: {i + 1}/{len(test_set)}", end="\r")
        input_text = sample["graph_nl"] + "\n\nQuestion: " + sample["question"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        start = time.time()
        outputs = model.generate(
            **inputs, max_new_tokens=4096, temperature=0.1, top_p=0.7,
            do_sample=True, repetition_penalty=1.1,
        )
        elapsed = time.time() - start

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)

        predictions.append({
            "sample_id": sample.get("sample_id", i),
            "raw_output": raw,
            "inference_time_s": round(elapsed, 2),
        })

    print()
    return predictions


# ── Baseline: Base Llama 3 ────────────────────────

def run_base_llama(test_set: list[dict], model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> list[dict]:
    """Run base Llama 3 model without any fine-tuning."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    print(f"Loading base model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    model.eval()

    predictions = []
    for i, sample in enumerate(test_set):
        print(f"  Base Llama 3: {i + 1}/{len(test_set)}", end="\r")
        input_text = sample["graph_nl"] + "\n\nQuestion: " + sample["question"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        start = time.time()
        outputs = model.generate(
            **inputs, max_new_tokens=4096, temperature=0.1, top_p=0.7,
            do_sample=True, repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
        elapsed = time.time() - start

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)

        predictions.append({
            "sample_id": sample.get("sample_id", i),
            "raw_output": raw,
            "inference_time_s": round(elapsed, 2),
        })

    print()
    return predictions


# ── Baseline: GPT-4o ────────────────────────────────────────────

def run_gpt4o(test_set: list[dict]) -> list[dict]:
    """Run GPT-4o via OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    predictions = []
    for i, sample in enumerate(test_set):
        print(f"  GPT-4o: {i + 1}/{len(test_set)}", end="\r")
        input_text = sample["graph_nl"] + "\n\nQuestion: " + sample["question"]

        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        elapsed = time.time() - start
        raw = response.choices[0].message.content

        predictions.append({
            "sample_id": sample.get("sample_id", i),
            "raw_output": raw,
            "inference_time_s": round(elapsed, 2),
        })

        # Rate limiting
        time.sleep(0.5)

    print()
    return predictions


# ── Baseline: Groq (Free LLaMA-3-70B API) ──────────────────────

# ── Baseline: Groq (Free LLaMA-3-70B API) ──────────────────────

def run_groq_llama70b(test_set: list[dict]) -> list[dict]:
    """Run LLaMA-3-70B via the free Groq API."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY environment variable (Get a free key at https://console.groq.com)")
        sys.exit(1)

    # Groq provides an OpenAI-compatible API client!
    try:
        from openai import OpenAI
    except ImportError:
        print("Please run: pip install openai")
        sys.exit(1)
        
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    predictions = []
    import time
    for i, sample in enumerate(test_set):
        # 1. Removed end="\r" so we don't overwrite the line
        print(f"==== Groq LLaMA: Sample {i + 1}/{len(test_set)} ====", flush=True)
        input_text = sample["graph_nl"] + "\n\nQuestion: " + sample["question"]

        start = time.time()
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", # 2. Updated to a working model ID
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content
            
            # 3. Print the actual text the model generated
            print(f"Response:\n{raw}\n")
            
        except Exception as e:
            raw = f"Error: {e}"
            # 4. Print the error immediately if the API fails
            print(f"API ERROR:\n{raw}\n")
            
        elapsed = time.time() - start

        predictions.append({
            "sample_id": sample.get("sample_id", i),
            "raw_output": raw,
            "inference_time_s": round(elapsed, 2),
        })

        # Respect Groq free tier rate limits (30 requests/minute max)
        time.sleep(2)

    print()
    return predictions

# ── Baseline: Claude Sonnet ─────────────────────────────────────

def run_claude(test_set: list[dict]) -> list[dict]:
    """Run Claude Sonnet via Anthropic API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    predictions = []
    for i, sample in enumerate(test_set):
        print(f"  Claude: {i + 1}/{len(test_set)}", end="\r")
        input_text = sample["graph_nl"] + "\n\nQuestion: " + sample["question"]

        start = time.time()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": input_text}],
        )
        elapsed = time.time() - start
        raw = message.content[0].text

        predictions.append({
            "sample_id": sample.get("sample_id", i),
            "raw_output": raw,
            "inference_time_s": round(elapsed, 2),
        })

        time.sleep(0.5)

    print()
    return predictions


# ── Evaluate predictions ────────────────────────────────────────

def evaluate_predictions(test_set: list[dict], predictions: list[dict]) -> list[EvalResult]:
    """Evaluate a set of predictions against the test set."""
    results = []
    for i, (sample, pred) in enumerate(zip(test_set, predictions)):
        raw_output = pred.get("raw_output", "")
        result = EvalResult()
        result.sample_id = sample.get("sample_id", i)
        result.task_type = sample.get("task_type", "")
        result.graph_size = sample.get("graph_size", "")
        result.num_nodes = sample.get("num_nodes", 0)

        result.json_valid, parsed = check_json_validity(raw_output)
        if result.json_valid and parsed:
            result.predicted_tool = parsed.get("step2_tool_name", "")
            result.expected_tool = sample.get("expected_tool", "")
            result.tool_correct = check_tool_accuracy(result.predicted_tool, result.expected_tool)

            result.predicted_params = parsed.get("step3_tool_parameters", {})
            result.expected_params = sample.get("expected_params", {})
            result.params_precision, result.params_recall = check_parameter_extraction(
                result.predicted_params, result.expected_params
            )

            graph_info = parsed.get("step1_graph_extraction", {})
            result.adj_edge_f1 = check_adjacency_extraction(
                graph_info.get("adjacency_list", ""),
                sample.get("graph_adj", ""),
                directed=sample.get("directed", False),
            )

            try:
                actual_answer = execute_tool(parsed)
                result.task_success = check_task_success(actual_answer, sample.get("expected_answer"))
            except Exception:
                result.task_success = False
        else:
            result.error_message = "Invalid JSON"

        results.append(result)

    return results


# ── Main ────────────────────────────────────────────────────────

BASELINES = {
    "base_qwen": run_base_qwen,
    "base_llama": run_base_llama,
    "groq_llama70b": run_groq_llama70b,
    "gpt4o": run_gpt4o,
    "claude": run_claude,
}


def main():
    parser = argparse.ArgumentParser(description="DecompGTI Baseline Comparison")
    parser.add_argument("--test-set", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True, choices=list(BASELINES.keys()))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for API cost control)")
    args = parser.parse_args()

    with open(args.test_set, "r") as f:
        test_set = json.load(f)

    if args.max_samples:
        test_set = test_set[:args.max_samples]

    print(f"Running baseline: {args.baseline} on {len(test_set)} samples...")

    runner = BASELINES[args.baseline]
    predictions = runner(test_set)

    results = evaluate_predictions(test_set, predictions)
    metrics = aggregate_results(results)

    print(f"\n{'='*55}")
    print(f"  Baseline: {args.baseline}")
    print(f"{'='*55}")
    metrics.print_report()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "baseline": args.baseline,
            "test_set": args.test_set,
            "num_samples": len(test_set),
            "metrics": metrics.to_dict(),
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        pred_path = output_path.with_name(output_path.stem + "_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
