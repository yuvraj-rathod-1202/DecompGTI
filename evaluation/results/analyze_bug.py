import json

def analyze_failures(prediction_file):
    with open(prediction_file, "r") as f:
        preds = json.load(f)
    
    total_samples = len(preds)
    
    # We want to find cases where the LLM correctly identified the tool
    # and perfectly extracted the parameters, but the Task Success was FALSE.
    # This indicates that either:
    # 1. The slight missing edges in the Adjacency List changed the mathematical answer.
    # 2. There are multiple valid answers (e.g., multiple shortest paths) and Python picked a different one than the dataset.
    
    failures_with_perfect_params = []
    
    for sample in preds:
        metric = sample.get("metrics", {})
        
        # If task failed
        if not metric.get("task_success", False):
            # Did it get the right tool and params?
            tool_correct = metric.get("tool_correct", False)
            param_f1 = metric.get("params_f1", 0)
            
            # Since params_f1 isn't saved directly in preds, wait, let's just check the logged data
            # Evaluate.py EvalResult fields are usually dumped.
            # Let's see what keys are printed
            if tool_correct and metric.get("params_recall", 0) == 1.0 and metric.get("params_precision", 0) == 1.0:
                failures_with_perfect_params.append({
                    "task": metric.get("task_type", "Unknown"),
                    "expected_ans": sample.get("expected_answer"),
                    "actual_ans": sample.get("actual_answer", "N/A"),
                    "adj_f1": metric.get("adj_edge_f1", 0)
                })

    print(f"=== Analysis for {prediction_file} ===")
    print(f"Total Failed Tasks out of {total_samples}: {len([s for s in preds if not s.get('metrics',{}).get('task_success')])}")
    print(f"Failures where Tool & Parameters were 100% PERFECT: {len(failures_with_perfect_params)}")
    
    if failures_with_perfect_params:
        print("Breakdown of these 'Phantom Failures':")
        for f in failures_with_perfect_params[:5]:
            print(f"  Task: {f['task']} | Adj F1: {f['adj_f1']:.3f} | Expected: {f['expected_ans']} | Got: {f['actual_ans']}")
    print("\n")


analyze_failures("/home/ramji.purwar/DecompGTI/DecompGTI/evaluation/results/qwen7b_v3/mini_predictions.json")
analyze_failures("/home/ramji.purwar/DecompGTI/DecompGTI/evaluation/results/qwen7b_v3/medium_predictions.json")
