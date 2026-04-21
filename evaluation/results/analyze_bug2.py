import sys
import os
import json
sys.path.append(os.path.abspath("/home/ramji.purwar/DecompGTI/DecompGTI"))

from evaluation.metrics import evaluate_single_sample
from scripts.evaluate import execute_tool
import evaluation.metrics as metrics

def debug_eval(test_set_path, predictions_path):
    with open(test_set_path) as f:
        test_set = json.load(f)
    with open(predictions_path) as f:
        preds = json.load(f)

    print(f"=== Debugging {os.path.basename(test_set_path)} ===")
    
    failures = []
    
    for i, (sample, pred) in enumerate(zip(test_set, preds)):
        raw = pred["raw_output"]
        
        # We need to manually do what compare_baselines does
        expected_tool = sample.get("expected_tool_name", sample["task_type"])
        expected_params = sample.get("expected_parameters", {})
        expected_adj = sample["graph_adj"]
        expected_directed = sample["directed"]
        expected_answer = sample.get("expected_answer")
        
        # 1. Parse JSON
        is_valid, parsed = metrics.check_json_validity(raw)
        if not is_valid:
            continue
            
        # 2. Extract things
        pred_tool = parsed.get("step2_tool_name", "")
        pred_params = parsed.get("step3_tool_parameters", {})
        
        tool_correct = metrics.check_tool_accuracy(pred_tool, expected_tool)
        p_prec, p_rec = metrics.check_parameter_extraction(pred_params, expected_params)
        param_perfect = (p_prec == 1.0 and p_rec == 1.0)
        
        # 3. Simulate execution
        actual_answer = execute_tool(parsed)
        
        # 4. Check success
        success = metrics.check_task_success(actual_answer, expected_answer)
        
        if tool_correct and param_perfect and not success:
            graph_info = parsed.get("step1_graph_extraction", {})
            pred_adj = graph_info.get("adjacency_list", "")
            adj_f1 = metrics.check_adjacency_extraction(pred_adj, expected_adj, expected_directed)
            
            failures.append({
                "id": sample.get("sample_id", i),
                "task": sample["task_type"],
                "expected": expected_answer,
                "actual": actual_answer,
                "adj_f1": adj_f1
            })
            
    print(f"Total failures where Tool & Params were 100% PERFECT: {len(failures)}")
    # Print the first few to diagnose
    for count, f in enumerate(failures[:10]):
        print(f"  {count+1}. Task: {f['task']:<20} | Adj F1: {f['adj_f1']:.3f} | Expected: {str(f['expected']):<15} | Got from Python: {str(f['actual']):<15}")
    print()

debug_eval("/home/ramji.purwar/DecompGTI/DecompGTI/data/test_set_mini.json", "/home/ramji.purwar/DecompGTI/DecompGTI/evaluation/results/qwen7b_v3/mini_predictions.json")
debug_eval("/home/ramji.purwar/DecompGTI/DecompGTI/data/test_set_medium.json", "/home/ramji.purwar/DecompGTI/DecompGTI/evaluation/results/qwen7b_v3/medium_predictions.json")
