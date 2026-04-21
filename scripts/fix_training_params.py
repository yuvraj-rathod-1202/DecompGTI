"""
Fix training data: fill in missing step3_tool_parameters for BFS, DFS, max_flow.

The original csv2json.py had the right code but the regex didn't match the
actual question formats. This script directly patches graph_reasoning.json.
"""
import json
import re
from pathlib import Path

TRAINING_JSON = Path(__file__).resolve().parents[1] / "GraphInstruct/LLaMAFactory/data/graph_reasoning.json"

# Patterns to extract source/target from question text
PATTERNS = [
    re.compile(r"from node (\d+) to node (\d+)", re.IGNORECASE),
    re.compile(r"starting from node (\d+).*?to node (\d+)", re.IGNORECASE),
    re.compile(r"from node (\d+).*?reach node (\d+)", re.IGNORECASE),
    re.compile(r"source node (\d+).*?sink node (\d+)", re.IGNORECASE),
]

SOURCE_ONLY_PATTERNS = [
    re.compile(r"starting from node (\d+)", re.IGNORECASE),
    re.compile(r"from node (\d+)", re.IGNORECASE),
    re.compile(r"start(?:ing)? at node (\d+)", re.IGNORECASE),
]

NEEDS_SOURCE_TARGET = {
    "breadth_first_search", "depth_first_search", "maximum_flow",
}
NEEDS_SOURCE = {
    "breadth_first_search", "depth_first_search",
}


def extract_params(tool_name, question):
    """Try to extract source/target from the question text."""
    params = {}

    # Try source+target patterns first
    for pat in PATTERNS:
        m = pat.search(question)
        if m:
            params["source"] = int(m.group(1))
            params["target"] = int(m.group(2))
            return params

    # For BFS/DFS, try source-only patterns
    if tool_name in NEEDS_SOURCE:
        for pat in SOURCE_ONLY_PATTERNS:
            m = pat.search(question)
            if m:
                params["source"] = int(m.group(1))
                return params

    return params


def main():
    print(f"Loading {TRAINING_JSON}...")
    with open(TRAINING_JSON) as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    fixed = 0
    already_ok = 0
    no_match = 0
    stats = {}

    for item in data:
        output = json.loads(item["output"])
        tool = output.get("step2_tool_name", "")
        old_params = output.get("step3_tool_parameters", {})
        question = item.get("input", "")

        if tool not in stats:
            stats[tool] = {"fixed": 0, "ok": 0, "no_match": 0}

        if old_params:
            already_ok += 1
            stats[tool]["ok"] += 1
            continue

        new_params = extract_params(tool, question)
        if new_params:
            output["step3_tool_parameters"] = new_params
            item["output"] = json.dumps(output, ensure_ascii=False)
            fixed += 1
            stats[tool]["fixed"] += 1
        else:
            no_match += 1
            stats[tool]["no_match"] += 1

    print(f"\nResults:")
    print(f"  Already had params: {already_ok}")
    print(f"  Fixed (params added): {fixed}")
    print(f"  No match (no params in question): {no_match}")
    print(f"\nPer tool:")
    for tool in sorted(stats):
        s = stats[tool]
        print(f"  {tool:<30} ok:{s['ok']:>5}  fixed:{s['fixed']:>5}  no_match:{s['no_match']:>5}")

    # Backup original
    backup = TRAINING_JSON.with_suffix(".json.bak")
    import shutil
    shutil.copy2(TRAINING_JSON, backup)
    print(f"\nBackup saved to {backup}")

    # Write fixed data
    with open(TRAINING_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Fixed data saved to {TRAINING_JSON}")

    # Verify
    print("\nVerification - sample outputs:")
    for item in data:
        output = json.loads(item["output"])
        tool = output["step2_tool_name"]
        params = output["step3_tool_parameters"]
        if tool in NEEDS_SOURCE_TARGET and params:
            print(f"  {tool}: {params}")
            break


if __name__ == "__main__":
    main()
