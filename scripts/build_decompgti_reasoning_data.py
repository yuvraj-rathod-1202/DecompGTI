from __future__ import annotations

import argparse
import json
from pathlib import Path

from decompgti.data.decompgti_reasoning import DECOMPGTI_TASKS, build_reasoning_records_for_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DecompGTI reasoning data for LLaMAFactory.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("GraphInstruct") / "data" / "dataset",
        help="Root directory containing the raw GraphInstruct task CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("GraphInstruct") / "LLaMAFactory" / "data" / "reasoning" / "decompgti",
        help="Directory where the generated train.json will be written.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(DECOMPGTI_TASKS),
        help="Subset of proposal tasks to convert.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_reasoning_records_for_tasks(args.dataset_root, tuple(args.tasks))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "train.json"
    output_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} records to {output_file}")


if __name__ == "__main__":
    main()