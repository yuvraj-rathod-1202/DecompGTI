from __future__ import annotations

import json

from decompgti.mcp_server.pipeline import execute_from_model_output
from decompgti.mcp_server.tool_catalog import get_model_tool_prompt_block


def main() -> None:
    print("=== Tool Prompt Block ===")
    print(get_model_tool_prompt_block())

    model_output = json.dumps(
        {
            "tool_name": "maximum_flow",
            "arguments": {
                "edges": [["S", "A", 3], ["S", "B", 2], ["A", "T", 2], ["B", "T", 3]],
                "source": "S",
                "target": "T",
            },
        }
    )

    execution = execute_from_model_output(model_output)

    print("\n=== Parsed + Executed ===")
    print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
