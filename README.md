# DecompGTI

Graph-tool reasoning project with an MCP server for deterministic execution.

## Acknowledgements and Citation

This project uses:

- GraphInstruct for graph-task data generation.
- LLaMAFactory for LLM fine-tuning workflows.

If you use this repository, please also cite the upstream works:

```bibtex
@article{graphinstruct,
	title={GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability},
	author={Luo, Zihan and Song, Xiran and Huang, Hong and Lian, Jianxun and Zhang, Chenhao and Jiang, Jinqi and Xie, Xing},
	journal={CoRR},
	volume={abs/2403.04483},
	year={2024},
	doi={10.48550/ARXIV.2403.04483},
	url={https://arxiv.org/abs/2403.04483}
}

@inproceedings{zheng-etal-2024-llamafactory,
	title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
	author={Zheng, Yaowei and Zhang, Richong and Zhang, Junhao and Ye, Yanhan and Luo, Zheyan and Feng, Zhangchi and Ma, Yongqiang},
	booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
	year={2024},
	publisher={Association for Computational Linguistics},
	address={Bangkok, Thailand},
	url={https://arxiv.org/abs/2403.13372}
}
```

## Setup

1. Sync dependencies:

```bash
uv sync
```

2. Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Run MCP Server

```bash
uv run decompgti-mcp-server
```

## MCP Tools Implemented

- `bfs`
- `dfs`
- `dijkstra_shortest_path`
- `maximum_flow`
- `bipartite_maximum_matching`
- `topological_sort`
- `cycle_detection`
- `minimum_spanning_tree`
- `extract_tool_name_from_model_output`
- `extract_tool_call_from_model_output`

## Model Routing Helpers

Routing utilities are in `src/decompgti/mcp_server/routing.py`:

- `extract_tool_name(model_output)`
- `extract_tool_call(model_output)`
- `build_tool_call_from_model_output(model_output)`

Tool metadata for prompting is in `src/decompgti/mcp_server/tool_catalog.py` via `get_model_tool_prompt_block()`.

## End-to-End Demo

Run a local roundtrip demo (model output -> parse -> execute tool):

```bash
uv run python scripts/demo_roundtrip.py
```

Execution helper used in tests and demo:

- `src/decompgti/mcp_server/pipeline.py` (`execute_from_model_output`)