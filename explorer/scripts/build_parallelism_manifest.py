#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

GENERATED_AT = "static"

STRATEGIES = [
    {
        "id": "dp",
        "label": "Data Parallelism",
        "objective": "Batch dimension",
        "bestFor": "Data scalability and standard training",
        "flags": ["--data-parallel-sharding-strategy"],
        "sourceRefs": [
            "docs/user-guide/parallelism-guide.md",
            "megatron/core/distributed/distributed_data_parallel.py",
        ],
        "desc": "Replicate or shard model state across ranks while splitting the batch.",
    },
    {
        "id": "tp",
        "label": "Tensor Parallelism",
        "objective": "Individual layers",
        "bestFor": "Large hidden dimensions and projection matrices",
        "flags": ["--tensor-model-parallel-size", "--sequence-parallel"],
        "sourceRefs": ["docs/user-guide/parallelism-guide.md", "megatron/core/tensor_parallel/layers.py"],
        "desc": "Shard large linear layers and attention heads across ranks.",
    },
    {
        "id": "pp",
        "label": "Pipeline Parallelism",
        "objective": "Model depth",
        "bestFor": "Very deep models",
        "flags": ["--pipeline-model-parallel-size", "--num-layers-per-virtual-pipeline-stage"],
        "sourceRefs": ["docs/user-guide/parallelism-guide.md", "megatron/core/pipeline_parallel/schedules.py"],
        "desc": "Partition layers by depth and schedule microbatches through stages.",
    },
    {
        "id": "cp",
        "label": "Context Parallelism",
        "objective": "Sequence length",
        "bestFor": "Long sequence training",
        "flags": ["--context-parallel-size", "--cp-comm-type"],
        "sourceRefs": ["docs/user-guide/parallelism-guide.md", "docs/user-guide/features/context_parallel.md"],
        "desc": "Split sequence/context work across ranks to reduce activation pressure.",
    },
    {
        "id": "ep",
        "label": "Expert Parallelism",
        "objective": "MoE experts",
        "bestFor": "Mixture-of-Experts models",
        "flags": ["--expert-model-parallel-size", "--num-experts", "--moe-grouped-gemm"],
        "sourceRefs": ["docs/user-guide/parallelism-guide.md", "megatron/core/transformer/moe/moe_layer.py"],
        "desc": "Distribute expert weights and route tokens to top-k experts.",
    },
    {
        "id": "fsdp",
        "label": "Megatron FSDP",
        "objective": "Model state",
        "bestFor": "Large models with data-parallel state pressure",
        "flags": ["--use-megatron-fsdp", "--data-parallel-sharding-strategy"],
        "sourceRefs": [
            "docs/user-guide/features/megatron_fsdp.md",
            "megatron/core/distributed/fsdp/mcore_fsdp_adapter.py",
        ],
        "desc": "Shard parameters, gradients, and optimizer state across data-parallel ranks.",
    },
]


PRESETS = [
    {
        "id": "llama3-8b",
        "label": "Llama3 8B local",
        "worldSize": 8,
        "tp": 1,
        "pp": 1,
        "cp": 1,
        "ep": 1,
        "note": "Matches examples/llama/configs/llama3_8b_long.yaml.",
    },
    {
        "id": "llama70b",
        "label": "Llama 70B guide",
        "worldSize": 64,
        "tp": 4,
        "pp": 4,
        "cp": 2,
        "ep": 1,
        "note": "From the parallelism guide recommendation table.",
    },
    {
        "id": "deepseek-v3",
        "label": "DeepSeek-V3 guide",
        "worldSize": 2048,
        "tp": 2,
        "pp": 16,
        "cp": 1,
        "ep": 64,
        "note": "Valid illustrative MoE layout derived from the parallelism guide values.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="explorer/public/data/parallelism")
    args = parser.parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = Path(args.repo_root) / out
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at": GENERATED_AT, "strategies": STRATEGIES, "presets": PRESETS}
    (out / "strategies.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "index.json").write_text(
        json.dumps(
            [{"slug": "strategies", "label": "Parallelism Strategies", "manifest": "parallelism/strategies.json"}],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
