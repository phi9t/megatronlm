#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _refs import ref  # noqa: E402

GENERATED_AT = "static"


def node(
    repo: Path,
    id: str,
    label: str,
    group: str,
    file: str,
    symbol: str | None,
    desc: str,
    docs: list[str] | None = None,
):
    return {
        "id": id,
        "label": label,
        "group": group,
        "ref": ref(repo, file, symbol),
        "symbol": symbol,
        "desc": desc,
        "docs": docs or [],
    }


def training_loop(repo: Path):
    nodes = [
        node(
            repo,
            "yaml",
            "YAML config",
            "entry",
            "examples/llama/llama_config.py",
            "class LlamaRunConfig",
            "Pydantic-style config objects turn YAML presets into Megatron CLI arguments.",
            ["examples/llama/README.md"],
        ),
        node(
            repo,
            "launcher",
            "Launcher",
            "entry",
            "examples/llama/run_llama.py",
            "def main",
            "The example launcher validates config, prepares runtime paths, and invokes the training command.",
        ),
        node(
            repo,
            "args",
            "Arguments",
            "training",
            "megatron/training/yaml_arguments.py",
            "def core_transformer_config_from_yaml",
            "Megatron normalizes CLI/YAML values into runtime arguments and transformer config.",
        ),
        node(
            repo,
            "initialize",
            "Initialize",
            "training",
            "megatron/training/initialize.py",
            "def initialize_megatron",
            "Distributed initialization, tokenizer setup, random seeds, and model-parallel groups.",
        ),
        node(
            repo,
            "train",
            "Training loop",
            "training",
            "megatron/training/training.py",
            "def pretrain",
            "Top-level pretraining loop wiring model, data iterators, forward/backward, optimizer, evaluation, and checkpointing.",
        ),
        node(
            repo,
            "dataset",
            "Datasets",
            "state",
            "megatron/core/datasets/gpt_dataset.py",
            "class GPTDataset",
            "GPT-style indexed datasets provide token samples to the training loop.",
        ),
        node(
            repo,
            "model",
            "Model provider",
            "core",
            "megatron/core/models/gpt/gpt_model.py",
            "class GPTModel",
            "The GPT model wraps embeddings, transformer layers, final norm, and output projection.",
        ),
        node(
            repo,
            "schedule",
            "Forward/backward",
            "parallel",
            "megatron/core/pipeline_parallel/schedules.py",
            "def get_forward_backward_func",
            "Pipeline schedules drive microbatch forward/backward execution.",
        ),
        node(
            repo,
            "optimizer",
            "Optimizer",
            "state",
            "megatron/core/optimizer/optimizer.py",
            "class MegatronOptimizer",
            "Optimizer wrapper coordinates parameter updates, grad scaling, clipping, and distributed state.",
        ),
        node(
            repo,
            "checkpoint",
            "Checkpointing",
            "state",
            "megatron/training/checkpointing.py",
            "def save_checkpoint",
            "Checkpoint save/load preserves model, optimizer, RNG, and distributed state.",
        ),
    ]
    edges = [
        ("yaml", "launcher"),
        ("launcher", "args"),
        ("args", "initialize"),
        ("initialize", "train"),
        ("train", "dataset"),
        ("train", "model"),
        ("train", "schedule"),
        ("schedule", "optimizer"),
        ("optimizer", "checkpoint"),
        ("checkpoint", "train"),
    ]
    return manifest("training-loop", "Training Loop", nodes, edges)


def core_stack(repo: Path):
    nodes = [
        node(
            repo,
            "gpt",
            "GPTModel",
            "core",
            "megatron/core/models/gpt/gpt_model.py",
            "class GPTModel",
            "GPTModel assembles embeddings, transformer block, final norm, and output layer.",
        ),
        node(
            repo,
            "block",
            "TransformerBlock",
            "core",
            "megatron/core/transformer/transformer_block.py",
            "class TransformerBlock",
            "A stack of transformer layers built from module specs.",
        ),
        node(
            repo,
            "layer",
            "TransformerLayer",
            "core",
            "megatron/core/transformer/transformer_layer.py",
            "class TransformerLayer",
            "Self-attention, residual paths, layer norms, dense MLP, or MoE.",
        ),
        node(
            repo,
            "attention",
            "Attention",
            "core",
            "megatron/core/transformer/attention.py",
            "class Attention",
            "Attention projection, core attention, and output projection.",
        ),
        node(
            repo,
            "mlp",
            "Dense MLP",
            "core",
            "megatron/core/transformer/mlp.py",
            "class MLP",
            "Dense feed-forward block used by standard transformer layers.",
        ),
        node(
            repo,
            "moe",
            "MoELayer",
            "core",
            "megatron/core/transformer/moe/moe_layer.py",
            "class MoELayer",
            "Mixture-of-Experts routing, dispatch, expert compute, and combine.",
        ),
        node(
            repo,
            "tp",
            "Tensor parallel",
            "parallel",
            "megatron/core/tensor_parallel/layers.py",
            "class ColumnParallelLinear",
            "Layer-level sharding for large projections.",
        ),
        node(
            repo,
            "pp",
            "Pipeline parallel",
            "parallel",
            "megatron/core/pipeline_parallel/schedules.py",
            "def get_forward_backward_func",
            "Layer-depth partitioning and microbatch schedules.",
        ),
        node(
            repo,
            "ddp",
            "Distributed",
            "parallel",
            "megatron/core/distributed/distributed_data_parallel.py",
            "class DistributedDataParallel",
            "Data-parallel gradient synchronization and overlap.",
        ),
        node(
            repo,
            "optim",
            "Optimizer",
            "state",
            "megatron/core/optimizer/distrib_optimizer.py",
            "class DistributedOptimizer",
            "Distributed optimizer state and parameter shard handling.",
        ),
    ]
    edges = [
        ("gpt", "block"),
        ("block", "layer"),
        ("layer", "attention"),
        ("layer", "mlp"),
        ("layer", "moe"),
        ("attention", "tp"),
        ("mlp", "tp"),
        ("moe", "tp"),
        ("block", "pp"),
        ("gpt", "ddp"),
        ("ddp", "optim"),
    ]
    return manifest("core-stack", "Megatron Core Stack", nodes, edges)


def manifest(slug: str, label: str, nodes: list[dict], edges: list[tuple[str, str]]):
    return {
        "slug": slug,
        "label": label,
        "generated_at": GENERATED_AT,
        "warnings": [],
        "nodes": nodes,
        "edges": [{"from": a, "to": b} for a, b in edges],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="explorer/public/data/graphs")
    args = parser.parse_args()
    repo = Path(args.repo_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graphs = [training_loop(repo), core_stack(repo)]
    for graph in graphs:
        (out / f"{graph['slug']}.json").write_text(
            json.dumps(graph, indent=2) + "\n", encoding="utf-8"
        )
    index = [
        {"slug": g["slug"], "label": g["label"], "manifest": f"graphs/{g['slug']}.json"}
        for g in graphs
    ]
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
