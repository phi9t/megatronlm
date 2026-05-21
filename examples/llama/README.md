# Llama Models

## Overview

This example provides a small, mock-data Llama anchor run for checking the local
Megatron-LM training environment in Docker. It follows the same repo-local
runtime pattern as the anchor training launcher:

- source is mounted read-only and copied into a writable run directory
- outputs, caches, and TensorBoard logs are written under `local/`
- the container runs as the host user
- CUDA, NCCL, PyTorch, and TransformerEngine checks can run before training

## Quick Start

Run the small Llama anchor training job:

```bash
mise run llama-anchor
```

Run a short foreground smoke check:

```bash
mise run llama-anchor-smoke
```

Run the preflight checks plus a 2-iteration mini train:

```bash
mise run llama-anchor-preflight
```

Preview the Docker command without launching it:

```bash
mise run llama-anchor-preflight -- --dry-run
```

The default outputs are written to `local/llama-anchor`, with smoke and preflight
artifacts under `local/llama-anchor-smoke` and `local/llama-anchor-preflight`.

## Launcher Options

The `mise` tasks call:

```bash
uv run python examples/llama/run_llama_anchor.py
```

Common options:

```bash
--image megatron-lm:smoke
--name mcore-llama-anchor
--output-dir local/llama-anchor
--duration-mins 130
--train-iters 500000
--preflight
--foreground
--dry-run
```

The small anchor configuration uses mock data with Llama-style model choices:
RoPE, RMSNorm, SwiGLU, grouped query attention, MCore GPT models, BF16, and
TransformerEngine.

## Long 8B FP8 Run

The managed long run wraps the Llama-3 8B FP8 example in the same repo-local
Docker pattern as the anchor run. It defaults to mock data, 8 local GPUs, and a
235-minute exit duration.

Run the foreground preflight first:

```bash
mise run llama3-8b-long-preflight
```

Launch the detached long run:

```bash
mise run llama3-8b-long
```

Preview the Docker command:

```bash
mise run llama3-8b-long-dry-run
```

The default outputs are written to `local/llama3-8b-long`. Checkpoints,
TensorBoard logs, caches, and profiler output are kept under that directory.

The long-run launcher can also run with real data:

```bash
uv run python examples/llama/run_llama3_8b_long.py \
  --tokenizer-model /path/to/tokenizer \
  --data-path /path/to/data_prefix
```

Use `--resume` to add `--load /outputs/checkpoints` when continuing from a
checkpoint in the output directory.

## 8B FP8 Configuration

Default parallelism strategy:

- Tensor Parallel: 1
- Pipeline Parallel: 1
- Context Parallel: 1

Llama-3 8B-style architecture:

- 32 layers
- Hidden size: 4096
- FFN hidden size: 14336
- Attention heads: 32
- Query groups: 8
- Sequence length: 8192
- RMSNorm normalization with SwiGLU and RoPE

Key training parameters:

- Micro-batch size: 1
- Global batch size: 128
- Learning rate: 1.5e-4
- Min learning rate: 1.0e-5
- Weight decay: 0.1
- FP8 format: hybrid

FP8 requires NVIDIA Hopper, Ada, or Blackwell GPUs.

The original `train_llama3_8b_h100_fp8.sh` script remains available for manual
benchmark runs, but the `mise` tasks above are the managed local path.
