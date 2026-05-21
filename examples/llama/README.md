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

Run the default Llama anchor training job:

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

## Advanced 8B FP8 Benchmark

The original `train_llama3_8b_h100_fp8.sh` script remains available for larger
manual benchmark runs. It targets a Llama-3 8B-style FP8 configuration and is
not the default smoke path.

Set host paths:

```bash
export HOST_MEGATRON_LM_DIR="/path/to/your/host/megatron-lm"
export HOST_CHECKPOINT_PATH="./checkpoints/llama3_8b_fp8"
export HOST_TENSORBOARD_LOGS_PATH="./tensorboard_logs/llama3_8b_fp8"
```

Run with mock data:

```bash
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:25.03-py3"

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "${HOST_MEGATRON_LM_DIR}:/workspace/megatron-lm" \
  -v "${HOST_CHECKPOINT_PATH}:/workspace/checkpoints" \
  -v "${HOST_TENSORBOARD_LOGS_PATH}:/workspace/tensorboard_logs" \
  --workdir /workspace/megatron-lm \
  "${PYTORCH_IMAGE}" \
  bash examples/llama/train_llama3_8b_h100_fp8.sh \
    /workspace/checkpoints \
    /workspace/tensorboard_logs
```

For custom data, pass a tokenizer model and data prefix as the third and fourth
arguments to `train_llama3_8b_h100_fp8.sh`.

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
