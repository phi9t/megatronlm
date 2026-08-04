# The Engineer's Hacker Guide to Megatron-LM

Welcome, hacker. This guide is a no-nonsense technical entry point into the Megatron-LM codebase. It's designed for engineers who want to understand the internals, modify the core, and run serious training jobs.

---

## Table of Contents

1. [The Mental Model](#1-the-mental-model)
2. [Setup for Hackers](#2-setup-for-hackers)
3. [Data Prep Hackery](#3-data-prep-hackery)
4. [Running Training](#4-running-training-the-real-way)
5. [Debugging like a Pro](#5-debugging-like-a-pro)
6. [Codebase Navigation](#6-codebase-navigation)
7. [Performance Tuning](#7-performance-tuning-tips)
8. [Deep Dive into MCore](#8-deep-dive-into-megatron-core-mcore)
9. [Parallelism Architecture](#9-parallelism-architecture)
10. [Transformer Internals](#10-transformer-internals)
11. [Module Specification System](#11-module-specification-system)
12. [Building Models](#12-building-models)
13. [Training Pipeline Internals](#13-training-pipeline-internals)
14. [Key Files Reference](#14-key-files-reference)
15. [Common Modifications](#15-common-modifications)
16. [The Hacker Script](#16-the-hacker-script-hacker_megatron_lmpy)

---

## 1. The Mental Model

Megatron-LM is split into two layers:
- **Megatron Core (MCore)**: The `megatron/core` directory. This is the "library" layer. It contains the optimized transformer blocks, parallelism logic (TP, PP, CP), and distributed optimizers. It's meant to be modular and framework-agnostic.
- **Megatron-LM (Training Scripts)**: The top-level scripts like `pretrain_gpt.py`, `pretrain_bert.py`, etc., and the `megatron/training` directory. This is the "application" layer that uses MCore to run specific training loops.

**Rule of Thumb**: If you're changing how a transformer layer works, go to `megatron/core`. If you're changing how the training arguments are parsed or how the logger works, go to `megatron/training`.

### Directory Overview

```
megatron-lm/
├── megatron/
│   ├── core/                          # Production library
│   │   ├── transformer/               # Transformer building blocks
│   │   │   ├── attention.py           # Self/Cross attention (49KB)
│   │   │   ├── mlp.py                 # Feed-forward layers (16KB)
│   │   │   ├── transformer_layer.py   # Single layer (38KB)
│   │   │   ├── transformer_block.py   # Layer stack (36KB)
│   │   │   ├── transformer_config.py  # Configuration (78KB)
│   │   │   ├── spec_utils.py          # Module spec system
│   │   │   └── moe/                   # Mixture of Experts
│   │   ├── models/                    # Complete models
│   │   │   ├── gpt/                   # GPT (gpt_model.py, gpt_layer_specs.py)
│   │   │   ├── bert/                  # BERT
│   │   │   ├── mamba/                 # State-space models
│   │   │   └── multimodal/            # Vision-language
│   │   ├── tensor_parallel/           # Intra-layer parallelism
│   │   ├── pipeline_parallel/         # Inter-layer parallelism
│   │   ├── distributed/               # DDP, gradient sync
│   │   ├── optimizer/                 # Distributed optimizers
│   │   ├── dist_checkpointing/        # Sharded checkpointing
│   │   ├── datasets/                  # Data loading
│   │   └── parallel_state.py          # Process groups (87KB)
│   │
│   ├── training/                      # Training infrastructure
│   │   ├── training.py                # Main loop (125KB)
│   │   ├── arguments.py               # All args (207KB)
│   │   ├── checkpointing.py           # Checkpoint I/O (83KB)
│   │   └── initialize.py              # Distributed setup
│   │
│   └── legacy/                        # Legacy code (being phased out)
│
├── pretrain_gpt.py                    # GPT entry point
├── pretrain_bert.py                   # BERT entry point
├── examples/
│   └── run_simple_mcore_train_loop.py # Minimal example
└── hacker_megatron_lm.py              # Interactive exploration script
```

---

## 2. Setup for Hackers

Hackers don't reinstall dependencies every time they restart a container. We use `uv` and persistent mounts to reuse environments, models, and datasets.

### The Persistence Layer
Create a scratch directory on your host to persist libraries, models, and data:
```bash
mkdir -p scratch/{libs,models,data}
```

### The Container Launch
Launch with mounts for code (`megatron-lm`), venvs (`libs`), models (`models`), and datasets (`data`).
```bash
docker run --runtime=nvidia --gpus all -it --rm \
  --ipc=host --shm-size=1g \
  --workdir /workspace/megatron-lm \
  -v $(pwd):/workspace/megatron-lm \
  -v $(pwd)/scratch/libs:/workspace/libs \
  -v $(pwd)/scratch/models:/workspace/models \
  -v $(pwd)/scratch/data:/workspace/data \
  -e HF_HOME=/workspace/models \
  -e UV_CACHE_DIR=/workspace/libs/uv_cache \
  nvcr.io/nvidia/pytorch:25.04-py3
```

### The "One-Time" Install
Inside the container, set up a persistent virtual environment using `uv`:
```bash
# 1. Install uv (ephemeral, but installs in milliseconds)
pip install uv

# 2. Create persistent venv (only does this if missing)
if [ ! -d "/workspace/libs/venv" ]; then
    uv venv /workspace/libs/venv
fi

# 3. Activate the persistent environment
source /workspace/libs/venv/bin/activate

# 4. Install Megatron in editable mode
uv pip install -e .[dev,mlm]
```
Now your environment persists across container restarts. Just run `source /workspace/libs/venv/bin/activate` next time.

---

## 3. Data Prep Hackery

Megatron uses a custom binary format (`.bin` and `.idx`) for high-performance data loading.

### Preprocessing Real Data
If you have a JSONL file where each line is `{"text": "..."}`:
```bash
python tools/preprocess_data.py \
       --input my_data.jsonl \
       --output-prefix my_dataset \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --append-eod \
       --workers 16
```
This produces `my_dataset_text_document.bin` and `my_dataset_text_document.idx`.

---

## 4. Running Training (The Real Way)

Standard scripts in `examples/` are your best starting point. Let's look at LLaMA 3 8B:
```bash
bash examples/llama/train_llama3_8b_h100_fp8.sh
```

### Key Arguments for Hackers:
- `--use-mcore-models`: Use the new modular Megatron Core models instead of legacy ones.
- `--tensor-model-parallel-size (TP)`: Splits layers across GPUs (intra-layer).
- `--pipeline-model-parallel-size (PP)`: Splits layers across GPUs (inter-layer).
- `--sequence-parallel`: Reduces activation memory when using TP.
- `--recompute-activations`: Trades compute for memory (checkpointing).
- `--overlap-grad-reduce`: Overlaps communication with computation.

---

## 5. Debugging like a Pro

Distributed training is hard to debug. Use these environment variables:

- **NCCL Debugging**: `export NCCL_DEBUG=INFO` (shows GPU communication status).
- **Distributed Debugging**: `export TORCH_DISTRIBUTED_DEBUG=DETAIL`.
- **Memory Debugging**: `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.

### Print-on-Rank-0
Don't use `print()`. Use `print_rank_0` from `megatron.training.utils`:
```python
from megatron.training.utils import print_rank_0
print_rank_0("Hacker alert: model is initializing...")
```

### Gradient Inspection
Register hooks to verify gradients:
```python
def grad_hook(grad):
    print(f"Grad shape: {grad.shape}, norm: {grad.norm().item():.6f}")
    return grad
tensor.register_hook(grad_hook)
```

---

## 6. Codebase Navigation

- **Kernels & Fusions**: `megatron/core/fusions/`. Check here for fused operators like `fused_bias_gelu.py`.
- **Parallelism Logic**:
  - `megatron/core/tensor_parallel/`: Mappings and layers for TP.
  - `megatron/core/pipeline_parallel/`: Schedules (1F1B, etc.) and P2P communication.
- **Model definitions**: `megatron/core/models/gpt/gpt_model.py` is the gold standard for how to build a model in MCore.
- **Dataset Internals**: `megatron/core/datasets/indexed_dataset.py` handles the low-level mmap logic.

---

## 7. Performance Tuning Tips

1. **FlashAttention**: Megatron uses it by default via Transformer Engine. Ensure `--attention-backend fused` is set.
2. **Micro-batch size (MBS) vs Global-batch size (GBS)**:
   - `GBS = MBS * DP_size * accumulation_steps`.
   - Keep MBS as large as possible without OOMing to maximize TFLOPS.
3. **Communication Overlap**: Always use `--overlap-grad-reduce` and `--overlap-param-gather` if you have high-speed interconnects (NVLink).

---

## 8. Deep Dive into Megatron Core (MCore)

MCore is where the high-performance magic happens. Here are the critical modules you'll likely touch:

- **`megatron.core.parallel_state`**: The heart of the distributed system. It manages `ProcessGroup`s for Tensor, Pipeline, Context, and Data parallelism. Use `get_tensor_model_parallel_group()` to see where your weights are sharded.
- **`megatron.core.tensor_parallel`**: Contains `ColumnParallelLinear` and `RowParallelLinear`, the foundation of TP. It also handles random seed management for reproducibility across ranks.
- **`megatron.core.transformer`**:
  - `transformer_config.py`: The `TransformerConfig` dataclass is passed everywhere.
  - `attention.py`: Implements multi-head and grouped-query attention.
  - `transformer_layer.py`: The high-level assembly of norm -> attn -> norm -> mlp.
- **`megatron.core.distributed`**: Implements MCore's version of DDP (`DistributedDataParallel`). Unlike PyTorch DDP, this is optimized for overlapping communication with the backward pass of pipeline stages.
- **`megatron.core.optimizer`**: The `DistributedOptimizer` shards optimizer states across data-parallel ranks to save memory (similar to ZeRO-1/2).
- **`megatron.core.datasets`**: Handles `IndexedDataset` (the `.bin`/`.idx` reader) and `BlendedMegatronDataset` for mixing multiple data sources on the fly.

---

## 9. Parallelism Architecture

### Process Groups

Megatron manages multiple communication groups:

```python
from megatron.core import parallel_state

# Initialize parallelism
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=2,    # TP: split weights within layers
    pipeline_model_parallel_size=2,  # PP: split layers across GPUs
    # Data parallelism = world_size / (TP * PP)
)

# Query current rank's position
tp_rank = parallel_state.get_tensor_model_parallel_rank()
pp_rank = parallel_state.get_pipeline_model_parallel_rank()
dp_rank = parallel_state.get_data_parallel_rank()

# Get process groups for communication
tp_group = parallel_state.get_tensor_model_parallel_group()
```

### Tensor Parallelism (TP)

Splits weight matrices across GPUs within a layer:

```
Linear Layer: Y = XW

TP=2 splits W into [W1, W2]:
  GPU 0: Y1 = X @ W1
  GPU 1: Y2 = X @ W2

Then all-gather or reduce-scatter to combine
```

Key classes in `megatron/core/tensor_parallel/layers.py`:
- `ColumnParallelLinear`: Splits output features (columns)
- `RowParallelLinear`: Splits input features (rows)
- `VocabParallelEmbedding`: Splits vocabulary

### Pipeline Parallelism (PP)

Splits transformer layers across GPUs:

```
PP=2:
  GPU 0: Embedding + Layers 0-5 (pre_process=True)
  GPU 1: Layers 6-11 + LM Head (post_process=True)

Forward: GPU0 → send activations → GPU1
Backward: GPU1 → send gradients → GPU0
```

Schedules in `megatron/core/pipeline_parallel/schedules.py`:
- **1F1B**: Alternates forward/backward for memory efficiency
- **GPipe**: All forwards, then all backwards

### Data Parallelism (DP)

Replicates model across GPUs, averages gradients:

```
DP=4:
  Each GPU has full model copy
  Each GPU processes different data batch
  All-reduce gradients across DP group
```

### Combined Example

```
World Size = 16 GPUs
TP=2, PP=4, DP=2

Arrangement:
  DP Group 0:
    PP Stage 0: GPU 0-1 (TP group)
    PP Stage 1: GPU 2-3 (TP group)
    PP Stage 2: GPU 4-5 (TP group)
    PP Stage 3: GPU 6-7 (TP group)

  DP Group 1:
    PP Stage 0: GPU 8-9 (TP group)
    ...
```

---

## 10. Transformer Internals

### Layer Structure

```
TransformerLayer (transformer_layer.py)
├── Input LayerNorm
├── SelfAttention (attention.py)
│   ├── QKV Linear (ColumnParallel)
│   ├── Core Attention (scaled dot-product)
│   └── Output Linear (RowParallel)
├── Residual Connection
├── Post-Attention LayerNorm
├── MLP (mlp.py)
│   ├── FC1 Linear (ColumnParallel)
│   ├── Activation (GELU, SwiGLU)
│   └── FC2 Linear (RowParallel)
└── Residual Connection
```

### Attention Variants

```python
# Standard Multi-Head Attention
SelfAttention with num_query_groups=None

# Grouped Query Attention (GQA)
SelfAttention with num_query_groups=4  # Fewer KV heads

# Multi-Latent Attention (MLA)
MLASelfAttention  # Compressed KV cache
```

### Position Embeddings

```python
config = TransformerConfig(
    position_embedding_type='rope',  # learned_absolute, rope, yarn
    rotary_base=10000,
    rotary_percent=1.0,
)
```

---

## 11. Module Specification System

Megatron uses a spec pattern for swappable implementations:

```python
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

# A spec defines what to build and how
attention_spec = ModuleSpec(
    module=SelfAttention,
    params={"attn_mask_type": AttnMaskType.causal},
    submodules=SelfAttentionSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=DotProductAttention,
        linear_proj=RowParallelLinear,
    ),
)

# Build module from spec
attention = build_module(attention_spec, config=transformer_config)
```

### Available Layer Specs

```python
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,                      # Pure PyTorch
    get_gpt_layer_with_transformer_engine_spec,   # TE optimized
    get_gpt_layer_with_inference_spec,            # Inference optimized
)
```

---

## 12. Building Models

### TransformerConfig

All model configuration flows through `TransformerConfig`:

```python
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    # Architecture
    num_layers=12,
    hidden_size=768,
    num_attention_heads=12,
    ffn_hidden_size=3072,  # Defaults to 4 * hidden_size

    # Parallelism
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    sequence_parallel=True,

    # Precision
    bf16=True,
    params_dtype=torch.bfloat16,

    # Regularization
    hidden_dropout=0.1,
    attention_dropout=0.1,

    # Other
    add_bias_linear=False,
    layernorm_epsilon=1e-5,
)
```

### Building a GPT Model

```python
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

model = GPTModel(
    config=config,
    transformer_layer_spec=get_gpt_layer_local_spec(),
    vocab_size=50257,
    max_sequence_length=1024,
    pre_process=True,   # Include embeddings
    post_process=True,  # Include LM head
)
```

### Forward Pass

```python
tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
attention_mask = torch.ones(batch_size, seq_len)

output = model(
    input_ids=tokens,
    position_ids=position_ids,
    attention_mask=attention_mask,
    labels=labels,  # Optional, for loss
)
```

---

## 13. Training Pipeline Internals

### Initialization Sequence

```python
# 1. Initialize distributed
torch.distributed.init_process_group()

# 2. Initialize model parallel groups
parallel_state.initialize_model_parallel(tp_size, pp_size)

# 3. Seed RNGs (important for TP determinism)
model_parallel_cuda_manual_seed(seed)

# 4. Build model
model = model_provider()

# 5. Wrap with DDP
model = DistributedDataParallel(config, ddp_config, model)

# 6. Create optimizer
optimizer = get_megatron_optimizer(model, optimizer_config)
```

### Training Loop

```python
forward_backward_func = get_forward_backward_func()

for iteration in range(num_iterations):
    optimizer.zero_grad()

    # Forward and backward with pipeline schedule
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iter,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=seq_len,
        micro_batch_size=micro_batch,
    )

    # Sync gradients across all parallel dimensions
    finalize_model_grads([model])

    # Update
    optimizer.step()
```

---

## 14. Key Files Reference

| File | Purpose | Size |
|------|---------|------|
| `megatron/core/parallel_state.py` | Process group management | 87KB |
| `megatron/core/transformer/transformer_config.py` | Model configuration | 78KB |
| `megatron/core/transformer/attention.py` | Attention implementations | 49KB |
| `megatron/core/transformer/mlp.py` | MLP layers | 16KB |
| `megatron/core/transformer/transformer_layer.py` | Single transformer layer | 38KB |
| `megatron/core/transformer/transformer_block.py` | Layer stack | 36KB |
| `megatron/core/models/gpt/gpt_model.py` | GPT model | 34KB |
| `megatron/core/models/gpt/gpt_layer_specs.py` | Layer spec builders | 27KB |
| `megatron/core/tensor_parallel/layers.py` | TP linear layers | Large |
| `megatron/core/pipeline_parallel/schedules.py` | Pipeline schedules | 100KB |
| `megatron/training/training.py` | Training loop | 125KB |
| `megatron/training/arguments.py` | All arguments | 207KB |
| `megatron/training/checkpointing.py` | Checkpoint I/O | 83KB |

---

## 15. Common Modifications

### Adding a New Attention Variant

1. Create attention class in `megatron/core/transformer/`:

```python
class MyAttention(MegatronModule):
    def __init__(self, config, submodules, ...):
        # Initialize layers

    def forward(self, hidden_states, attention_mask, ...):
        # Your attention logic
        return output, bias
```

2. Add to layer specs in `gpt_layer_specs.py`:

```python
def get_gpt_layer_with_my_attention_spec():
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=MyAttention,
                submodules=...,
            ),
            ...
        ),
    )
```

### Adding a New Model Type

1. Create model class in `megatron/core/models/mymodel/`
2. Inherit from `LanguageModule` or `MegatronModule`
3. Create layer specs
4. Add training script in root directory

### Modifying the Training Loop

Edit `megatron/training/training.py`:
- `train_step()`: Single training iteration
- `train()`: Main training loop
- `setup_model_and_optimizer()`: Model initialization

---

## 16. The Hacker Script (`hacker_megatron_lm.py`)

Use `hacker_megatron_lm.py` for interactive exploration of Megatron modules without running full training.

### Single GPU (No Distributed)
```bash
python hacker_megatron_lm.py
```

### Multiple GPUs with Tensor Parallelism
```bash
torchrun --nproc_per_node=2 hacker_megatron_lm.py --tp-size 2
```

### Multiple GPUs with Pipeline Parallelism
```bash
torchrun --nproc_per_node=4 hacker_megatron_lm.py --pp-size 4
```

### Combined Parallelism
```bash
torchrun --nproc_per_node=8 hacker_megatron_lm.py --tp-size 2 --pp-size 4
```

The script demonstrates:
- Individual transformer components (attention, MLP, layers)
- Tensor parallel operations (scatter, gather, reduce)
- Pipeline parallel communication patterns
- Full model forward/backward passes
- Gradient inspection and verification

---

## Tips for Hackers

1. **Start with `examples/run_simple_mcore_train_loop.py`** - Minimal working example.

2. **Use `--use-cpu-initialization`** - Avoids CUDA memory during debugging.

3. **Print tensor shapes** - Add `print(f"{name}: {tensor.shape}")` to track data flow.

4. **Check process groups** - Use `parallel_state.get_*_rank()` to verify GPU assignments.

5. **Test with small models first** - `num_layers=2, hidden_size=64` is enough to verify logic.

6. **Read the specs** - Understanding `ModuleSpec` and layer specs is key to architecture.

7. **Use the debugger** - Set breakpoints in attention forward to see Q, K, V shapes.

8. **Verify gradient flow** - Register hooks to confirm gradients propagate correctly.

---

## Resources

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) - Original tensor parallelism
- [Megatron-LM v2 Paper](https://arxiv.org/abs/2104.04473) - Pipeline parallelism
- [Megatron-LM v3 Paper](https://arxiv.org/abs/2205.05198) - Sequence parallelism
- [NVIDIA Megatron-Core Docs](https://docs.nvidia.com/megatron-core/)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

---

Happy Hacking.
