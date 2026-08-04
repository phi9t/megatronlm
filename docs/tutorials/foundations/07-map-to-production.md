<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 07 — Map to Production

You now have a Core-shaped mental model. Production runs still use that model—
they add **argument surfaces**, **defaults**, **logging**, **resilience**, and
**data pipelines**.

## Bridge table (mini loop → real entrypoints)

| Mini loop idea | Production home (typical) |
| --- | --- |
| `initialize_distributed` | `megatron/training/initialize.py` + torchrun env |
| `TransformerConfig` in code | Built from CLI via `megatron/training/arguments.py` |
| `model_provider` | Model providers in `pretrain_*.py` / examples |
| Data iterator mock | Preprocessed `.bin`/`.idx` + dataloaders |
| Manual Adam | Dist optimizer / FSDP / CPU offload options |
| Few iterations | `megatron/training/training.py` loop |
| Optional local ckpt | Distributed checkpointing + full job config |

## Lab — entrance exam with flags

Open an example launcher (for instance under `examples/llama/` or your team’s
YAML launcher). Fill:

```text
TP = 
PP = 
CP = 
EP = 
micro_batch =
global_batch =
sequence_length =
num_layers /
hidden_size /
num_attention_heads =
precision (bf16/fp8/...) =
```

Then answer:

1. Which core module will feel that TP most?
2. Is the global batch large enough to hide PP bubbles later?
3. Is the sequence length long enough that CP becomes the first lever?

## Inspect — pretrain entry

Open `pretrain_gpt.py` (or the model family entry your team uses) and only
search for:

- How `model_provider` is registered / passed
- How `train` / `pretrain` is invoked

Do **not** try to understand every import on the first pass.

## Checkpoint — graduation

- [ ] You can place a new feature in Core vs Training without a coin flip.
- [ ] You can run the one-layer lab and the mini loop and explain each stage.
- [ ] You can decode a launcher’s TP/PP/DP(CP/EP) product into `WORLD_SIZE`.
- [ ] You know which doc to open next for production flags.

## Keep going

| Goal | Start here |
| --- | --- |
| First cluster job | [Quickstart](../../get-started/quickstart.md) |
| Parallelism depth | [Parallelism guide](../../user-guide/parallelism-guide.md) |
| Data plane | [Data preparation](../../user-guide/data-preparation.md) |
| MoE / dist optim / offload | [Features index](../../user-guide/features/index.md) |
| Scaling intuition | [Ultra-Scale Playbook](../../ultra-scale-playbook/index.md) |
| Contribute | [Developer contribute](../../developer/contribute.md) |

## Back

[Foundations index](index.md)
