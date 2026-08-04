<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 01 — Two-Layer Stack

Megatron has two mental piles. Mixing them is the main source of "I changed
the wrong file" thrash.

| Layer | Lives in | Owns | Change here when… |
| --- | --- | --- | --- |
| **Megatron Core** | `megatron/core/` | Layers, models, parallel groups, dist optimizers, dist checkpointing | The math, sharding, or module graph changes |
| **Megatron-LM training** | `megatron/training/`, top-level `pretrain_*.py` | Args, init, data plumbing, loggers, the big train loop | The *job* changes: flags, checkpoint cadence, logging |

**Rule of thumb:** layer math → Core. CLI / loop orchestration → Training.

Read the product map once: [Overview](../../get-started/overview.md).

## Checkpoint

Say out loud which layer owns each item:

1. `TransformerLayer` forward
2. `--global-batch-size` parsing
3. Tensor-parallel linear splits
4. TensorBoard / WandB logging helpers
5. `finalize_model_grads` after backward

**Answers:** 1 Core · 2 Training · 3 Core · 4 Training · 5 Core (called from training loops).

## Lab — walk the tree without reading everything

```bash
# From repository root
ls megatron/core
ls megatron/training
ls examples | head
```

**Inspect** — open these files only long enough to read the first docstring /
class name:

- `megatron/core/transformer/transformer_layer.py`
- `megatron/training/training.py` (scroll for `pretrain` / train-step names only)
- `examples/run_simple_mcore_train_loop.py` (full file is short — skim later in module 06)

## Checkpoint

- [ ] You can explain MCore vs Training to someone without opening a browser.
- [ ] You know `examples/run_simple_mcore_train_loop.py` is a **Core-centric**
      mini app (not the full `pretrain_gpt.py` stack).

## Map

- Deep engineering path: repo-root `HACKERS_GUIDE.md` (when present on your checkout)
- Production walkthrough after foundations: [Quickstart](../../get-started/quickstart.md)

## Next

[02 — Parallel state](02-parallel-state.md)
