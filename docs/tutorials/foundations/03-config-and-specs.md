<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 03 — Config and Specs

Almost every Core module is parameterized by **`TransformerConfig`**.  
How the layer is *wired* (local PyTorch vs Transformer Engine, fused ops, MoE
blocks, …) is selected by a **layer / model specification** (`ModuleSpec` /
layer specs), not by free-form subclass soup.

| Piece | Role |
| --- | --- |
| `TransformerConfig` | Sizes, dtypes, features flags (`num_layers`, `hidden_size`, heads, …) |
| Layer spec (`get_gpt_layer_local_spec()`, …) | Which submodule classes fill attention / MLP slots |
| Model (`GPTModel`) | Stack of layers + embeddings / head under one config |

## Checkpoint

Without looking at code: if you change **hidden size**, do you edit the spec
or the config? If you swap **local Attention for a TE Attention**, config or
spec?

**Expect:** size → config · implementation choice of submodule → spec.

## Lab — read only three symbols

1. Open `megatron/core/transformer/transformer_config.py` and find the
   `TransformerConfig` class definition (fields will be numerous—don’t memorize).
2. Open `megatron/core/models/gpt/gpt_layer_specs.py` and locate
   `get_gpt_layer_local_spec`.
3. Open `examples/run_simple_mcore_train_loop.py` `model_provider` and match
   config fields to the model constructor.

**Write the mapping for the mini model:**

| Field in config | Value in mini loop | Why that value is ok for a demo |
| --- | --- | --- |
| `num_layers` | ? | ? |
| `hidden_size` | ? | ? |
| `num_attention_heads` | ? | ? |

## Inspect — vocab and sequence

In the same mini loop, note:

- `vocab_size=100` on `GPTModel` (demo only—not a real tokenizer vocab)
- `_SEQUENCE_LENGTH = 64`

**Checkpoint:** can you explain why a tiny vocab is fine for **shape testing**
but useless for language quality?

## Map

- Spec system notes appear in engineer docs / HACKERS_GUIDE when present
- Production models: [models index](../../models/index.md)

## Next

[04 — One layer lab](04-one-layer-lab.md)
