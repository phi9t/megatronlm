<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 04 — One Layer Lab

Goal: stand up **one** `TransformerLayer`, run a forward, and read shapes.
This is the smallest honest Core unit: not a full GPT, not the training stack.

## Shapes to memorize

Megatron Core transformers commonly use **time-major** layout for layer I/O:

```text
hidden_states: [sequence_length, batch_size, hidden_size]
```

If you come from HF `[B, S, H]` conventions, convert deliberately.

## Lab — run the companion

```bash
# GPU required
python examples/tutorials/foundations/lab_runner.py layer
```

**Expect success lines** that print parameter count, input shape, and output
shape (same rank as input for a residual layer stack cell).

Optional variant:

```bash
python examples/tutorials/foundations/lab_runner.py layer --hidden-size 128 --seq-len 16 --batch-size 1
```

**Change one axis** (hidden, seq, batch) and re-predict output shape before run.

## Lab — force a failure and diagnose

```bash
# Intentionally wrong: seq-len cannot be zero
python examples/tutorials/foundations/lab_runner.py layer --seq-len 0
```

**Checkpoint:** does the error point at shape math, CUDA, distributed init, or
import? Log which and what you would fix first on a real cluster.

## Inspect (optional)

If you have a local `hack.py` (not always in every checkout), compare it to
`lab_runner.py layer`. Same idea: init dist → parallel_state → config →
spec → layer → forward.

## Checkpoint

- [ ] You can write the `TransformerLayer` I/O layout without looking.
- [ ] You know a layer is built from **config + submodules spec**, not from
      inventing new base classes for every experiment.

## Next

[05 — Parallelism vocabulary](05-parallelism-vocabulary.md)
