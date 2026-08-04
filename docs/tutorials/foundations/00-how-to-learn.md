<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 00 — How to Learn Here

This series is designed for **active** reading. Each later module assumes you
did the **Lab** steps of the previous ones.

## Learning contract

1. **One idea, then a lab.** Prefer a 15-minute loop over a two-hour skim.
2. **Predict before run.** Write the shapes / ranks / groups you expect; only
   then launch a command.
3. **Change one axis.** When experimenting, vary only TP, only batch, or only
   layer count—not three things at once.
4. **Prefer path maps over memorization.** If a name is blurry, return to the
   tree and name which **layer of the stack** owns it (Core vs Training).

## Environment for labs

- Editable install from the repo root (see [install](../../get-started/install.md)).
- At least one CUDA GPU for layer and mini-loop labs.
- From repo root, use relative paths only (for example
  `examples/run_simple_mcore_train_loop.py`).

## Checkpoint

Before module 01:

- [ ] You can open this repo root and list `megatron/core`, `megatron/training`,
      and `examples/`.
- [ ] `python -c "import megatron.core; print(megatron.core.__file__)"` works
      (or your environment docs show the expected failure until install).

## Next

[01 — Two-layer stack](01-two-layer-stack.md)
