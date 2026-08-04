<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 02 — Parallel State

Before a model module is constructed, ranks must agree on **process groups**:
who shares tensor-parallel shards, who is on the same pipeline stage, who
participates in data-parallel reduces.

Core API of record: `megatron.core.parallel_state`.

Training jobs usually call this after `torch.distributed.init_process_group`,
often via `megatron/training/initialize.py`. The mini loop does both itself.

## Vocabulary (minimal)

| Symbol | Idea |
| --- | --- |
| `WORLD_SIZE` | Total ranks in the process group |
| `RANK` | Global process id |
| `LOCAL_RANK` | GPU index on this node (for `cuda.set_device`) |
| `TP` | Tensor model parallel size (shards *inside* layers) |
| `PP` | Pipeline model parallel size (shards *across* layers) |
| Remaining | Data parallel (and later CP/EP complicate the product) |

Rough product (dense, no CP/EP):  
`WORLD_SIZE ≈ TP × PP × DP`.

## Checkpoint — predict before you run

On **2 GPUs**, `torchrun --nproc_per_node=2`, with `TP=1`, `PP=1`:

- What is each rank’s `WORLD_SIZE`?
- How many data-parallel ranks?

**Expect:** world size 2, DP size 2, TP/PP size 1.

## Lab — inspect init in the mini loop

Open `examples/run_simple_mcore_train_loop.py` and find `initialize_distributed`.

```text
Inspect prompts
1. Which env vars does it read?
2. Which two integers does it pass to initialize_model_parallel?
3. Who destroys prior parallel state, and why might that matter if you re-run
   in the same process?
```

Optional run (needs 2 GPUs) once you finish module 06:

```bash
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

For a **1-GPU** poke path today, use the lab runner after module 04:

```bash
python examples/tutorials/foundations/lab_runner.py parallel-state
```

It prints rank world size and parallel sizes from `parallel_state`.

## Checkpoint

- [ ] You can expand `WORLD_SIZE ≈ TP × PP × DP` and point to where TP/PP are set.
- [ ] You know process groups are **global process state**, not attributes of a
      random nn.Module you forgot to pass in (the library is moving toward
      explicit groups—see AGENTS.md process-group guidance for Core production code).

## Map

- Full strategies: [Parallelism guide](../../user-guide/parallelism-guide.md)
- Scaling narratives: [Ultra-Scale Playbook](../../ultra-scale-playbook/index.md)

## Next

[03 — Config and specs](03-config-and-specs.md)
