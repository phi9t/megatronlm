<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Foundations Lab

An **interactive** path through Megatron Core and Megatron-LM fundamentals.
Read less in one sitting; **stop at each lab**, change one thing, and confirm
what you observe before continuing.

These pages assume you have completed [installation](../../get-started/install.md).
You do not need a multi-node cluster: most labs run on **1 GPU**; the mini loop
shows a **2-GPU** optional path.

## How this lab works

| Marker | Meaning |
| --- | --- |
| **Checkpoint** | Quiet self-check: answer mentally before scrolling |
| **Lab** | Command or code change you should run |
| **Inspect** | Open a specific file and answer the prompt |
| **Map** | Links into deeper references after the idea is solid |

Do not treat this as documentation of every flag. For full flags and production
recipes, use the [user guide](../../user-guide/index.md) and
[parallelism guide](../../user-guide/parallelism-guide.md).

## Path

```{toctree}
:maxdepth: 1

00-how-to-learn
01-two-layer-stack
02-parallel-state
03-config-and-specs
04-one-layer-lab
05-parallelism-vocabulary
06-miniloop-walkthrough
07-map-to-production
```

Suggested pace: **one module per session**. If you rush modules 05–07 before
feeling 01–04, the production entrypoints will look like magic.

## Companion script

From the repository root (GPU required for the layer lab):

```bash
python examples/tutorials/foundations/lab_runner.py --list
python examples/tutorials/foundations/lab_runner.py layer
```

## Related material

- [Overview](../../get-started/overview.md) — product map (Core vs LM vs Bridge)
- [Quickstart](../../get-started/quickstart.md) — first training commands
- [Ultra-Scale Playbook index](../../ultra-scale-playbook/index.md) — scaling vocabulary
- Repo-root `HACKERS_GUIDE.md` — engineer-focused navigation notes (local / fork content when present)
