<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 05 — Parallelism Vocabulary

You only need a first-order map. Details live in the
[parallelism guide](../../user-guide/parallelism-guide.md).

| Strategy | What is sharded | Typical pain if oversized |
| --- | --- | --- |
| **DP** | Batch / replica of model (with all-reduce or sharded opts) | Comm of grads; global batch too small per GPU |
| **TP** | Matrix multiplies *inside* layers | Extra collectives mid-layer; small dims |
| **PP** | Layers across stages | Pipeline bubble unless enough microbatches |
| **CP** | Sequence / context dimension | Long-context activation pressure; extra seq collectives |
| **EP** | MoE experts across ranks | Routing all-to-all |

## Interactive drill (no GPU)

Pick numbers and fill the blanks:

| knobs | value |
| --- | --- |
| GPUs | 8 |
| TP | 2 |
| PP | 2 |
| CP | 1 |
| EP | 1 |
| DP | ? |

**Expect:** \( DP = 8 / (2 \times 2 \times 1 \times 1) = 2 \) when those axes multiply cleanly.

Second drill:

| knobs | value |
| --- | --- |
| Global batch | 32 |
| Micro batch | 2 |
| DP | 4 |
| Grad accumulate steps | ? |

**Expect:** \( accum = GBS / (micro \times DP) \) when nothing else rewrites batch math.  
Check training args carefully when MoE / packing / virtual PP are on.

## Checkpoint

Match the **symptom** → **axis** (first instinct only):

1. Activations blow up as sequence length grows, even with modest model size.
2. Each GPU holds only a subset of experts.
3. Layer 1–16 on ranks 0–1, layer 17–32 on ranks 2–3.
4. QKV linear weights are column-sliced across 4 GPUs within a layer.

**Answers:** 1 CP/activation · 2 EP · 3 PP · 4 TP

## Lab — write the product on a sticky

For a job you care about (or an example script), write:

```text
WORLD = TP * PP * DP * CP * EP = ?
micro, GBS, accum = ?
```

Use real flags later in [map to production](07-map-to-production.md).

## Map

- [Parallelism guide](../../user-guide/parallelism-guide.md)
- Ultra-scale and FLOP/memory intuition: [Ultra-Scale Playbook](../../ultra-scale-playbook/index.md)
- Agent/explain path: skill `explain-model-scaling` when working with an assistant in this repo

## Next

[06 — Mini loop walkthrough](06-miniloop-walkthrough.md)
