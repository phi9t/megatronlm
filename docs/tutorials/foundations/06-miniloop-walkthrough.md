<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 06 — Mini Loop Walkthrough

The mini loop is the best **interactive map** of Core training without the full
`pretrain_gpt.py` surface area.

Entry: `examples/run_simple_mcore_train_loop.py`

## Stages (read the file top → bottom)

| Stage | Function(s) | Idea |
| --- | --- | --- |
| 1 | `initialize_distributed` | torch.distributed + `parallel_state` |
| 2 | `model_provider` | Tiny `GPTModel` via config + local layer spec |
| 3 | `get_train_data_iterator` | Mock GPT dataset → dataloader → iterator |
| 4 | Wrap model | DDP / distributed config |
| 5 | `forward_step_func` + schedule | Pipeline-aware forward-backward helper |
| 6 | Optimizer + train iterations | Adam, loss, optional checkpoint |
| 7 | Cleanup | Destroy process groups |

## Checkpoint — order

Without scrolling: can you put these in order?

- `finalize_model_grads` / grad sync behavior
- Construct optimizer
- Forward-backward with schedule
- Init parallel state
- Build model

**Expect:** init parallel → model (+DDP) → data → optimizer → forward-backward → grads finalize as designed by the loop.

## Lab — run it

From [quickstart](../../get-started/quickstart.md):

```bash
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

If you only have one GPU, try:

```bash
torchrun --nproc_per_node=1 examples/run_simple_mcore_train_loop.py
```

**Observe and write down:**

1. Did ranks print in lockstep or interleave?
2. Which step failed first if helpers need compile (`compile_helpers`)?
3. Did a checkpoint path get written?

## Lab — kill one stage mentally

Pick **one** removal and predict the error class:

| Experiment (mental) | Expected failure class |
| --- | --- |
| Skip `initialize_model_parallel` | Distributed / parallel group |
| Zero micro batches / empty iterator | Data / hang |
| Wrong dtype on config vs activations | Runtime tensor dtype |

## Inspect — schedules

Locate the import:

```python
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
```

**Checkpoint:** even with `PP=1`, Core often still uses a **schedule abstraction**
so the same loop grows into pipelined execution later.

## Map

- Full training recipes: [training examples](../../user-guide/training-examples.md)
- Data plane: [data preparation](../../user-guide/data-preparation.md)

## Next

[07 — Map to production](07-map-to-production.md)
