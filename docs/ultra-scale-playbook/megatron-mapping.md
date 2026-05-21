<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Ultra-Scale Playbook to Megatron-LM Mapping

Use this page as the Megatron-LM companion to the Ultra-Scale Playbook. The
Playbook explains the scaling mechanism; this page points to the Megatron flags,
docs, examples, and implementation paths that exercise the same mechanism.

## Reading Strategy

For each concept, connect four layers:

1. The Playbook mechanism: what resource is being traded.
2. The launch surface: which Megatron flag or YAML field enables it.
3. The user-facing example: where a concrete run config uses it.
4. The implementation path: where to inspect the code.

The local Llama examples are the best first anchor:

- `examples/llama/README.md`
- `examples/llama/configs/llama_anchor*.yaml`
- `examples/llama/configs/llama3_8b*.yaml`
- `examples/llama/llama_config.py`

## Concept Map

| Playbook concept | Megatron-LM surface | Main code paths |
| --- | --- | --- |
| Data parallelism | `--data-parallel-sharding-strategy no_shard`, implicit `DP = world_size / (TP * PP * CP)` | `megatron/training/yaml_arguments.py`, `megatron/training/initialize.py`, `megatron/core/distributed/` |
| Distributed optimizer / ZeRO-style sharding | `--use-distributed-optimizer`, `--overlap-grad-reduce`, `--overlap-param-gather` | `megatron/core/distributed/param_and_grad_buffer.py`, `megatron/core/optimizer/distrib_optimizer.py`, `docs/user-guide/features/dist_optimizer.md` |
| Megatron-FSDP / ZeRO-3-style parameter sharding | `--use-megatron-fsdp`, `--data-parallel-sharding-strategy optim_grads_params` | `megatron/core/distributed/fsdp/`, `docs/user-guide/features/megatron_fsdp.md` |
| Tensor parallelism | `--tensor-model-parallel-size`, `parallelism.tensor_model_parallel_size` | `megatron/core/tensor_parallel/layers.py`, `megatron/core/tensor_parallel/mappings.py`, `docs/api-guide/core/tensor_parallel.md` |
| Sequence parallelism | `--sequence-parallel` | `megatron/core/tensor_parallel/`, `megatron/training/yaml_arguments.py` |
| Context parallelism | `--context-parallel-size`, `--cp-comm-type` | `megatron/core/parallel_state.py`, `megatron/core/transformer/attention.py`, `docs/user-guide/features/context_parallel.md` |
| Pipeline parallelism | `--pipeline-model-parallel-size`, `--num-layers-per-virtual-pipeline-stage` | `megatron/core/pipeline_parallel/schedules.py`, `megatron/core/transformer/pipeline_parallel_layer_layout.py`, `docs/user-guide/features/pipeline_parallel_layout.md` |
| Expert parallelism / MoE | `--expert-model-parallel-size`, `--num-experts`, `--moe-grouped-gemm`, `--sequence-parallel` with TP+EP | `megatron/core/transformer/moe/`, `docs/user-guide/features/moe.md` |
| Activation recomputation | `--recompute-granularity`, `--recompute-method`, `--recompute-num-layers` | `megatron/core/transformer/transformer_block.py`, `megatron/training/yaml_arguments.py`, `megatron/training/theoretical_memory_usage.py` |
| FP8 training | `--fp8-format hybrid`, `--fp8-amax-history-len`, `--fp8-amax-compute-algo`, `--fp8-param-gather` | `megatron/core/fp8_utils.py`, `megatron/core/optimizer/distrib_optimizer.py`, `examples/llama/configs/llama3_8b_long.yaml` |
| Gradient accumulation | `--micro-batch-size`, `--global-batch-size`; `GAS = GBS / (MBS * DP)` when divisible | `megatron/core/num_microbatches_calculator.py`, `megatron/training/yaml_arguments.py` |
| Memory accounting | Model, gradients, optimizer state, master weights, activations, buffers | `megatron/training/theoretical_memory_usage.py`, `docs/user-guide/parallelism-guide.md` |
| Profiling and throughput | profiler output, TensorBoard logs, tokens/sec, MFU/HFU-style interpretation | `megatron/training/training.py`, `examples/llama/README.md` |

## Common Log Lines

`Using reduce-scatter for gradient reductions because self.ddp_config.use_distributed_optimizer=True`
: The run is using the distributed optimizer path. Gradients are reduced and
  sharded instead of fully all-reduced on every data-parallel rank. Inspect
  `megatron/core/distributed/param_and_grad_buffer.py`.

`setting training iterations to X`
: The run is sample-scheduled. For GPT-style pretraining this usually comes
  from `train_samples // global_batch_size`, with token work per step equal to
  `global_batch_size * seq_length`.

`MockGPTDataset`
: The training loop is real, including forward, backward, optimizer,
  collectives, logging, and checkpoint plumbing. The token stream is synthetic
  instead of read from a data prefix.

## Local Example Walkthrough

The Llama 8B long-run preset is a compact way to connect the Playbook concepts
to a concrete Megatron launch:

```bash
mise run llama3-8b-long-dry-run
```

Read the generated command and map it as follows:

- `--tensor-model-parallel-size` controls whether attention heads and MLP
  matrix multiplications are sharded across tensor-parallel ranks.
- `--context-parallel-size` controls whether long sequence tensors are split
  across context-parallel ranks.
- `--sequence-parallel` reduces activation replication when tensor parallelism
  is active.
- `--use-distributed-optimizer` switches gradient synchronization to a
  reduce-scatter based optimizer-state sharding path.
- `--overlap-grad-reduce` and `--overlap-param-gather` attempt to hide
  communication behind backward and forward compute.
- `--fp8-*` flags select the FP8 recipe and parameter-gather behavior.

For the smallest local proof, use:

```bash
mise run llama-anchor-smoke
```

The model is tiny and uses mock data, but it exercises the same Megatron
training pipeline that the larger run uses.
