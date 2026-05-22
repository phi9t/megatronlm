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
Playbook explains why a scaling technique exists; this page shows where the same
idea appears in Megatron config, launcher code, and implementation code.

Each section follows the same path:

1. **Playbook idea:** the resource tradeoff.
2. **Megatron config:** a compact YAML or flag-style excerpt.
3. **Megatron code:** short source snippets that show where the decision is made.
4. **Source trail:** collapsible navigation for the next files to inspect.

The local Llama launcher is the easiest first anchor:
[`examples/llama/llama_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/llama_config.py)
turns YAML presets under
[`examples/llama/configs/`](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/llama/configs)
into Megatron CLI arguments. Core Megatron argument validation lives in
[`megatron/training/yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py).

## One-GPU Training and Batch Math

**Playbook idea:** start with the single-device training loop, then separate
the per-GPU microbatch from the global training batch. More accumulation raises
the global batch without increasing activation memory for one forward/backward
microbatch.

**Megatron config:**

```yaml
training:
  micro_batch_size: 1
  global_batch_size: 1
  train_iters: 2
parallelism:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
data:
  mode: mock
```

Source: [`examples/llama/configs/llama_anchor_smoke.yaml`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/configs/llama_anchor_smoke.yaml#L21-L40).

**Megatron code:**

```python
add_arg(args, "--micro-batch-size", self.micro_batch_size)
add_arg(args, "--global-batch-size", self.global_batch_size)
add_arg(args, "--train-iters", self.train_iters)
add_arg(args, "--train-samples", self.train_samples)
```

Why this matters: the Llama YAML keeps the Playbook batch symbols visible, then
renders them to the exact Megatron flags consumed by the training loop.

Source: [`TrainingConfig.to_args()`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/llama_config.py#L223-L230).

```python
if args.global_batch_size is None:
    args.global_batch_size = args.micro_batch_size * args.data_parallel_size
    if args.rank == 0:
        print('setting global batch size to {}'.format(
            args.global_batch_size), flush=True)
assert args.global_batch_size > 0
```

Why this matters: if the user does not set a global batch, Megatron defaults it
to one microbatch per data-parallel rank.

Source: [`yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py#L103-L118).

<details>
<summary>Source trail</summary>

- [`examples/llama/configs/llama_anchor_smoke.yaml`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/configs/llama_anchor_smoke.yaml)
- [`examples/llama/llama_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/llama_config.py)
- [`megatron/core/num_microbatches_calculator.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/num_microbatches_calculator.py)
- [`megatron/training/training.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py)

```text
examples/llama/
├── configs/llama_anchor_smoke.yaml
├── configs/llama3_8b_long.yaml
└── llama_config.py
```
</details>

## Data Parallelism and Gradient Accumulation

**Playbook idea:** DP replicates model state and splits the batch. Gradient
accumulation reuses the same model replica for several microbatches before an
optimizer step, so the effective batch is `micro_batch_size * data_parallel_size
* gradient_accumulation_steps`.

**Megatron config:**

```yaml
distributed:
  nproc_per_node: 8
training:
  micro_batch_size: 1
  global_batch_size: 128
parallelism:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  context_parallel_size: 1
```

Source: [`examples/llama/configs/llama3_8b_long.yaml`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/configs/llama3_8b_long.yaml#L9-L63).

**Megatron code:**

```python
model_parallel_size = args.model_parallel.pipeline_model_parallel_size * \
                      args.model_parallel.tensor_model_parallel_size
assert args.world_size % (model_parallel_size * args.model_parallel.context_parallel_size) == 0

# data_parallel_size is not in model parallel config
args.data_parallel_size = args.world_size // (model_parallel_size * args.model_parallel.context_parallel_size)
```

Why this matters: DP is the remaining degree of freedom after TP, PP, and CP
consume ranks, exactly matching the Playbook formula `DP = world_size / (TP *
PP * CP)`.

Source: [`yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py#L66-L75).

```python
if args.step_batch_size_schedule is not None and is_global_batch_size_explicitly_specified:
    raise ValueError(
        'Cannot specify both --step-batch-size-schedule and --global-batch-size'
    )
```

Why this matters: the batch schedule and a fixed global batch are mutually
exclusive, so Megatron has one source of truth for accumulation.

Source: [`yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py#L106-L114).

<details>
<summary>Source trail</summary>

- [`megatron/training/yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py)
- [`megatron/core/num_microbatches_calculator.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/num_microbatches_calculator.py)
- [`megatron/training/training.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py)

```text
megatron/training/
├── yaml_arguments.py
└── training.py
```
</details>

## Config-to-CLI Bridge

**Playbook idea:** a scaling recipe is only useful if it survives the launch
surface. In this repo, the Llama YAML files are intentionally small and the
launcher renders them to Megatron flags.

**Megatron config:**

```yaml
parallelism:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  context_parallel_size: 1
  sequence_parallel: true
  use_distributed_optimizer: true
  overlap_grad_reduce: true
  overlap_param_gather: true
```

Source: [`examples/llama/configs/llama3_8b_long.yaml`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/configs/llama3_8b_long.yaml#L56-L63).

**Megatron code:**

```python
add_arg(args, "--tensor-model-parallel-size", self.tensor_model_parallel_size)
add_arg(args, "--pipeline-model-parallel-size", self.pipeline_model_parallel_size)
add_arg(args, "--context-parallel-size", self.context_parallel_size)
add_bool_flag(args, "--sequence-parallel", self.sequence_parallel)
add_bool_flag(args, "--use-distributed-optimizer", self.use_distributed_optimizer)
```

Why this matters: the launcher keeps YAML names close to Megatron flag names,
so the config can be read as the scaling recipe.

Source: [`ParallelismConfig.to_args()`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/llama_config.py#L309-L323).

```python
if self.mode == "mock":
    args.append("--mock-data")
    add_arg(args, "--tokenizer-type", self.tokenizer_type or "NullTokenizer")
else:
    add_arg(args, "--data-path", self.data_path)
    add_arg(args, "--tokenizer-type", self.tokenizer_type or "HuggingFaceTokenizer")
```

Why this matters: mock data still exercises the real model, optimizer,
collective, logging, and checkpoint plumbing; only the token source changes.

Source: [`DataConfig.to_args()`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/llama_config.py#L353-L365).

<details>
<summary>Source trail</summary>

- [`examples/llama/llama_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/llama_config.py)
- [`examples/llama/run_llama.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/run_llama.py)
- [`examples/llama/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/README.md)

```text
examples/llama/
├── README.md
├── run_llama.py
└── llama_config.py
```
</details>

## Distributed Optimizer and ZeRO-Style Sharding

**Playbook idea:** ZeRO-style sharding reduces replicated optimizer and gradient
memory. Megatron's distributed optimizer switches gradient reduction from full
all-reduce to reduce-scatter and can overlap communication with backward
compute and later parameter gather.

**Megatron config:**

```yaml
parallelism:
  use_distributed_optimizer: true
  overlap_grad_reduce: true
  overlap_param_gather: true
training:
  grad_reduce_in_bf16: true
```

Source: [`examples/llama/configs/llama3_8b_long.yaml`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/configs/llama3_8b_long.yaml#L25-L63).

**Megatron code:**

```python
reduction_collective = (
    "reduce-scatter" if self.ddp_config.use_distributed_optimizer else "all-reduce"
)
log_single_rank(
    logger,
    logging.INFO,
    f"Using {reduction_collective} for gradient reductions because "
    f"{self.ddp_config.use_distributed_optimizer=}",
)
```

Why this matters: the common log line tells you whether the run is taking the
distributed-optimizer communication path.

Source: [`param_and_grad_buffer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/param_and_grad_buffer.py#L211-L219).

```python
if self.ddp_config.use_distributed_optimizer and not force_all_reduce:
    local_data_view = self.cached_grad_buffer_shard_list[idx][
        self.intra_distributed_optimizer_instance_rank
    ]
    grad_reduce_handle = dist_reduce_scatter_func(
        local_data_view,
        bucket.grad_data,
        op=reduce_op,
        group=communication_group,
        async_op=async_op,
    )
else:
    torch.distributed.all_reduce(
        bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op
    )
```

Why this matters: this is the reduce-scatter versus all-reduce branch that
turns the Playbook memory tradeoff into a concrete collective.

Source: [`param_and_grad_buffer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/param_and_grad_buffer.py#L617-L639).

<details>
<summary>Source trail</summary>

- [`megatron/core/distributed/param_and_grad_buffer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/param_and_grad_buffer.py)
- [`megatron/core/distributed/distributed_data_parallel.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel.py)
- [`megatron/core/distributed/distributed_data_parallel_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel_config.py)
- [`megatron/core/optimizer/distrib_optimizer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py)

```text
megatron/core/distributed/
├── distributed_data_parallel.py
├── distributed_data_parallel_config.py
└── param_and_grad_buffer.py
```
</details>

## Megatron-FSDP and ZeRO-3-Style Sharding

**Playbook idea:** FSDP/ZeRO-3 shards parameters as well as gradients and
optimizer state. That lowers model-state memory but adds parameter all-gather
and sharding lifecycle work around forward and backward.

**Megatron config:**

```yaml
use_megatron_fsdp: true
data_parallel_sharding_strategy: optim_grads_params
ckpt_format: fsdp_dtensor
init_model_with_meta_device: true
```

Source: [Parallelism Strategies Guide](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/parallelism-guide.md#L40-L57).

**Megatron code:**

```python
def fully_shard_model(
    module: torch.nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    dp_shard_dim: Optional[str] = None,
    dp_outer_dim: Optional[str] = None,
    tp_dim: Optional[str] = None,
    zero_dp_strategy: str | int = 3,
    outer_dp_sharding_strategy: str | int = 0,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
) -> torch.nn.Module:
```

Why this matters: the public FSDP surface exposes the sharding strategy,
hybrid-sharding dimensions, and overlap switches directly.

Source: [`fully_shard.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py#L75-L108).

```python
ddp_config = DistributedDataParallelConfig(
    data_parallel_sharding_strategy=zero_dp_strategy,
    outer_dp_sharding_strategy=outer_dp_sharding_strategy,
    overlap_grad_reduce=overlap_grad_reduce,
    overlap_param_gather=overlap_param_gather,
)
```

Why this matters: Megatron-FSDP normalizes the high-level sharding recipe into
the same DDP config object used by the lower communication layers.

Source: [`fully_shard.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py#L350-L356).

<details>
<summary>Source trail</summary>

- [`docs/user-guide/features/megatron_fsdp.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/megatron_fsdp.md)
- [`megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py)
- [`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py)
- [`megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/fsdp/mcore_fsdp_adapter.py)

```text
megatron/core/distributed/fsdp/
├── mcore_fsdp_adapter.py
└── src/megatron_fsdp/
    ├── fully_shard.py
    └── megatron_fsdp.py
```
</details>

## Tensor Parallelism and Sequence Parallelism

**Playbook idea:** TP splits large layer matrix operations across ranks. SP
then avoids replicating some sequence-dimension activations across those same
TP ranks, reducing activation memory at the cost of extra gather/scatter
collectives.

**Megatron config:**

```yaml
parallelism:
  tensor_model_parallel_size: 2
  sequence_parallel: true
```

**Megatron code:**

```python
if args.model_parallel.tensor_model_parallel_size == 1:
    args.model_parallel.sequence_parallel = False

if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
    if args.model_parallel.sequence_parallel:
        raise RuntimeError(
            "Using sequence parallelism requires setting the environment variable "
            "CUDA_DEVICE_MAX_CONNECTIONS to 1")
```

Why this matters: SP is meaningful only with TP, and Megatron enforces the CUDA
connection setting needed by this communication pattern.

Source: [`yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py#L315-L325).

```python
def scatter_to_sequence_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: split, backward: AG <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _ScatterToSequenceParallelRegion.apply(input_, group)

def gather_from_sequence_parallel_region(
    input_,
    tensor_parallel_output_grad=True,
    group=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
```

Why this matters: the TP/SP implementation is a set of autograd-aware
collective wrappers, not only a launcher flag.

Source: [`mappings.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py#L512-L526).

<details>
<summary>Source trail</summary>

- [`docs/api-guide/core/tensor_parallel.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/core/tensor_parallel.md)
- [`megatron/core/tensor_parallel/layers.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py)
- [`megatron/core/tensor_parallel/mappings.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py)
- [`megatron/training/yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py)

```text
megatron/core/tensor_parallel/
├── layers.py
└── mappings.py
```
</details>

## Context Parallelism

**Playbook idea:** CP splits long sequence activations across ranks. It targets
long-context memory and attention communication, and it also reduces the ranks
left for ordinary DP.

**Megatron config:**

```yaml
parallelism:
  context_parallel_size: 2
```

**Megatron code:**

```python
model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size

if world_size % model_size != 0:
    raise RuntimeError(f"world_size ({world_size}) is not divisible by {model_size}")
```

Why this matters: CP participates in the same rank-product check as TP and PP,
so it changes both sequence placement and the derived DP size.

Source: [`parallel_state.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L728-L736).

```python
for ranks in decoder_rank_generator.get_ranks('cp'):
    group = create_group(
        ranks,
        timeout=timeout,
        pg_options=get_nccl_options("cp", nccl_comm_cfgs),
        group_desc="CONTEXT_PARALLEL_GROUP",
    )
```

Why this matters: CP is represented as explicit process groups that the
attention stack can use for context exchange.

Source: [`parallel_state.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L953-L966).

<details>
<summary>Source trail</summary>

- [`docs/user-guide/features/context_parallel.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/context_parallel.md)
- [`megatron/core/parallel_state.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py)
- [`megatron/core/transformer/attention.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py)

```text
megatron/core/
├── parallel_state.py
└── transformer/attention.py
```
</details>

## Pipeline Parallelism

**Playbook idea:** PP splits model depth across ranks. It trades activation and
parameter memory for pipeline bubbles and point-to-point activation transfers;
virtual pipeline stages can reduce bubbles when there are enough microbatches.

**Megatron config:**

```yaml
parallelism:
  pipeline_model_parallel_size: 4
num_layers_per_virtual_pipeline_stage: 2
```

**Megatron code:**

```python
if args.num_layers_per_virtual_pipeline_stage is not None:
    assert args.model_parallel.pipeline_model_parallel_size > 2
    assert args.language_model.num_layers % args.model_parallel.transformer_pipeline_model_parallel_size == 0
    num_layers_per_pipeline_stage = args.language_model.num_layers // args.model_parallel.transformer_pipeline_model_parallel_size
    args.model_parallel.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
        args.num_layers_per_virtual_pipeline_stage
else:
    args.model_parallel.virtual_pipeline_model_parallel_size = None
```

Why this matters: virtual PP is derived from layer count, PP size, and the
per-virtual-stage layer target rather than being an independent knob.

Source: [`yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py#L129-L142).

```python
if pp_size > 1:
    if vp_size is not None:
        forward_backward_func = forward_backward_pipelining_with_interleaving
    else:
        forward_backward_func = forward_backward_pipelining_without_interleaving
else:
    forward_backward_func = forward_backward_no_pipelining
```

Why this matters: the schedule switches from ordinary training to 1F1B
pipelining as soon as PP is greater than one, with interleaving selected by
virtual PP.

Source: [`schedules.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py#L141-L153).

<details>
<summary>Source trail</summary>

- [`docs/user-guide/features/pipeline_parallel_layout.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/pipeline_parallel_layout.md)
- [`megatron/core/pipeline_parallel/schedules.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)
- [`megatron/core/transformer/pipeline_parallel_layer_layout.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/pipeline_parallel_layer_layout.py)

```text
megatron/core/pipeline_parallel/
└── schedules.py
megatron/core/transformer/
└── pipeline_parallel_layer_layout.py
```
</details>

## Expert Parallelism and MoE

**Playbook idea:** EP shards experts instead of dense layers. It saves expert
parameter and compute memory per rank, but routing, all-to-all traffic, and TP
interactions add constraints.

**Megatron config:**

```yaml
model_parallel:
  tensor_model_parallel_size: 2
  expert_model_parallel_size: 2
  sequence_parallel: true
language_model:
  num_moe_experts: 8
  moe_grouped_gemm: true
  moe_router_topk: 2
```

**Megatron code:**

```python
if args.language_model.num_moe_experts is not None:
    if args.model_parallel.tensor_model_parallel_size > 1:
        assert args.model_parallel.sequence_parallel, \
            "When using MoE and tensor parallelism, sequence parallelism must be used."

if args.model_parallel.expert_model_parallel_size  > 1:
    assert args.language_model.num_moe_experts is not None
    assert args.language_model.num_moe_experts % args.model_parallel.expert_model_parallel_size == 0
```

Why this matters: Megatron validates the common Playbook caveat that TP+EP
needs SP and that experts must divide cleanly across EP ranks.

Source: [`yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py#L327-L337).

<details>
<summary>Source trail</summary>

- [`docs/user-guide/features/moe.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/moe.md)
- [`megatron/core/transformer/moe/`](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe)
- [`megatron/training/yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py)

```text
megatron/core/transformer/moe/
├── router.py
├── token_dispatcher.py
└── experts.py
```
</details>

## Activation Recompute

**Playbook idea:** activation recomputation trades extra forward compute for
lower activation memory by not storing every intermediate tensor.

**Megatron config:**

```yaml
language_model:
  recompute_granularity: full
  recompute_method: uniform
  recompute_num_layers: 4
  distribute_saved_activations: true
```

**Megatron code:**

```python
if args.language_model.distribute_saved_activations:
    assert args.model_parallel.tensor_model_parallel_size > 1
    assert args.language_model.recompute_granularity == 'full'
    assert args.language_model.recompute_method is not None

if args.language_model.recompute_granularity == 'selective':
    assert args.language_model.recompute_method is None
```

Why this matters: distributed saved activations are only valid with full
recompute and TP, while selective recompute is a separate mode.

Source: [`yaml_arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/yaml_arguments.py#L294-L313).

```python
if self.config.recompute_granularity == 'full' and self.training:
    checkpointed_result = checkpointed_forward(
        self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        context=context,
        context_mask=context_mask,
    )
```

Why this matters: the transformer block enters the checkpointed forward path
only when full recomputation is selected during training.

Source: [`transformer_block.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py#L619-L635).

<details>
<summary>Source trail</summary>

- [`megatron/core/transformer/transformer_block.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py)
- [`megatron/core/recompute/`](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/recompute)
- [`megatron/training/theoretical_memory_usage.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/theoretical_memory_usage.py)

```text
megatron/core/
├── recompute/
└── transformer/transformer_block.py
```
</details>

## FP8

**Playbook idea:** FP8 reduces activation, parameter, and communication volume
where Transformer Engine supports the recipe. FP8 parameter gather extends that
idea into the distributed optimizer path by gathering lower-precision model
parameters while retaining master weights for optimization.

**Megatron config:**

```yaml
fp8:
  enabled: true
  format: hybrid
  amax_history_len: 1024
  amax_compute_algo: max
  param_gather: true
```

Source: [`examples/llama/configs/llama3_8b_long.yaml`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/configs/llama3_8b_long.yaml#L49-L54).

**Megatron code:**

```python
if not self.enabled:
    return []
args: list[str] = []
add_arg(args, "--fp8-format", self.format)
add_arg(args, "--fp8-amax-history-len", self.amax_history_len)
add_arg(args, "--fp8-amax-compute-algo", self.amax_compute_algo)
add_bool_flag(args, "--fp8-param-gather", self.param_gather)
```

Why this matters: the Llama launcher keeps FP8 disabled by default and emits
the Transformer Engine recipe flags only when the YAML explicitly enables it.

Source: [`Fp8Config.to_args()`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/llama_config.py#L285-L295).

```python
if self.ddp_config.fp8_param_gather:
    fp8_params, shard_fp32_from_fp8, shard_offsets_in_fp8 = (
        self._get_fp8_params_and_shard_fp32_from_fp8()
    )
    ...
    quantize_param_shard(
        expanded_fp8_params,
        expanded_shard_fp32_from_fp8,
        expanded_shard_offsets_in_fp8,
        self.data_parallel_group,
    )
```

Why this matters: the distributed optimizer casts FP32 master shards back to FP8
model parameters before the parameter-gather path needs them.

Source: [`distrib_optimizer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py#L2730-L2757).

<details>
<summary>Source trail</summary>

- [`megatron/core/fp8_utils.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fp8_utils.py)
- [`megatron/core/optimizer/distrib_optimizer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py)
- [`examples/llama/configs/llama3_8b_long.yaml`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/configs/llama3_8b_long.yaml)

```text
megatron/core/
├── fp8_utils.py
└── optimizer/distrib_optimizer.py
```
</details>

## Profiling, Throughput, and Common Logs

**Playbook idea:** scaling decisions should be checked against throughput,
iteration time, memory pressure, and communication overlap. Megatron exposes
these through stdout, TensorBoard, profiler traces, and progress logs.

**Megatron config:**

```yaml
logging:
  log_interval: 1
  log_throughput: true
  profile: true
  profile_step_start: 4
  profile_step_end: 6
  ckpt_format: torch_dist
```

Source: [`examples/llama/configs/llama3_8b_long.yaml`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/configs/llama3_8b_long.yaml#L71-L80).

**Megatron code:**

```python
throughput = num_floating_point_operations(
    args,
    batch_size,
    seqlen_squared_sum_in_batch=seqlen_squared_sum_in_batch,
    total_real_tokens_in_batch=total_real_tokens_in_batch,
) / (
    elapsed_time_per_iteration * 10**12 * args.world_size
)
...
if args.log_throughput:
    log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
```

Why this matters: Megatron reports per-GPU TFLOP/s from the same iteration
timing and model-shape terms that drive Playbook MFU/HFU reasoning.

Source: [`training.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L2584-L2616).

```python
prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=max(args.profile_step_start - 1, 0),
        warmup=1 if args.profile_step_start > 0 else 0,
        active=args.profile_step_end - args.profile_step_start,
        repeat=1,
    ),
    on_trace_ready=trace_handler,
)
```

Why this matters: the profile window in YAML maps directly to the PyTorch
profiler schedule, so short windows can capture steady-state steps.

Source: [`training.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L3321-L3331).

### Log Lines to Recognize

`Using reduce-scatter for gradient reductions because self.ddp_config.use_distributed_optimizer=True`
: The run is using the distributed optimizer path. Gradients are reduced and
  sharded instead of fully all-reduced on every data-parallel rank.

`setting training iterations to X`
: The run is sample-scheduled. For GPT-style pretraining this usually comes
  from `train_samples // global_batch_size`, with token work per step equal to
  `global_batch_size * seq_length`.

`MockGPTDataset`
: The training loop is real, including forward, backward, optimizer,
  collectives, logging, and checkpoint plumbing. The token stream is synthetic
  instead of read from a data prefix.

<details>
<summary>Source trail</summary>

- [`examples/llama/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/README.md)
- [`megatron/training/training.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py)
- [`megatron/training/theoretical_memory_usage.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/theoretical_memory_usage.py)

```text
megatron/training/
├── training.py
└── theoretical_memory_usage.py
```
</details>

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
- `--log-throughput` and `--profile-step-*` make the throughput and trace
  evidence visible for scaling comparisons.

For the smallest local proof, use:

```bash
mise run llama-anchor-smoke
```

The model is tiny and uses mock data, but it exercises the same Megatron
training pipeline that the larger run uses.
