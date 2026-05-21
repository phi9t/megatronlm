# Ultra-Scale Playbook Notes

Source: https://huggingface.co/spaces/nanotron/ultrascale-playbook

These notes distill the GPU-cluster training mechanisms most useful for
explaining Megatron-LM runs. The playbook's practical frame is: fit the step in
memory, hit the intended global batch size, then optimize throughput by changing
parallelism and overlap.

## Table of Contents

- Core framing
- Single-GPU training anatomy
- Memory accounting
- Gradient accumulation and batch terminology
- Profiling
- Data parallelism and overlap
- ZeRO stages
- Tensor and sequence parallelism
- Context parallelism
- Pipeline parallelism
- Expert parallelism
- Configuration selection
- Megatron-LM log mapping

## Core Framing

Every scaling technique addresses one or more of:

- Memory usage: a hard feasibility constraint.
- Compute efficiency: keep GPUs doing useful math.
- Communication overhead: avoid idle time from data movement and synchronization.

Most techniques trade between those resources:

- Activation recomputation saves memory but adds compute.
- Tensor parallelism saves per-GPU memory/compute but adds collectives.
- ZeRO/FSDP saves duplicated optimizer/parameter memory but adds parameter
  gathers and gradient reduce-scatters.
- Pipeline parallelism saves layer memory but introduces bubbles and scheduling
  constraints.
- Context parallelism enables longer sequences but adds attention communication.

## Single-GPU Training Anatomy

A basic training step:

1. Forward pass computes logits/loss and stores activations needed by backward.
2. Backward pass computes gradients and frees many activations as it goes.
3. Optimizer step updates parameters and optimizer states.

The first step can have different memory behavior because CUDA/PyTorch allocators
and optimizer states are initialized lazily. A run that fits step 0 can still OOM
on later steps when optimizer state appears.

## Memory Accounting

Training memory categories:

- Model weights.
- Gradients.
- Optimizer states.
- Master weights or FP32 copies when used.
- Activations retained for backward.
- CUDA kernels, temporary buffers, allocator fragmentation, profiler buffers.

Mixed-precision Adam-style rough accounting:

```text
BF16 params:          2 bytes/param
BF16 grads:           2 bytes/param
FP32 master weights:  4 bytes/param, when maintained
Adam first moment:    4 bytes/param
Adam second moment:   4 bytes/param
FP32 grad accum:      +4 bytes/param, when gradients accumulate in FP32
```

Activation memory scales with:

```text
layers * micro_batch_size * sequence_length * hidden_size
```

Attention activations include sequence-squared terms unless the kernel avoids
materializing the full attention matrix. FlashAttention effectively performs
selective recomputation by not storing the full attention matrix.

Activation recomputation:

- Full recompute stores only major boundaries and recomputes more forward work
  during backward.
- Selective recompute targets large, cheap-to-recompute activations, often in
  attention.
- Report both memory savings and compute/time overhead. A higher HFU from doing
  extra work is not necessarily a better end-to-end training run.

MFU vs HFU:

- MFU: useful model FLOPs divided by available hardware FLOPs over time.
- HFU: actual hardware FLOPs, including recomputation, divided by available
  hardware FLOPs over time.
- Use MFU for end-to-end model training efficiency comparisons; use HFU when
  diagnosing whether hardware is busy.

## Gradient Accumulation and Batch Terminology

The playbook distinguishes:

```text
MBS = micro batch size per forward/backward pass
GAS = gradient accumulation steps
DP  = data parallel replicas
GBS = global batch size per optimizer step
```

Common equation:

```text
GBS = MBS * GAS * DP
```

For language models, batch may be reported as tokens:

```text
global_tokens_per_step = GBS * sequence_length
```

Megatron nuance:

- Megatron's `--global-batch-size` is in samples/sequences for GPT pretraining.
- Token batch is `global_batch_size * seq_length`.
- `train_samples` divided by `global_batch_size` gives optimizer iterations for
  sample-based schedules.
- If sequence packing or curriculum schedules are enabled, simple equations may
  need adjustment.

Gradient accumulation:

- Reduces activation memory by running smaller microbatches.
- Increases the number of forward/backward passes per optimizer step.
- Can reduce communication frequency if gradients are synchronized once per
  accumulated global batch.

## Profiling

Use profiler traces to check:

- CPU launch overhead and kernel gaps.
- Compute streams vs communication streams.
- Whether gradient synchronization overlaps backward kernels.
- Whether parameter all-gather overlaps forward compute.
- Memory peaks across forward, backward, optimizer, evaluation, checkpointing,
  and profiling windows.

Useful diagnosis patterns:

- Long idle GPU regions: scheduling, CPU launch, data loading, or sync waits.
- Communication after all backward compute: overlap failed or buckets too large.
- Many tiny collectives: bucket sizing, TP/CP granularity, or latency-bound
  communication.
- OOM after first step: optimizer state or allocator behavior.
- Profiled steps slower than normal: profiler overhead and trace export cost.

## Data Parallelism and Overlap

Data parallelism replicates model weights, processes different samples on each
rank, then synchronizes gradients.

Naive DP:

- Forward/backward local to each rank.
- Full gradient AllReduce before optimizer step.
- Parameters, gradients, and optimizer states are duplicated on every DP rank.

Optimized DP:

- Overlap gradient synchronization with backward pass.
- Bucket gradients so a bucket can synchronize as soon as all gradients in that
  bucket are ready.
- Avoid synchronizing every microbatch when using gradient accumulation; sync at
  the accumulation boundary.

Megatron flags:

- `--overlap-grad-reduce`: attempt backward/gradient-communication overlap.
- `--grad-reduce-in-bf16`: reduce gradients in BF16 where supported.
- DDP bucket settings influence how early communication can begin and how large
  collective messages are.

## ZeRO Stages

ZeRO removes duplicated model-training state across DP ranks.

| Stage | Sharded state | Communication impact |
| --- | --- | --- |
| ZeRO-1 | optimizer states | params must be restored/gathered after optimizer update |
| ZeRO-2 | optimizer states and gradients | gradients are reduce-scattered instead of fully all-reduced |
| ZeRO-3 / FSDP | optimizer states, gradients, parameters | parameters are all-gathered before use; gradients reduce-scattered |

Megatron distributed optimizer is closest to ZeRO-style optimizer/gradient
sharding:

- `--use-distributed-optimizer` changes gradient reduction from full AllReduce
  to ReduceScatter in many Megatron code paths.
- The log line `Using reduce-scatter for gradient reductions because
  self.ddp_config.use_distributed_optimizer=True` is expected.
- `--overlap-param-gather` attempts to gather parameter shards early enough that
  forward compute does not wait.

Explanation pattern:

1. Say what state is no longer replicated.
2. Say which collective replaces the naive collective.
3. Say which communication is expected to overlap.
4. Say what can still bottleneck: small buckets, topology, low batch per rank,
   or unsupported overlap path.

## Tensor and Sequence Parallelism

Tensor parallelism shards weight matrices and corresponding activations across
GPU ranks.

Common Megatron-style intuition:

- Column-parallel linears split output features.
- Row-parallel linears split input features and reduce partial outputs.
- Attention heads can be partitioned across TP ranks.
- MLP up/gate/down projections are sharded so each rank owns a slice.

Costs:

- More TP reduces per-GPU parameter/activation work.
- TP adds collectives on the critical path.
- TP is usually best kept inside a fast NVLink/NVSwitch domain.
- Very high TP can become latency/communication dominated.

Sequence parallelism:

- Complements TP by sharding some sequence-dimension activations.
- Reduces activation memory for operations that otherwise replicate sequence
  activations across TP ranks.
- Often tied to tensor-parallel layouts and layernorm/dropout-style operations.

Megatron flags:

- `--tensor-model-parallel-size`.
- `--sequence-parallel`.

## Context Parallelism

Context parallelism shards the sequence dimension for attention-heavy regions.

Use when:

- Long sequence length makes activation or attention memory too large.
- Need to spread long-context attention work across ranks.

Costs:

- Attention requires cross-rank exchange of K/V or partial attention results.
- Ring attention and zig-zag schedules are ways to balance communication and
  compute.
- CP usually matters more as sequence length grows.

Megatron flag:

- `--context-parallel-size`.

## Pipeline Parallelism

Pipeline parallelism shards layers across stages.

Benefits:

- Reduces per-rank parameter and optimizer memory by layer partitioning.
- Communication is primarily activation tensors between neighboring stages.
- Can be useful across slower inter-node links when TP would communicate too
  often inside each layer.

Costs:

- Pipeline bubbles reduce utilization unless enough microbatches are in flight.
- Load balancing matters; embeddings, output heads, and uneven layers can skew
  stages.
- Schedules such as all-forward/all-backward, 1F1B, interleaving, zero-bubble,
  or DualPipe trade memory, scheduling complexity, and utilization.

Megatron flags:

- `--pipeline-model-parallel-size`.
- virtual/interleaved pipeline flags when enabled by the run config.

## Expert Parallelism

Expert parallelism shards MoE experts.

Pattern:

- Router chooses top-k experts for each token.
- Tokens are sent to expert-owning ranks, usually via AllToAll.
- Expert outputs are sent back and combined.

Costs:

- AllToAll can dominate, especially across slow network boundaries.
- Load imbalance can cause stragglers.
- Expert parallelism only affects MoE blocks; dense attention and non-expert
  layers still need DP/TP/PP/CP choices.

## Configuration Selection

Practical decision process:

1. Fit a training step in memory.
   - Try recomputation and gradient accumulation first for small runs.
   - For models under roughly 10B params on 8 GPUs, a single primary sharding
     technique may be enough.
   - For 10B-100B params, combine TP with PP or TP with ZeRO/FSDP.
   - For very long sequences, add CP.
   - For MoE, add EP.
2. Achieve the target global batch size.
   - Increase DP or gradient accumulation to raise GBS.
   - Reduce DP in favor of TP/PP/CP if GBS is too large or DP comms dominate.
3. Optimize throughput.
   - Prefer fast intra-node bandwidth for TP.
   - Increase DP/ZeRO while communication remains hidden.
   - Move to PP when DP/FSDP communication becomes the bottleneck at scale.
   - Benchmark multiple MBS/GAS/DP/TP/PP/ZeRO combinations; theory narrows the
     search but does not replace measurement.

Do not present these as universal thresholds. They depend on GPU memory, GPU
count per node, NVLink/NVSwitch/InfiniBand topology, kernel quality, checkpoint
policy, precision, model shape, and batch size.

## Megatron-LM Log Mapping

Use this mapping when explaining pasted logs:

| Log or flag | Meaning |
| --- | --- |
| `Using reduce-scatter... use_distributed_optimizer=True` | distributed optimizer is sharding gradient reduction instead of full AllReduce |
| `setting training iterations to X` | sample-based schedule converted `train_samples / global_batch_size` to optimizer steps |
| `learning rate decay style: cosine` | LR scheduler will decay according to cosine over configured sample/iteration horizon |
| `MockGPTDataset` | real training loop with synthetic token data |
| `Build and save ... indices` | dataset index/cache construction for deterministic data iteration |
| `--mock-data` | bypass real indexed dataset; useful for environment/performance smoke runs |
| `--fp8-format hybrid` | TransformerEngine FP8 path; explain hardware support and numerical caveats |
| `--overlap-grad-reduce` | attempt to hide gradient communication under backward compute |
| `--overlap-param-gather` | attempt to prefetch/gather parameter shards before forward use |
| `--profile --profile-step-start N` | profiling is enabled for selected steps; expect overhead and trace artifacts |

Local Llama example interpretation:

- `run_llama_anchor.py`: tiny model, one local GPU by default, mock data, same
  Megatron training mechanics.
- `run_llama3_8b_long.py`: 8-GPU local Docker run, Llama-3-style 8B dimensions,
  GQA, BF16/FP8, sample-based schedule, distributed optimizer, overlap flags,
  checkpoint/tensorboard/profiler outputs.

When explaining a run, always connect the log back to:

1. what state is stored,
2. what work is computed,
3. what data moves between ranks,
4. what overlaps with what,
5. what resource is likely limiting.
