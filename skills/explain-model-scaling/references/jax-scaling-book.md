# JAX Scaling Book Notes

Source: https://jax-ml.github.io/scaling-book/

These notes distill the parts most useful for explaining Megatron-LM model
architecture, training logs, and distributed training decisions. They follow the
book's system model: quantify work, quantify bytes, then ask which resource
dominates.

## Table of Contents

- Roofline model
- Sharding and collectives
- Transformer accounting
- Training parallelism
- LLaMA training estimates
- GPU-specific notes
- Explanation checklist

## Roofline Model

A training run is constrained by three resources:

- Compute: accelerator FLOPs/s.
- Bandwidth: bytes/s for HBM, NVLink, InfiniBand, ICI/DCN, PCIe, etc.
- Capacity: memory bytes available for params, optimizer, activations, caches,
  temporary buffers, kernels, and allocator fragmentation.

First-order timing:

```text
T_math  = compute_FLOPs / accelerator_FLOPs_per_second
T_comm  = communication_bytes / bandwidth_bytes_per_second
T_step >= max(T_math, T_comm) when overlap is possible
T_step <= T_math + T_comm as a loose upper bound
```

Arithmetic intensity:

```text
intensity = compute_FLOPs / communication_bytes
critical_intensity = accelerator_FLOPs_per_second / bandwidth_bytes_per_second
```

Interpretation:

- `intensity > critical_intensity`: compute-bound if implementation can reach
  the roofline.
- `intensity < critical_intensity`: bandwidth or communication bound.
- Overlap can hide communication only if there is enough independent compute on
  the critical path.

Useful rules:

- Large dense matmuls have unusually high arithmetic intensity because FLOPs
  scale roughly cubically while operand bytes scale roughly quadratically.
- For BF16 matmuls, the critical local token batch is roughly a few hundred
  tokens on common TPU/GPU hardware; below that, memory bandwidth or launch
  overhead often matters.
- Small collectives can be latency-bound even when the bandwidth formula looks
  cheap.

## Sharding and Collectives

The book's notation separates logical tensor shape from device-local shape:

```text
A[I, J]        # logical shape
A[I_X, J]      # I is sharded across mesh axis X
A[I_X, J_Y]    # I over X and J over Y
A[I_XY, J]     # I over flattened X/Y mesh axes
```

Sharded matmul cases:

| Case | Pattern | Communication |
| --- | --- | --- |
| No sharded contracting dimension | `A[I_X,J] @ B[J,K_Y]` | Local matmul is valid; no collective needed |
| One sharded contracting dimension | `A[I,J_X] @ B[J,K]` | Usually AllGather the contracted input |
| Both inputs shard same contracting dimension | `A[I,J_X] @ B[J_X,K]` | Local partial sums, then AllReduce or ReduceScatter |
| Non-contracting dimensions use same mesh axis | `A[I_X,J] @ B[J,K_X]` | Invalid layout until one side/result is gathered or resharded |

Collective vocabulary:

- AllGather removes a sharding subscript by giving every participant the full
  logical tensor along that mesh axis.
- ReduceScatter reduces partial sums and leaves a selected logical dimension
  sharded.
- AllReduce is equivalent to ReduceScatter plus AllGather for first-order byte
  modeling.
- AllToAll moves shards between logical axes; it is common in MoE and resharding
  between incompatible layouts.

Backpropagation connection:

- The transpose of AllGather is ReduceScatter.
- The transpose of ReduceScatter is AllGather.
- This is why forward-pass gathers often become backward-pass reduce-scatters.

Overlapped collective matmul:

- Communication can sometimes be tiled and overlapped with chunks of a matmul.
- Treat overlap claims as implementation-specific; use profiler traces to verify
  whether comm streams run concurrently with compute kernels.

## Transformer Accounting

Symbols:

```text
B = batch size for the operation, often tokens or microbatch sequences
T = sequence length
D = hidden size
F = feed-forward hidden size
V = vocabulary size
L = number of layers
N = query heads
K = KV heads or KV groups
H = head dimension
```

Basic FLOPs:

```text
dot(x[P], y[P])      ~= 2P
matvec([N,P] @ [P])  ~= 2NP
matmul([N,P]@[P,M])  ~= 2NPM
training matmul      ~= 6NPM  # forward + two backward matmuls
```

Dense or GQA decoder layer parameters:

```text
MLP params per layer       ~= 3 * D * F        # gated MLP
QKVO params per layer      ~= 2 * D * (N+K) * H
layer params               ~= MLP + QKVO
embedding params           ~= V * D
unembedding params         ~= V * D if untied
```

Dense Transformer training cost:

```text
training_FLOPs ~= 6 * parameter_count * token_count
```

Use this as a rule of thumb for dense Transformers at ordinary context lengths.
It excludes or approximates:

- dot-product attention when `T` is very large,
- MoE sparsity and routing,
- recomputation,
- norms, elementwise ops, optimizer kernels,
- pipeline bubbles and non-overlapped communication.

Attention cost:

```text
QKVO projection train FLOPs ~= 12 * B * T * D * (N+K) * H
attention train FLOPs       ~= 12 * B * T^2 * N * H
```

With standard MHA assumptions (`F = 4D`, `D = N*H`, `K = N`), dot-product
attention becomes comparable to MLP/projection matmuls when `T` approaches
`8D`. For smaller models or very long context, attention matters earlier.

KV cache:

```text
KV cache elements per sequence ~= 2 * S * L * K * H
KV cache bytes                 ~= elements * bytes_per_element
```

Use GQA/MQA (`K < N`) to reduce KV cache memory and bandwidth during inference.

Gradient checkpointing / rematerialization:

- Saving every intermediate activation is expensive.
- Block remat stores only layer inputs and recomputes much of the forward pass
  during backward.
- Selective policies keep expensive matmul outputs and recompute cheaper
  attention/elementwise pieces.
- In explanations, separate useful model FLOPs from extra recomputation FLOPs.

## Training Parallelism

Primary schemes:

| Scheme | What is sharded | Main benefit | Main cost |
| --- | --- | --- | --- |
| Data Parallelism | batch/sequence samples | Simple throughput scaling and activation sharding | replicated params/optimizer; gradient AllReduce |
| FSDP / ZeRO-3 | params, grads, optimizer across DP ranks | major memory reduction | param AllGather and grad ReduceScatter |
| Tensor Parallelism | hidden/FFN/attention matmul dimensions | shard large layer compute and weights | activation AllGather/ReduceScatter on critical path |
| Pipeline Parallelism | layers | reduces per-stage parameter memory | pipeline bubbles and scheduling complexity |
| Expert Parallelism | MoE experts | scales parameter count sparsely | token-routing AllToAll |
| Context/Sequence Parallelism | sequence-related tensors | long-context activation and attention memory relief | attention communication and layout constraints |

Important JAX-book maxim:

```text
FSDP moves weights.
Tensor parallelism moves activations.
```

Implications:

- FSDP communication gets worse when many ranks shard small batches, because
  there is less compute per shard to hide weight movement.
- TP communication is tied to activation size and TP degree; it is usually kept
  to fast intra-node or high-bandwidth mesh axes.
- FSDP and TP can complement each other: TP shrinks FSDP's weight shards; FSDP
  shrinks TP activation gathers by reducing per-replica batch.

For explaining Megatron-LM:

- `--use-distributed-optimizer` corresponds to the ZeRO/FSDP idea of sharding
  optimizer state and reducing gradients as shards.
- `--overlap-grad-reduce` tries to hide ReduceScatter/AllReduce behind backward
  compute.
- `--overlap-param-gather` tries to prefetch gathered parameters before use.
- `--tensor-model-parallel-size > 1` introduces TP-style activation/gradient
  communication around MLP and attention matmuls.
- `--pipeline-model-parallel-size > 1` splits layers and makes microbatch count
  important for bubble efficiency.
- `--context-parallel-size > 1` or sequence-parallel features shift sequence
  activations/attention work across ranks.

## LLaMA Training Estimates

For LLaMA-style dense decoder models:

- Gated MLP matrices dominate parameters and FLOPs.
- A quick cost estimate is `6 * params * training_tokens`.
- Time estimate:

```text
seconds ~= total_training_FLOPs / (num_accelerators * peak_FLOPs_per_accelerator * MFU)
```

Use this to sanity-check claims like:

- How many iterations are implied by `train_samples / global_batch_size`.
- Whether a run duration is a real full pretraining run or a timed stress run.
- Whether the run is more likely limited by compute, memory, or communication.

For LLaMA-3-like configs:

- Check whether output embeddings are tied or untied.
- Use GQA groups for `K`.
- Treat `seq_length` and `global_batch_size` as separate: total tokens per
  optimizer step is usually `global_batch_size * seq_length`.

## GPU-Specific Notes

Modern NVIDIA GPUs:

- SMs contain Tensor Cores for matmul, CUDA cores for vector work, registers,
  shared memory/SMEM, L2, and HBM.
- H100/H200/B200 class GPUs have high Tensor Core FLOPs but must be fed by HBM,
  L2/SMEM/TMEM, and the network.
- FP8/BF16 peak numbers may assume details such as sparsity; be explicit about
  whether the quoted peak is usable for the workload.

GPU network hierarchy:

- Within an 8xH100 node, NVLink/NVSwitch gives high GPU-to-GPU bandwidth.
- Cross-node traffic uses InfiniBand or an equivalent fabric and has different
  bandwidth/latency behavior.
- GB200 NVL72 changes the topology by putting many GPUs in one NVLink domain.

Collective rules of thumb:

- AllGather/ReduceScatter cost is roughly bytes divided by the effective egress
  bandwidth for large messages.
- AllReduce is usually about twice AG/RS unless in-network reductions help.
- AllToAll within a node can be much cheaper than AllGather, but cross-node
  AllToAll can degrade sharply.
- Real NCCL bandwidth can be well below claimed peak for moderate message sizes,
  so profiler traces and NCCL measurements matter.

## Explanation Checklist

When using this source to explain a Megatron run:

1. Extract `L, D, F, V, T, N, K, H`.
2. Compute or estimate params per layer and total params.
3. Convert samples to tokens if the run is sequence training.
4. Estimate training FLOPs with `6 * params * tokens`; adjust for long context,
   MoE, and recomputation.
5. Account for memory: params, grads, optimizer, master weights, activations,
   temporary buffers.
6. Map every parallelism flag to what it shards and which collective it adds.
7. Explain whether overlap is enabled and whether it is likely enough.
8. Identify likely bottleneck: compute, HBM, NVLink, InfiniBand, latency,
   pipeline bubble, data loading, checkpointing, or profiler overhead.
