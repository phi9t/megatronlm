---
name: explain-model-scaling
description: Use when explaining LLM architecture or training; interpreting Megatron-LM launch flags or logs; mapping examples/llama to model behavior; estimating parameters, FLOPs, tokens, memory, throughput, MFU/HFU, or communication bottlenecks; explaining DP, FSDP/ZeRO, distributed optimizer, TP, PP, CP/SP, EP/MoE, gradient accumulation, activation recomputation, FP8/BF16, or profiling traces.
---

# Explain Model Scaling

## Overview

Use this skill to explain Megatron-LM models and training runs the way the
scaling literature explains them: start from the actual artifact, do first-order
shape/FLOP/memory accounting, then connect the result to compute, memory, and
communication bottlenecks.

Primary source style:

- JAX scaling book: https://jax-ml.github.io/scaling-book/
- Ultra-Scale Playbook: https://huggingface.co/spaces/nanotron/ultrascale-playbook

Bundled references:

- `references/source-material.md`: upstream links, licenses, citations, and
  section map for the vendored source materials.
- `references/jax-scaling-book.md`: roofline, sharding, Transformer accounting,
  training parallelism, LLaMA estimates, and GPU notes.
- `references/ultrascale-playbook.md`: GPU-cluster memory, overlap, ZeRO, TP,
  CP, PP, EP, configuration selection, and Megatron log mapping.
- `references/source-materials/`: raw upstream markdown, code, figures, assets,
  and PDF pulled from the source projects.

Read the relevant reference before giving a detailed explanation. For quick
answers, `source-material.md` plus the matching focused reference is usually
enough.

## Workflow

1. Read the concrete artifact first: command, script, YAML, model config, log
   lines, profiler trace, or code path. Do not start from generic architecture
   explanation.
2. Extract the model and training symbols:
   - `L`: layers
   - `D`: hidden size
   - `F`: FFN hidden size
   - `V`: vocabulary size
   - `T`: sequence length
   - `B`: micro batch size per rank
   - `GBS`: global batch size
   - `N`: attention heads
   - `K`: KV/query groups for GQA or MQA
   - `H`: head dimension, usually `D / N`
   - `DP`, `TP`, `PP`, `CP`, `EP`: data, tensor, pipeline, context, expert parallel sizes
3. Explain the training step in order:
   - batch construction and tokenization/mock data
   - forward pass
   - loss
   - backward pass
   - gradient synchronization
   - optimizer update
   - LR schedule, logging, evaluation, checkpointing
4. Add first-order accounting only where it clarifies the run. Show the formula
   and plug in values when they are available.
5. End with the systems interpretation: what is likely compute-bound,
   memory-bound, communication-bound, or launch/profiling overhead-bound.

## Quick Reference

| Topic | Explain With |
| --- | --- |
| Transformer size | Per-layer params plus embeddings/unembedding |
| Training cost | Roughly `6 * parameters * tokens` for dense Transformers at ordinary context lengths |
| Long context | Attention cost grows with `T^2`; attention becomes important when `T` approaches `8D` |
| Memory fit | Parameters, gradients, optimizer states, master weights, activations, temporary buffers |
| Batch math | `GBS = micro_batch * DP * grad_accum` unless schedules or packing alter it |
| Distributed optimizer | Gradients reduce-scatter; optimizer owns shards; params are gathered as needed |
| TP | Shards matmuls; trades memory/compute per GPU for collectives in attention/MLP |
| PP | Shards layers; introduces bubbles unless enough microbatches fill the pipeline |
| CP/SP | Shards sequence-related tensors; useful for long contexts and activation memory |
| EP/MoE | Shards experts; adds token-routing AllToAll communication |
| Recomputation | Saves activation memory by repeating selected forward work during backward |
| MFU vs HFU | MFU counts useful model FLOPs; HFU includes extra work such as recomputation |

## Accounting Patterns

Use these as approximations, not exact claims. Mention ignored terms such as
norms, biases, tied embeddings, padding, kernels, fragmentation, and framework
buffers when they matter.

Dense or GQA decoder layer parameters:

```text
attention params ~= 2 * D * (N + K) * H
gated MLP params ~= 3 * D * F
layer params     ~= attention params + gated MLP params
embedding params ~= V * D
unembedding      ~= V * D if output weights are untied
```

Training FLOPs:

```text
matmul forward FLOPs ~= 2 * M * N * K
matmul train FLOPs   ~= 6 * M * N * K
dense Transformer    ~= 6 * parameter_count * token_count
```

Memory:

```text
BF16 params:         2 * params bytes
BF16 grads:          2 * params bytes
FP32 master weights: 4 * params bytes
Adam states:         8 * params bytes
FP32 grad accum:     +4 * params bytes when used
activations:         grow with layers * micro_batch * sequence * hidden
attention activations include sequence-squared terms
```

Communication vocabulary:

- AllReduce: every rank receives the reduced full tensor.
- ReduceScatter: reduction plus sharding; common with distributed optimizers.
- AllGather: reconstructs sharded tensors; common before using sharded params.
- AllToAll: routes different chunks to different ranks; common in MoE and some
  sequence/context layouts.
- Overlap: launch communication while later compute is still running; explain
  whether the log/config suggests overlap is enabled.

## Megatron-LM Examples

When explaining `examples/llama`, map the launcher flags to the generic
mechanisms:

- `run_llama_anchor.py`: small local environment check. Explain it as the same
  Megatron training pipeline with tiny dimensions and mock data.
- `run_llama3_8b_long.py`: local 8-GPU Llama-3-style FP8 run. Explain the
  architecture, sample-based schedule, mock-vs-real data path, distributed
  optimizer, overlap flags, checkpoint/log/profile outputs, and local Docker
  runtime.
- Logs such as `Using reduce-scatter... use_distributed_optimizer=True` mean
  Megatron is synchronizing gradients as shards for a distributed optimizer,
  not doing a full gradient AllReduce.
- Logs such as `setting training iterations to X` usually come from
  `train_samples // global_batch_size` when the run is sample-based.
- Logs showing `MockGPTDataset` mean the training loop is real, but the tokens
  are generated synthetically rather than read from a data prefix.

## Explanation Style

Prefer compact, concrete explanations:

- Anchor every claim to a flag, log line, code path, or formula.
- Define symbols before using them.
- Keep exact math separate from intuition.
- Say when a number is a rough estimate.
- Compare mechanisms by the resource they trade: memory, compute, or
  communication.
- Avoid implying a universal best parallelism layout; hardware topology,
  model size, context length, batch size, and kernel support decide.

## Common Mistakes

- Explaining DP/TP/PP/CP abstractly without mapping them to the actual command.
- Treating mock-data runs as fake training. The data is synthetic, but model,
  forward/backward, optimizer, collectives, logging, and checkpointing are real.
- Reporting `6 * params * tokens` as exact for long-context or MoE runs.
- Ignoring optimizer and activation memory when discussing whether a model fits.
- Confusing micro batch size, data-parallel batch size, global batch size, and
  gradient accumulation.
- Calling a high HFU result better without checking whether extra recomputation
  slowed end-to-end training.
