# Source Material

This skill is grounded in two public scaling references. This file indexes the
repo-local vendored source material and preserves upstream attribution.

## JAX Scaling Book

- Title: `How To Scale Your Model`
- URL: https://jax-ml.github.io/scaling-book/
- Source repo: https://github.com/jax-ml/scaling-book
- License: MIT, per `jax-ml/scaling-book` repository license.
- Citation requested by source:

```text
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
```

Primary chapters to consult:

| Need | Source |
| --- | --- |
| Roofline, arithmetic intensity, compute vs bandwidth | https://jax-ml.github.io/scaling-book/roofline/ |
| Sharded arrays, collectives, sharded matmul rules | https://jax-ml.github.io/scaling-book/sharding/ |
| Transformer params, FLOPs, KV cache, remat | https://jax-ml.github.io/scaling-book/transformers/ |
| DP, FSDP/ZeRO, TP, PP training rooflines | https://jax-ml.github.io/scaling-book/training/ |
| LLaMA 3 worked training estimates | https://jax-ml.github.io/scaling-book/applied-training/ |
| GPU hardware, GPU collectives, GPU rooflines | https://jax-ml.github.io/scaling-book/gpus/ |

Local distilled notes:

- `references/jax-scaling-book.md`

Vendored raw source:

- Directory: `references/source-materials/jax-scaling-book/`
- Included: top-level markdown chapters, upstream `README.md`, upstream
  `LICENSE`, and instructional figure/plot assets under `assets/img/`,
  `assets/gpu/`, and `assets/plotly/`.
- Not included: Jekyll theme plumbing, webfonts, generated CSS/JS, package
  management files, and other non-instructional site infrastructure.

## Ultra-Scale Playbook

- Title: `The Ultra-Scale Playbook: Training LLMs on GPU Clusters`
- URL: https://huggingface.co/spaces/nanotron/ultrascale-playbook
- Source repo: https://huggingface.co/spaces/nanotron/ultrascale-playbook/tree/main
- License: Apache-2.0, per Hugging Face Space metadata in `README.md`.
- Citation requested by source:

```text
Tazi et al., "The Ultra-Scale Playbook: Training LLMs on GPU Clusters", 2025.
```

Primary sections to consult:

| Need | Source section |
| --- | --- |
| Memory, compute, communication framing | High level overview |
| Single-GPU training anatomy and memory | First Steps: Training on one GPU |
| Activations and recomputation | Activation recomputation |
| Batch and microbatch mechanics | Gradient accumulation |
| Profiling compute and communication | Profiling GPU compute and communication |
| DP and overlap | Data Parallelism |
| ZeRO stages | ZeRO |
| TP and SP | Tensor Parallelism / Sequence Parallelism |
| CP and ring attention | Context Parallelism |
| PP schedules | Pipeline Parallelism |
| MoE routing | Expert Parallelism |
| Configuration selection | Finding the Best Training Configuration |

Local distilled notes:

- `references/ultrascale-playbook.md`

Vendored raw source:

- Directory: `references/source-materials/ultrascale-playbook/`
- Included: upstream `README.md`, `.gitattributes`, `.gitignore`,
  `ultra_blog.md`, conversion scripts, source-side `src/`, `python/`,
  `assets/data/`, `assets/images/`, and the original PDF
  `The_Ultra-Scale_Playbook_Training_LLMs_on_GPU_Clusters.pdf`.
- Not included: generated `dist/` duplicates and large audio files under
  `assets/audio/`.

## Copyright Boundary

The source projects are permissively licensed at the time they were pulled:
MIT for `jax-ml/scaling-book` and Apache-2.0 for the Hugging Face Space. Keep
the upstream license/metadata files with any copied source material. If a future
task refreshes these files, re-check upstream licenses and update this index.
