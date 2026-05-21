<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Ultra-Scale Playbook

The Ultra-Scale Playbook is a docs-owned companion for understanding large-scale
LLM training systems and mapping those concepts back to Megatron-LM.

The upstream material is from
[The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook).
The vendored source lives in `docs/ultra-scale-playbook/app/` and keeps the
upstream Apache-2.0 metadata in `app/README.md`.

## Build and Launch

Build the static Playbook from the vendored Node app:

```bash
cd docs/ultra-scale-playbook/app
npm ci
npm run build
```

Launch the local development server:

```bash
cd docs/ultra-scale-playbook/app
npm run dev
```

Or serve the built static site directly:

```bash
python -m http.server 4173 --bind 127.0.0.1 --directory docs/ultra-scale-playbook/app/dist
```

Generated `dist/` and `node_modules/` output is intentionally not committed.

## GitHub Pages

The personal-fork GitHub Pages target is:

```text
https://phi9t.github.io/megatronlm/
```

The workflow `.github/workflows/deploy-ultra-scale-playbook-pages.yml` builds
this app and publishes `docs/ultra-scale-playbook/app/dist` to the `gh-pages`
branch.
It is gated to run only in `phi9t/megatronlm`, so the workflow will not publish
from the upstream `NVIDIA/Megatron-LM` repository.

To publish from the fork:

1. Push the branch to `phi9t/megatronlm`.
2. In the fork repository settings, configure Pages to serve the `gh-pages`
   branch.
3. Run the `Deploy Ultra-Scale Playbook Pages` workflow manually, or push to
   the configured fork branch.

## Megatron-LM Mapping

Use the [Megatron-LM mapping](megatron-mapping.md) when reading the Playbook
alongside this repository. It maps the Playbook's DP, ZeRO/FSDP, TP/SP, CP, PP,
EP/MoE, recomputation, FP8, profiling, and batch-size concepts to Megatron
flags, configs, docs, examples, and source paths.

```{toctree}
:maxdepth: 1
:hidden:

megatron-mapping
```
