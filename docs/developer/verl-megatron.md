<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# verl + Megatron-Bridge Preflight

This repo vendors the external pieces needed to reproduce the verl Megatron-FSDP
smoke path under `third_party/`:

```text
third_party/
├── Megatron-Bridge
│   └── 3rdparty/Megatron-LM
└── verl
```

Pinned revisions:

* verl: `7dc39fec1c37da4098b50e68840e78906b69f16a`
* Megatron-Bridge: `94dc04baf65157463181eef0c19549d5a6e4ccec`
* Megatron-Bridge nested Megatron-LM: `38986a98aae6a0cc4c8ae7b435db3288a890b0cb`

## Setup

Run the setup task from the repo root:

```bash
mise run verl-megatron-setup
```

The setup command:

* initializes the two submodules and the nested Megatron-LM submodule,
* verifies the expected commits,
* applies `tools/verl_megatron/patches/0001-megatron-bridge-fsdp-dtensor-local-shape.patch`,
* builds `verl-megatron-fsdp:local` from `tools/verl_megatron/Dockerfile`,
* prepares the GSM8K SFT parquet data under `local/verl-data/gsm8k_sft`.

The patch is intentionally repo-owned instead of committed inside the
third-party submodule. It fixes the Megatron-Bridge HF-to-Megatron conversion
path for FSDP DTensors by using the local `orig_param` shape for tensor-parallel
scatter sizing. Without it, the Qwen Megatron-FSDP smoke can fail during
checkpoint conversion with a TP-sized vocab or output tensor mismatch.

To print commands without changing the checkout:

```bash
mise run verl-megatron-dry-run
```

To pre-download the Qwen checkpoint into `local/hf-home` during setup:

```bash
UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py setup --download-model
```

## Fast Preflight

Run the cheap import and GPU count check:

```bash
mise run verl-megatron-preflight
```

This verifies the pinned submodules, confirms the Bridge patch is applied,
checks the Docker image, imports verl, Megatron-Bridge, and Megatron-Core inside
the container, and confirms that at least eight CUDA devices are visible. For a
CPU-only command-shape check, use:

```bash
UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py preflight --min-gpus 0 --dry-run
```

## One-Step Smoke

Run the end-to-end verl Megatron-FSDP SFT smoke:

```bash
mise run verl-megatron-smoke
```

The smoke uses Qwen/Qwen2.5-Math-7B, TP=4, Megatron-FSDP enabled, and a single
training step. Checkpoints and logs stay under:

```text
local/verl-runs/sft-qwen-mfsdp-smoke-model-only
```

The smoke saves only model contents:

```text
checkpoint.save_contents=["model"]
```

That avoids the optimizer-save path that is not needed for the preflight and
keeps the validation focused on conversion, distributed initialization,
training, and model checkpoint writing.

## Production Validation

Run the bounded SFT plus RL production validation on a single 8-GPU B200 node:

```bash
mise run verl-megatron-production-validation
```

For command review without launching training:

```bash
mise run verl-megatron-production-validation-dry-run
```

Each run writes a timestamped evidence directory under:

```text
local/verl-runs/production-validation/<timestamp>/evidence
```

The evidence includes `evidence.json`, `summary.md`, `sft_command.sh`,
`rl_command.sh`, command records, phase log paths, checkpoint inventories,
fatal-log scan results, and pass/fail gates. The first production validation
uses the base Qwen/Qwen2.5-Math-7B model for RL and verifies the SFT phase
through its own checkpoint output.
