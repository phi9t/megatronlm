# Model Architecture Source Links Design

## Problem

The Megatron Explorer model architecture page shows correct repo-relative source
files and line anchors, but it combines them with the wrong commit SHA.

Generated model manifests currently set:

```text
source = d681f6898b64d613991e0b1ac4b2c631bf9aca36
```

That SHA exists only in the local feature history context that produced the
manifests. It is not reachable from `phi9t/megatronlm` on GitHub, so URLs such
as this return 404:

```text
https://github.com/phi9t/megatronlm/blob/d681f6898b64d613991e0b1ac4b2c631bf9aca36/megatron/core/models/gpt/gpt_model.py#L154
```

The referenced files and lines themselves are correct in the fork. The fix is
to publish model manifests with a valid source revision.

## Findings

The source audit checked all model architecture refs in:

- `explorer/public/data/models/qwen3-0_6b.json`
- `explorer/public/data/models/qwen3-8b.json`
- `explorer/public/data/models/qwen3-30b-a3b.json`
- `explorer/public/data/models/deepseek-v3.json`

The audit found 27 unique block refs and 76 total block uses. Every referenced
file exists in this repo, every referenced line is in range, and the line text
matches the intended symbol.

Examples:

- `GPTModel.embedding` -> `megatron/core/models/gpt/gpt_model.py:154`
- `SelfAttention.linear_qkv` -> `megatron/core/transformer/attention.py:1397`
- `MultiLatentAttention.linear_q_down_proj` -> `megatron/core/transformer/multi_latent_attention.py:527`
- `SharedExpertMLP` -> `megatron/core/transformer/moe/shared_experts.py:96`

## Design

Keep the existing block-level `ref` values as repo-relative `file:line` strings.
They are valid and should remain independent from the publish target.

Change model manifest generation so the manifest-level `source` field is
provided by the build environment:

```text
MEGATRON_EXPLORER_SOURCE_REF
```

When the variable is unset, default to `main`. This keeps local checked-in data
stable and readable.

When the GitHub Pages workflow builds the explorer, it already has access to the
pushed commit via `${{ github.sha }}`. The workflow should pass that value to
`build_model_arch.py` through `MEGATRON_EXPLORER_SOURCE_REF`. The live Pages JSON
will then link to the exact commit that produced the app.

The UI can continue to use `manifest.source` in
`ArchitectureExplorer.tsx`. Once generation writes a valid published SHA, the UI
does not need special-case logic.

## Verification

Local verification should prove:

- `./scripts/workflow.sh verify` still passes.
- Generated model manifests default `source` to `main` when no environment
  variable is provided.
- Generated model manifests use the provided
  `MEGATRON_EXPLORER_SOURCE_REF` when set.
- Existing file/line refs still pass manifest verification.

Published-link verification should prove:

- The live model JSON source equals the pushed Pages build SHA.
- Sample model-architecture code links return HTTP 200 for
  `phi9t/megatronlm`.
- Live explorer assets and model JSON contain no
  `github.com/NVIDIA/Megatron-LM` links for repo-local model architecture
  source.

## Scope

This change is limited to model architecture source revision plumbing and
verification. It does not change the model architecture diagrams, parameter
estimates, block definitions, or UI layout.
