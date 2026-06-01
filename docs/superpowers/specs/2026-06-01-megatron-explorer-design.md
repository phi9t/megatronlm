# Megatron Explorer Design

Date: 2026-06-01

## Goal

Create a standalone `megatronlm/explorer` web app in the style of
`~/CodeBase/vllm/explorer` that teaches the inner workings of Megatron-LM and
Megatron Core. The first version should feel like the vLLM explorer family: a
dark Observatory shell, switchable modes, source-linked diagrams, detail
drawers, generated static manifests, and no backend.

The app is a documentation and visualization tool. It does not launch or verify
training jobs.

## Architecture

`megatronlm/explorer` will be a standalone Vite app using React, TypeScript,
Tailwind 4, lucide icons, and the Observatory design language from the vLLM
explorer. It will copy/adapt the useful shell and shared explorer-kit patterns
instead of extracting a shared cross-repo package.

The top-level app will expose four modes through a family switcher:

- **Guide**: a rendered Megatron inner-workings guide generated from curated
  repo-local docs.
- **Components**: clickable subsystem graphs with source-linked drawers.
- **Parallelism**: interactive TP, PP, DP, CP, EP, and FSDP diagrams and rank
  formulas.
- **Model Architecture**: Qwen3 and DeepSeek-V3 circuit views adapted from the
  vLLM model architecture explorer and anchored to Megatron Core source.

The SPA is a pure reader. Python scripts under `explorer/scripts` generate
static JSON and Markdown into `explorer/public/data`, and React views render
those manifests.

## Mode Design

### Guide

`GuideExplorer` renders `public/data/guide.md` with Markdown and Mermaid support.
The guide is generated from a curated set of source documents:

- `README.md`
- `docs/get-started/overview.md`
- `docs/user-guide/parallelism-guide.md`
- `docs/ultra-scale-playbook/megatron-mapping.md`
- selected feature docs under `docs/user-guide/features/`

The guide should be concise and navigable, not a full docs mirror. It should
explain the main path from launcher/config to training loop, Megatron Core
building blocks, parallelism, optimizer/checkpointing, and model families.

### Components

`ComponentExplorer` renders switchable subsystem graphs. Version 1 includes at
least two subjects:

- **Training loop**: config/launcher to initialization, dataset, model provider,
  forward/backward, optimizer, checkpointing, and logging.
- **Megatron Core stack**: `GPTModel`, `TransformerBlock`, `TransformerLayer`,
  attention, dense MLP, MoE, tensor parallel, pipeline parallel, distributed
  data parallel/FSDP, optimizer, and datasets.

Each node opens a drawer with a short explanation, a `file:line` source link,
and related docs where available.

### Parallelism

`ParallelismExplorer` teaches how Megatron composes parallelism dimensions. It
uses strategy cards, rank-grid diagrams, and controls for:

- `world_size`
- tensor parallel size
- pipeline parallel size
- context parallel size
- expert parallel size
- inferred data parallel size

The mode shows formulas such as `DP = world_size / (TP * PP * CP)` for dense
models and flags invalid combinations inline. It also explains where each
strategy appears in config flags and implementation files.

Inputs come from `docs/user-guide/parallelism-guide.md`,
`docs/ultra-scale-playbook/megatron-mapping.md`, and
`examples/llama/configs/*.yaml`.

### Model Architecture

`ArchitectureExplorer` adapts the vLLM model circuit mode to Megatron. Initial
subjects:

- Qwen3-0.6B
- Qwen3-8B
- Qwen3-30B-A3B
- DeepSeek-V3

The generator uses bundled offline configs so data generation does not require
network access. Circuit blocks should be anchored to Megatron Core concepts and
files where possible, including:

- `megatron/core/models/gpt/gpt_model.py`
- `megatron/core/models/gpt/gpt_layer_specs.py`
- `megatron/core/models/gpt/moe_module_specs.py`
- `megatron/core/models/gpt/experimental_attention_variant_module_specs.py`
- `megatron/core/transformer/transformer_block.py`
- `megatron/core/transformer/transformer_layer.py`
- `megatron/core/transformer/attention.py`
- `megatron/core/transformer/mlp.py`
- `megatron/core/transformer/moe/*`
- `megatron/core/transformer/multi_latent_attention.py`

The view keeps the vLLM-style lenses:

- **Flow**
- **Shapes**
- **Compute**
- **Memory**

It also keeps a token-count slider and concise block labels with full
explanations in the drawer.

## Data Contracts

Every mode follows the same static manifest pattern:

- `index.json` lists available subjects.
- Each subject has one manifest JSON.
- Generated Markdown is stored under `public/data`.
- React components do not hard-code subsystem or model content except view
  mechanics.

Planned generators:

- `scripts/build_guide.py`
- `scripts/build_component_manifest.py`
- `scripts/build_parallelism_manifest.py`
- `scripts/build_model_arch.py`
- `scripts/workflow.sh`

The workflow is:

```text
repo docs/source/YAML
  -> explorer/scripts/*.py
  -> explorer/public/data/*.json and guide.md
  -> React views
  -> diagrams, drawers, source links
```

Source references should be resolved with symbol grep where possible so line
drift is tolerated. If a symbol cannot be resolved, the manifest should retain a
file-level link and record a warning rather than failing the whole explorer
build.

Generated data must use repo-relative paths only. No absolute host paths should
be emitted.

## Frontend Structure

Planned structure:

```text
explorer/
  src/
    App.tsx
    explorer-kit/
      AsyncBoundary.tsx
      DetailDrawer.tsx
      SubjectSwitcher.tsx
      ViewTabs.tsx
      mode.ts
    guide/
      GuideExplorer.tsx
    components-deepdive/
      ComponentExplorer.tsx
      ArchitectureGraph.tsx
      ComponentDrawer.tsx
      types.ts
    parallelism/
      ParallelismExplorer.tsx
      RankGrid.tsx
      StrategyDiagram.tsx
      types.ts
    architecture/
      ArchitectureExplorer.tsx
      ModelCircuit.tsx
      blockTypes.ts
      modelArch.ts
    lib/
      assets.ts
      fetch.ts
      utils.ts
    index.css
  public/data/
  scripts/
```

The app uses local React state only. It should not introduce a router or global
store.

## Interaction Requirements

- Every graph and circuit node is clickable and keyboard-operable with Enter and
  Space.
- Every mode with multiple subjects uses a subject switcher.
- Drawers show concise explanation text, `file:line` source links, related docs,
  and useful cross-links to other modes.
- Parallelism controls keep the UI interactive even when combinations are
  invalid.
- Focus outlines, reduced-motion handling, and skip-link behavior should match
  the vLLM explorer baseline.
- Diagram labels must stay short. Full prose belongs in drawers, not inside
  graph boxes.

## Error Handling

- Missing manifest: show an `AsyncBoundary` error that tells the user to run
  `./scripts/workflow.sh gen-data`.
- Failed JSON fetch: show the failing manifest path and HTTP status.
- Unresolved source anchor: show a file-level link and generator warning.
- Invalid parallelism tuple: show the failed divisibility formula inline.
- Missing optional doc input: generator should skip that input with a warning if
  the mode still has enough data to render.

## Verification

Automated checks:

- `./scripts/workflow.sh gen-data`
- `npm run typecheck`
- `npm run lint`
- `npm run build`
- generator smoke checks for manifest shape, existing source files, finite model
  metrics, and text-label budgets

Manual browser checks:

- desktop and mobile layouts render without overlap
- family switcher changes modes
- subject switchers work
- all graph and circuit nodes can be reached by keyboard
- drawers open and show valid source links
- reduced-motion mode removes or freezes animation
- model architecture labels do not overflow at any token-slider value
- parallelism invalid combinations are clearly marked

## Scope Boundaries

Included in v1:

- standalone explorer app under `megatronlm/explorer`
- four implemented modes with generated static data
- Qwen3 and DeepSeek-V3 model subjects
- source-linked manifests grounded in repo-local files
- build/dev workflow script

Excluded from v1:

- shared package extraction with `vllm/explorer`
- backend service
- URL routing
- live training execution
- network-dependent data generation
- exhaustive coverage of every Megatron feature

## Open Decisions Resolved

- The app will be standalone in `megatronlm/explorer`.
- It will copy/adapt the vLLM explorer style and primitives instead of creating
  a shared package.
- The first model architecture subjects will mirror the vLLM explorer's Qwen3
  and DeepSeek-V3 set where possible.
- The implementation will prioritize a complete four-mode v1 over a single
  deeper subsystem explorer.
