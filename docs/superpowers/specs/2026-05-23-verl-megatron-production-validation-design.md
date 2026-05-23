# verl Megatron-FSDP Production Validation Design

## Goal

Run the verl Megatron-FSDP example end to end on a single 8-GPU B200 node in a
bounded production-validation mode. The run proves the operational path is
working without spending a full training budget.

The validation covers both phases from the upstream verl example:

1. Megatron-FSDP SFT on GSM8K using Qwen/Qwen2.5-Math-7B.
2. Megatron-FSDP GRPO/RL using the corresponding verl Megatron-FSDP example.

The output must include enough structured evidence to show what ran, where it
ran, what versions were used, which checkpoints were written, and which pass/fail
gates succeeded.

## Scope

Target environment:

* single node
* 8 visible B200 GPUs
* existing Docker-based workflow
* vendored `third_party/verl`
* vendored `third_party/Megatron-Bridge`
* nested Megatron-LM from Megatron-Bridge
* repo-owned Megatron-Bridge compatibility patch applied by setup tooling

This design does not add multi-node orchestration, SLURM submission, or full
training-length production jobs. Those can build on this once the single-node
validation is stable.

## Architecture

Extend `tools/verl_megatron.py` with a production-validation layer instead of
creating a separate launcher. The existing tool already owns submodule pin
checks, Bridge patch checks, Docker image handling, Python path construction,
and container execution. The production validator reuses those primitives.

Add one new top-level subcommand:

```bash
UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py validate-production
```

Add a matching mise task:

```bash
mise run verl-megatron-production-validation
```

The implementation may split evidence writing into a helper module if
`tools/verl_megatron.py` becomes too broad. The helper stays narrowly focused on
evidence files and does not own run orchestration.

## Runtime Layout

Each run writes to a timestamped repo-local ignored directory:

```text
local/verl-runs/production-validation/<timestamp>/
├── setup/
├── data/
├── sft/
├── rl/
├── checkpoints/
├── logs/
└── evidence/
```

The evidence directory contains:

```text
evidence.json
summary.md
sft_command.sh
rl_command.sh
environment.json
```

The command files must contain the exact container-inner command lines that were
executed, including Hydra overrides.

## Data Flow

### 1. Host Preflight

Before launching training, validate and record:

* parent repository commit
* dirty status of the parent repository
* `third_party/verl` commit
* `third_party/Megatron-Bridge` commit
* nested `third_party/Megatron-Bridge/3rdparty/Megatron-LM` commit
* Bridge patch state
* Docker image tag and image ID
* expected GPU count

The expected commits are the pins currently codified in `tools/verl_megatron.py`.

### 2. Container Preflight

Inside Docker, validate and record:

* `torch.cuda.device_count() == 8`
* GPU names
* CUDA version
* PyTorch version
* selected Python package versions where available
* imports for:
  * `verl`
  * `verl.trainer.sft_trainer`
  * `verl.workers.engine.megatron.transformer_impl`
  * `megatron.bridge.models.conversion.param_mapping`
  * `megatron.core`

### 3. Data Preparation

Prepare the datasets expected by the upstream example:

* GSM8K multiturn SFT parquet for SFT.
* RL train/test parquet files used by the GRPO Megatron-FSDP example.

Record for each dataset:

* path
* file size
* row count
* creation command

### 4. Bounded SFT

Run SFT with the same production path as the upstream example:

* `torchrun --standalone --nnodes=1 --nproc_per_node=8`
* `verl.trainer.sft_trainer`
* Megatron engine
* Megatron-FSDP enabled
* Megatron-Bridge enabled
* TP=4
* PP=1
* EP=1
* Qwen/Qwen2.5-Math-7B
* bounded step count of 2 training steps by default
* checkpoint saving enabled for model contents

Collect:

* exit code
* start and end timestamps
* full stdout/stderr log
* final log tail
* loss values if emitted
* checkpoint directory inventory
* checkpoint tracker metadata if present

### 5. Bounded RL

Run the upstream GRPO Megatron-FSDP path in a bounded validation mode:

* `python3 -m verl.trainer.main_ppo`
* `--config-path=config`
* `--config-name=ppo_megatron_trainer.yaml`
* 8 GPUs
* Megatron actor
* Megatron-FSDP enabled
* Qwen/Qwen2.5-Math-7B as the model input
* bounded training/update step count of 2 update steps by default
* checkpoint/log output enabled

The first production validation uses the base model for RL because the upstream
GRPO Megatron-FSDP script is parameterized by `HF_MODEL_PATH` and defaults to
Qwen/Qwen2.5-Math-7B. The SFT phase is still required to write and verify its
own checkpoint, but the initial RL phase is not chained from that SFT output.
Evidence records this explicitly so a future design can add an SFT-to-RL
handoff check as a separate extension.

Collect:

* exit code
* start and end timestamps
* full stdout/stderr log
* actor initialization evidence
* rollout initialization evidence
* at least one completed training or update step
* reward, advantage, or loss metrics if emitted
* checkpoint directory inventory

### 6. Evidence Report

At the end of every run, successful or failed, write:

* `evidence/evidence.json` for machine-readable checks and paths
* `evidence/summary.md` for human review

The summary includes:

* overall pass/fail status
* run directory
* versions and commits
* image ID
* GPU inventory
* SFT status and checkpoint evidence
* RL status and checkpoint evidence
* failed gate details if any

## Pass/Fail Gates

The validation passes only if all gates succeed:

1. Expected submodule and nested submodule commits match.
2. Bridge compatibility patch is applied.
3. Docker image exists and its image ID is recorded.
4. Container sees exactly 8 CUDA devices.
5. Container imports all required verl, Megatron-Bridge, and Megatron-Core modules.
6. SFT and RL data files exist and have nonzero row counts.
7. SFT exits with code 0.
8. SFT writes model checkpoint content.
9. RL exits with code 0.
10. RL emits at least one completed training or update step.
11. Logs do not contain fatal patterns:
    * `Traceback`
    * `RuntimeError`
    * `CUDA out of memory`
    * `NCCL error`
    * `AssertionError`
    * `KeyError`
12. `evidence.json` and `summary.md` are written.

If any gate fails, the tool stops after the failing phase, writes partial
evidence, and returns a nonzero exit code.

## Error Handling

The validator fails closed. Any missing dataset, missing image, mismatched
commit, unapplied patch, import failure, nonzero process exit, missing checkpoint,
or fatal log pattern is a failed validation.

Partial evidence is still required on failure. The evidence includes the
phase that failed, the command that failed, the exit code when available, and the
log path.

The validator does not delete prior run directories. Each run gets a new
timestamped path.

## Testing

Add focused unit tests for command construction and evidence gate behavior:

* production-validation SFT command includes `torchrun`, 8 ranks,
  `verl.trainer.sft_trainer`, Megatron engine, Megatron-FSDP, TP=4, and bounded
  step overrides.
* production-validation RL command includes the upstream GRPO Megatron-FSDP
  entrypoint, 8-GPU launch shape, Megatron-FSDP, and bounded step overrides.
* Docker command uses the repo-relative `--mount type=bind,source=.,target=/workspace`.
* evidence writer records pass and fail gates.
* fatal log scanner flags known fatal patterns and ignores benign warnings.

Run-time verification for the implemented feature includes:

```bash
python -m py_compile tools/verl_megatron.py tests/unit_tests/tools/test_verl_megatron.py
git diff --check
UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py validate-production --dry-run
UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py validate-production
```

The final command is the evidence-producing B200 validation. It must produce a
passing `summary.md` and `evidence.json`.
