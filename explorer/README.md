# Megatron Explorer

An interactive, static explorer for Megatron-LM and Megatron Core internals.

## Modes

- Guide: generated inner-workings guide.
- Components: source-linked training loop and Megatron Core graphs.
- Parallelism: TP, PP, DP, CP, EP, and FSDP rank geometry.
- Model Architecture: Qwen3 and DeepSeek-V3 circuits grounded in Megatron Core source.

## Run

```bash
cd explorer
./scripts/workflow.sh gen-data
./scripts/workflow.sh install
./scripts/workflow.sh dev
```

## Verify

```bash
cd explorer
./scripts/workflow.sh verify
```

The explorer is a documentation and visualization tool. It does not launch Megatron training jobs.
