#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Interactive lab helpers for docs/tutorials/foundations.

Usage (from repository root, after install):

  python examples/tutorials/foundations/lab_runner.py --list
  python examples/tutorials/foundations/lab_runner.py layer
  python examples/tutorials/foundations/lab_runner.py parallel-state
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict


def _ensure_single_process_env() -> None:
    """Set torchrun-like env for a single local process when absent."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "6011")


def _init_core_parallel(tp: int = 1, pp: int = 1) -> None:
    import torch

    from megatron.core import parallel_state

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this lab.")

    _ensure_single_process_env()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    # Idempotent re-runs in interactive shells.
    try:
        parallel_state.destroy_model_parallel()
    except Exception:
        pass
    parallel_state.initialize_model_parallel(tp, pp)


def lab_parallel_state() -> None:
    """Print basic parallel group sizes after init."""
    from megatron.core import parallel_state

    _init_core_parallel(1, 1)
    print("=== Lab: parallel-state ===")
    print(f"world_size      = {torch_world_size()}")
    print(f"rank            = {torch_rank()}")
    print(f"tp_size         = {parallel_state.get_tensor_model_parallel_world_size()}")
    print(f"pp_size         = {parallel_state.get_pipeline_model_parallel_world_size()}")
    print(f"dp_size         = {parallel_state.get_data_parallel_world_size()}")
    print("OK: parallel_state is initialized (TP=1, PP=1).")


def torch_world_size() -> int:
    import torch

    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def torch_rank() -> int:
    import torch

    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def lab_layer(hidden_size: int, seq_len: int, batch_size: int, heads: int) -> None:
    """Build one TransformerLayer and run a forward pass."""
    import torch

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if hidden_size % heads != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by heads ({heads})")
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")

    _init_core_parallel(1, 1)

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=heads,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        bf16=False,
        fp16=False,
    )
    spec = get_gpt_layer_local_spec()
    layer = TransformerLayer(
        config=config, submodules=spec.submodules, layer_number=1
    ).cuda()

    n_params = sum(p.numel() for p in layer.parameters())
    print("=== Lab: layer ===")
    print(f"params          = {n_params}")
    print(f"layout          = [sequence, batch, hidden] = [{seq_len}, {batch_size}, {hidden_size}]")

    hidden_states = torch.randn(seq_len, batch_size, hidden_size, device="cuda")
    # Core layers accept extra kwargs depending on version; keep the call minimal.
    output = layer(hidden_states, attention_mask=None)
    if isinstance(output, tuple):
        output = output[0]
    print(f"input_shape     = {tuple(hidden_states.shape)}")
    print(f"output_shape    = {tuple(output.shape)}")
    print("OK: single-layer forward completed.")


LABS: Dict[str, str] = {
    "parallel-state": "Init torch.distributed + parallel_state; print TP/PP/DP sizes",
    "layer": "Build one TransformerLayer and run a random forward pass",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lab",
        nargs="?",
        choices=sorted(LABS.keys()),
        help="Lab name (see --list)",
    )
    parser.add_argument("--list", action="store_true", help="List labs and exit")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    args = parser.parse_args(argv)

    if args.list or args.lab is None:
        print("Available labs:")
        for name, desc in sorted(LABS.items()):
            print(f"  {name:16s}  {desc}")
        if args.lab is None and not args.list:
            print("\nPick a lab, e.g.: python examples/tutorials/foundations/lab_runner.py layer")
            return 0
        if args.list:
            return 0

    runners: Dict[str, Callable[[], None]] = {
        "parallel-state": lab_parallel_state,
        "layer": lambda: lab_layer(
            args.hidden_size, args.seq_len, args.batch_size, args.heads
        ),
    }
    try:
        runners[args.lab]()
    except Exception as exc:  # noqa: BLE001 - educational surface
        print(f"Lab failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Tip: install editable package (`pip install -e .` / project install docs) "
            "and use a CUDA GPU.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
