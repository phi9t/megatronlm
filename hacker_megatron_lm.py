#!/usr/bin/env python3
"""
Megatron-LM Hacker Script
=========================

Interactive exploration of Megatron-LM components for understanding the codebase.
Run individual modules in memory using one or multiple GPUs.

Usage:
------
Single GPU (no distributed):
    python hacker_megatron_lm.py

Multiple GPUs with Tensor Parallelism:
    torchrun --nproc_per_node=2 hacker_megatron_lm.py --tp-size 2

Multiple GPUs with Pipeline Parallelism:
    torchrun --nproc_per_node=4 hacker_megatron_lm.py --pp-size 4

Combined Parallelism:
    torchrun --nproc_per_node=8 hacker_megatron_lm.py --tp-size 2 --pp-size 4

Interactive Mode:
    python hacker_megatron_lm.py --interactive

"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple, Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utility Functions
# =============================================================================

def print_rank_0(message: str):
    """Print only on rank 0 to avoid duplicate output."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(message)


def print_separator(title: str = "", char: str = "=", width: int = 80):
    """Print a visual separator."""
    if title:
        padding = (width - len(title) - 2) // 2
        print_rank_0(f"\n{char * padding} {title} {char * padding}")
    else:
        print_rank_0(char * width)


def print_tensor_info(name: str, tensor: torch.Tensor, show_stats: bool = True):
    """Print tensor shape and optionally statistics."""
    info = f"{name}: shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
    if show_stats and tensor.numel() > 0:
        info += f", min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.float().mean().item():.4f}"
    print_rank_0(info)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(num_params: int) -> str:
    """Format parameter count for readability."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


# =============================================================================
# Distributed Setup
# =============================================================================

def initialize_distributed(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> bool:
    """
    Initialize torch.distributed and Megatron-Core model parallel groups.

    Returns True if running distributed, False for single-GPU mode.
    """
    from megatron.core import parallel_state

    # Clean up any existing state
    parallel_state.destroy_model_parallel()

    # Check if we're in a distributed environment
    local_rank = os.environ.get("LOCAL_RANK")

    if local_rank is not None:
        # Distributed mode via torchrun
        rank = int(local_rank)
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

        torch.cuda.set_device(rank)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=world_size,
                rank=rank,
            )

        # Initialize Megatron model parallel groups
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
        )

        print_rank_0(f"Distributed initialized: world_size={world_size}, TP={tensor_model_parallel_size}, PP={pipeline_model_parallel_size}")
        return True
    else:
        # Single GPU mode - still need to initialize parallel state
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        # Initialize with minimal setup for single GPU
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        print_rank_0("Single GPU mode initialized")
        return False


def cleanup_distributed():
    """Clean up distributed resources."""
    from megatron.core import parallel_state
    parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# =============================================================================
# Demo 1: Explore TransformerConfig
# =============================================================================

def demo_transformer_config():
    """Demonstrate TransformerConfig and its options."""
    print_separator("TransformerConfig Exploration")

    from megatron.core.transformer.transformer_config import TransformerConfig

    # Create a small configuration
    config = TransformerConfig(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=1024,  # 4x hidden_size
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        layernorm_epsilon=1e-5,
    )

    print_rank_0("\nTransformerConfig created:")
    print_rank_0(f"  num_layers: {config.num_layers}")
    print_rank_0(f"  hidden_size: {config.hidden_size}")
    print_rank_0(f"  num_attention_heads: {config.num_attention_heads}")
    print_rank_0(f"  ffn_hidden_size: {config.ffn_hidden_size}")
    print_rank_0(f"  kv_channels: {config.kv_channels} (hidden_size / num_attention_heads)")
    print_rank_0(f"  tensor_model_parallel_size: {config.tensor_model_parallel_size}")
    print_rank_0(f"  pipeline_model_parallel_size: {config.pipeline_model_parallel_size}")
    print_rank_0(f"  add_bias_linear: {config.add_bias_linear}")

    return config


# =============================================================================
# Demo 2: Individual Transformer Components
# =============================================================================

def demo_attention_module(config=None):
    """Demonstrate the SelfAttention module."""
    print_separator("SelfAttention Module")

    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.enums import AttnMaskType

    if config is None:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )

    # Create attention submodules spec
    submodules = SelfAttentionSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=DotProductAttention,
        linear_proj=RowParallelLinear,
    )

    # Create attention module
    attention = SelfAttention(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = attention.to(device)

    print_rank_0(f"\nSelfAttention module created:")
    print_rank_0(f"  Parameters: {format_params(count_parameters(attention))}")
    print_rank_0(f"  Device: {device}")

    # Run forward pass
    batch_size, seq_len = 2, 64
    hidden_states = torch.randn(seq_len, batch_size, config.hidden_size, device=device)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, device=device)

    print_rank_0(f"\nInput hidden_states: {list(hidden_states.shape)} [seq, batch, hidden]")

    with torch.no_grad():
        output, bias = attention(hidden_states, attention_mask)

    print_rank_0(f"Output: {list(output.shape)} [seq, batch, hidden]")
    print_tensor_info("Output stats", output)

    return attention


def demo_mlp_module(config=None):
    """Demonstrate the MLP module."""
    print_separator("MLP (Feed-Forward) Module")

    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    if config is None:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            ffn_hidden_size=1024,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
        )

    # Create MLP submodules
    submodules = MLPSubmodules(
        linear_fc1=ColumnParallelLinear,
        linear_fc2=RowParallelLinear,
    )

    # Create MLP module
    mlp = MLP(config=config, submodules=submodules)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = mlp.to(device)

    print_rank_0(f"\nMLP module created:")
    print_rank_0(f"  Parameters: {format_params(count_parameters(mlp))}")
    print_rank_0(f"  FC1: {config.hidden_size} -> {config.ffn_hidden_size}")
    print_rank_0(f"  FC2: {config.ffn_hidden_size} -> {config.hidden_size}")

    # Run forward pass
    batch_size, seq_len = 2, 64
    hidden_states = torch.randn(seq_len, batch_size, config.hidden_size, device=device)

    print_rank_0(f"\nInput: {list(hidden_states.shape)}")

    with torch.no_grad():
        output, bias = mlp(hidden_states)

    print_rank_0(f"Output: {list(output.shape)}")
    print_tensor_info("Output stats", output)

    return mlp


def demo_transformer_layer(config=None):
    """Demonstrate a complete TransformerLayer."""
    print_separator("TransformerLayer (Attention + MLP)")

    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.spec_utils import build_module

    if config is None:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            ffn_hidden_size=1024,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )

    # Get the layer spec (pure PyTorch implementation)
    layer_spec = get_gpt_layer_local_spec()

    # Build the transformer layer from spec
    layer = build_module(layer_spec, config=config, layer_number=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = layer.to(device)

    print_rank_0(f"\nTransformerLayer created from spec:")
    print_rank_0(f"  Parameters: {format_params(count_parameters(layer))}")
    print_rank_0(f"  Spec type: local (pure PyTorch)")

    # Print layer structure
    print_rank_0("\nLayer structure:")
    for name, module in layer.named_children():
        param_count = count_parameters(module)
        print_rank_0(f"  {name}: {type(module).__name__} ({format_params(param_count)})")

    # Run forward pass
    batch_size, seq_len = 2, 64
    hidden_states = torch.randn(seq_len, batch_size, config.hidden_size, device=device)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, device=device)

    print_rank_0(f"\nInput: {list(hidden_states.shape)} [seq, batch, hidden]")

    with torch.no_grad():
        output, _ = layer(hidden_states, attention_mask=attention_mask)

    print_rank_0(f"Output: {list(output.shape)}")
    print_tensor_info("Output stats", output)

    return layer


# =============================================================================
# Demo 3: GPT Model
# =============================================================================

def demo_gpt_model(config=None, vocab_size: int = 1000, seq_len: int = 64):
    """Demonstrate building and running a GPT model."""
    print_separator("GPT Model")

    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if config is None:
        config = TransformerConfig(
            num_layers=4,
            hidden_size=256,
            num_attention_heads=8,
            ffn_hidden_size=1024,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            pipeline_dtype=torch.float32,
        )

    # Seed for reproducibility with tensor parallelism
    model_parallel_cuda_manual_seed(42)

    # Build GPT model
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        pre_process=True,   # Include embedding layer
        post_process=True,  # Include output layer (LM head)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print_rank_0(f"\nGPTModel created:")
    print_rank_0(f"  Parameters: {format_params(count_parameters(model))}")
    print_rank_0(f"  Vocab size: {vocab_size}")
    print_rank_0(f"  Max sequence length: {seq_len}")
    print_rank_0(f"  Num layers: {config.num_layers}")
    print_rank_0(f"  Hidden size: {config.hidden_size}")
    print_rank_0(f"  Attention heads: {config.num_attention_heads}")

    # Print model structure
    print_rank_0("\nModel structure:")
    for name, module in model.named_children():
        param_count = count_parameters(module)
        print_rank_0(f"  {name}: {type(module).__name__} ({format_params(param_count)})")

    # Run forward pass
    batch_size = 2
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    print_rank_0(f"\nForward pass:")
    print_rank_0(f"  Input tokens: {list(tokens.shape)} [batch, seq]")

    with torch.no_grad():
        output = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

    print_rank_0(f"  Output logits: {list(output.shape)} [seq, batch, vocab]")
    print_tensor_info("  Logits stats", output)

    return model


# =============================================================================
# Demo 4: Forward-Backward Pass with Gradients
# =============================================================================

def demo_training_step(config=None, vocab_size: int = 1000, seq_len: int = 64):
    """Demonstrate a complete training step with gradients."""
    print_separator("Training Step (Forward + Backward)")

    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if config is None:
        config = TransformerConfig(
            num_layers=2,  # Small for demo
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=512,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            pipeline_dtype=torch.float32,
        )

    model_parallel_cuda_manual_seed(42)

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        pre_process=True,
        post_process=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    print_rank_0(f"\nModel: {format_params(count_parameters(model))} parameters")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create batch
    batch_size = 4
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = tokens.clone()  # For language modeling, predict the same tokens

    print_rank_0(f"Batch: {batch_size} sequences of length {seq_len}")

    # Forward pass
    print_rank_0("\n1. Forward pass...")
    optimizer.zero_grad()

    output = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    # Output is loss when labels provided, or logits otherwise
    # For demo, compute loss manually
    if output.dim() == 3:
        # Got logits [seq, batch, vocab]
        logits = output.transpose(0, 1)  # [batch, seq, vocab]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
        )
    else:
        loss = output.mean()  # Already loss

    print_rank_0(f"  Loss: {loss.item():.4f}")

    # Backward pass
    print_rank_0("\n2. Backward pass...")
    loss.backward()

    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    print_rank_0("\n3. Gradient norms (sample):")
    for i, (name, norm) in enumerate(grad_norms.items()):
        if i < 5:  # Show first 5
            print_rank_0(f"  {name}: {norm:.6f}")
    print_rank_0(f"  ... ({len(grad_norms)} parameters total)")

    # Optimizer step
    print_rank_0("\n4. Optimizer step...")
    optimizer.step()

    print_rank_0("Training step complete!")

    return model, loss.item()


# =============================================================================
# Demo 5: Tensor Parallelism Exploration
# =============================================================================

def demo_tensor_parallelism():
    """Demonstrate tensor parallel operations."""
    print_separator("Tensor Parallelism")

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
        VocabParallelEmbedding,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    print_rank_0(f"\nTensor Parallel Configuration:")
    print_rank_0(f"  TP Size: {tp_size}")
    print_rank_0(f"  TP Rank: {tp_rank}")

    if tp_size == 1:
        print_rank_0("\n  Note: Running with TP=1. Use 'torchrun --nproc_per_node=2 hacker_megatron_lm.py --tp-size 2' for actual TP.")

    # Create config
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        tensor_model_parallel_size=tp_size,
        use_cpu_initialization=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Demo ColumnParallelLinear
    print_rank_0("\n1. ColumnParallelLinear (splits output features):")
    input_size, output_size = 256, 1024
    col_linear = ColumnParallelLinear(
        input_size=input_size,
        output_size=output_size,
        config=config,
        init_method=lambda x: torch.nn.init.xavier_uniform_(x),
        bias=False,
        gather_output=True,  # Gather results from all TP ranks
    ).to(device)

    local_output_size = output_size // tp_size
    print_rank_0(f"  Full output: {output_size}, Local output per rank: {local_output_size}")

    x = torch.randn(4, 32, input_size, device=device)
    with torch.no_grad():
        y, _ = col_linear(x)
    print_rank_0(f"  Input: {list(x.shape)} -> Output: {list(y.shape)}")

    # Demo RowParallelLinear
    print_rank_0("\n2. RowParallelLinear (splits input features):")
    row_linear = RowParallelLinear(
        input_size=1024,
        output_size=256,
        config=config,
        init_method=lambda x: torch.nn.init.xavier_uniform_(x),
        bias=False,
        input_is_parallel=False,
    ).to(device)

    x = torch.randn(4, 32, 1024, device=device)
    with torch.no_grad():
        y, _ = row_linear(x)
    print_rank_0(f"  Input: {list(x.shape)} -> Output: {list(y.shape)}")

    # Demo VocabParallelEmbedding
    print_rank_0("\n3. VocabParallelEmbedding (splits vocabulary):")
    vocab_size, embedding_dim = 10000, 256
    embedding = VocabParallelEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        config=config,
        init_method=lambda x: torch.nn.init.normal_(x, std=0.02),
    ).to(device)

    local_vocab = vocab_size // tp_size
    print_rank_0(f"  Full vocab: {vocab_size}, Local vocab per rank: {local_vocab}")

    tokens = torch.randint(0, vocab_size, (4, 32), device=device)
    with torch.no_grad():
        embeddings = embedding(tokens)
    print_rank_0(f"  Input tokens: {list(tokens.shape)} -> Embeddings: {list(embeddings.shape)}")


# =============================================================================
# Demo 6: Pipeline Parallelism Exploration
# =============================================================================

def demo_pipeline_parallelism():
    """Demonstrate pipeline parallel concepts."""
    print_separator("Pipeline Parallelism")

    from megatron.core import parallel_state

    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    print_rank_0(f"\nPipeline Parallel Configuration:")
    print_rank_0(f"  PP Size: {pp_size}")
    print_rank_0(f"  PP Rank: {pp_rank}")

    if pp_size == 1:
        print_rank_0("\n  Note: Running with PP=1. Use 'torchrun --nproc_per_node=4 hacker_megatron_lm.py --pp-size 4' for actual PP.")
        return

    # Determine what this rank should process
    is_first_stage = parallel_state.is_pipeline_first_stage()
    is_last_stage = parallel_state.is_pipeline_last_stage()

    print_rank_0(f"\nThis rank (PP={pp_rank}):")
    print_rank_0(f"  is_first_stage (has embedding): {is_first_stage}")
    print_rank_0(f"  is_last_stage (has LM head): {is_last_stage}")

    # Demo creating model with PP-aware pre/post processing
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    num_layers = 8  # Will be split across PP stages
    layers_per_stage = num_layers // pp_size

    config = TransformerConfig(
        num_layers=layers_per_stage,  # Each stage gets a portion
        hidden_size=256,
        num_attention_heads=8,
        pipeline_model_parallel_size=pp_size,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )

    print_rank_0(f"\nModel configuration for PP={pp_size}:")
    print_rank_0(f"  Total layers: {num_layers}")
    print_rank_0(f"  Layers per stage: {layers_per_stage}")

    # Create model with appropriate pre/post processing
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=1000,
        max_sequence_length=64,
        pre_process=is_first_stage,   # Only first stage has embedding
        post_process=is_last_stage,   # Only last stage has LM head
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print_rank_0(f"\nModel on PP rank {pp_rank}:")
    print_rank_0(f"  Parameters: {format_params(count_parameters(model))}")
    print_rank_0(f"  Has embedding: {is_first_stage}")
    print_rank_0(f"  Has LM head: {is_last_stage}")


# =============================================================================
# Demo 7: Distributed Data Parallel
# =============================================================================

def demo_ddp():
    """Demonstrate DistributedDataParallel wrapping."""
    print_separator("Distributed Data Parallel (DDP)")

    from megatron.core import parallel_state
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.distributed import DistributedDataParallelConfig

    if not torch.distributed.is_initialized():
        print_rank_0("\nDDP requires distributed initialization. Skipping in single-GPU mode.")
        return

    dp_size = parallel_state.get_data_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()

    print_rank_0(f"\nData Parallel Configuration:")
    print_rank_0(f"  DP Size: {dp_size}")
    print_rank_0(f"  DP Rank: {dp_rank}")

    # Create a small model
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=1000,
        max_sequence_length=64,
        pre_process=True,
        post_process=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Wrap with DDP
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )

    ddp_model = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=model,
    )

    print_rank_0(f"\nDDP-wrapped model:")
    print_rank_0(f"  Original parameters: {format_params(count_parameters(model))}")
    print_rank_0(f"  grad_reduce_in_fp32: {ddp_config.grad_reduce_in_fp32}")
    print_rank_0(f"  overlap_grad_reduce: {ddp_config.overlap_grad_reduce}")

    return ddp_model


# =============================================================================
# Demo 8: Layer Specs Comparison
# =============================================================================

def demo_layer_specs():
    """Compare different layer spec implementations."""
    print_separator("Layer Specification System")

    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
    )
    from megatron.core.transformer.spec_utils import ModuleSpec

    print_rank_0("\nAvailable layer specs:")
    print_rank_0("  1. get_gpt_layer_local_spec() - Pure PyTorch implementation")
    print_rank_0("  2. get_gpt_layer_with_transformer_engine_spec() - Transformer Engine (requires TE)")
    print_rank_0("  3. get_gpt_layer_with_inference_spec() - Inference optimized (requires TE)")

    # Examine local spec structure
    spec = get_gpt_layer_local_spec()

    print_rank_0(f"\nLocal spec structure (get_gpt_layer_local_spec):")
    print_rank_0(f"  Module: {spec.module.__name__}")

    if hasattr(spec, 'submodules') and spec.submodules is not None:
        print_rank_0(f"  Submodules:")
        for field_name in spec.submodules.__dataclass_fields__:
            submodule = getattr(spec.submodules, field_name)
            if submodule is not None:
                if isinstance(submodule, ModuleSpec):
                    print_rank_0(f"    {field_name}: {submodule.module.__name__}")
                else:
                    print_rank_0(f"    {field_name}: {submodule.__name__ if hasattr(submodule, '__name__') else type(submodule).__name__}")


# =============================================================================
# Demo 9: Checkpoint State Dict
# =============================================================================

def demo_checkpointing():
    """Demonstrate checkpoint state dict structure."""
    print_separator("Checkpointing")

    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=1000,
        max_sequence_length=64,
        pre_process=True,
        post_process=True,
    )

    # Get standard PyTorch state dict
    state_dict = model.state_dict()

    print_rank_0(f"\nPyTorch state_dict keys ({len(state_dict)} tensors):")
    for i, (key, tensor) in enumerate(state_dict.items()):
        if i < 10:  # Show first 10
            print_rank_0(f"  {key}: {list(tensor.shape)}")
    if len(state_dict) > 10:
        print_rank_0(f"  ... ({len(state_dict) - 10} more)")

    # Get sharded state dict (for distributed checkpointing)
    try:
        sharded_state_dict = model.sharded_state_dict(prefix="")
        print_rank_0(f"\nSharded state_dict keys ({len(sharded_state_dict)} entries):")
        for i, key in enumerate(sharded_state_dict.keys()):
            if i < 5:
                print_rank_0(f"  {key}")
        if len(sharded_state_dict) > 5:
            print_rank_0(f"  ... ({len(sharded_state_dict) - 5} more)")
    except Exception as e:
        print_rank_0(f"\nSharded state dict not available: {e}")


# =============================================================================
# Interactive Mode
# =============================================================================

def interactive_mode():
    """Run in interactive mode for exploration."""
    print_separator("Interactive Mode")

    print_rank_0("""
Available objects for exploration:
  - parallel_state: Megatron parallel state module
  - TransformerConfig: Configuration class
  - GPTModel: GPT model class
  - get_gpt_layer_local_spec: Layer spec function

Available demo functions:
  - demo_transformer_config()
  - demo_attention_module()
  - demo_mlp_module()
  - demo_transformer_layer()
  - demo_gpt_model()
  - demo_training_step()
  - demo_tensor_parallelism()
  - demo_pipeline_parallelism()
  - demo_ddp()
  - demo_layer_specs()
  - demo_checkpointing()

Type 'exit' or 'quit' to exit.
""")

    # Import commonly used modules for interactive use
    from megatron.core import parallel_state
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    # Make them available in local scope
    local_vars = {
        'parallel_state': parallel_state,
        'TransformerConfig': TransformerConfig,
        'GPTModel': GPTModel,
        'get_gpt_layer_local_spec': get_gpt_layer_local_spec,
        'torch': torch,
        'demo_transformer_config': demo_transformer_config,
        'demo_attention_module': demo_attention_module,
        'demo_mlp_module': demo_mlp_module,
        'demo_transformer_layer': demo_transformer_layer,
        'demo_gpt_model': demo_gpt_model,
        'demo_training_step': demo_training_step,
        'demo_tensor_parallelism': demo_tensor_parallelism,
        'demo_pipeline_parallelism': demo_pipeline_parallelism,
        'demo_ddp': demo_ddp,
        'demo_layer_specs': demo_layer_specs,
        'demo_checkpointing': demo_checkpointing,
    }

    import code
    code.interact(local=local_vars, banner="")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Megatron-LM Hacker Script - Interactive exploration of Megatron components"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1,
        help="Tensor parallel size (default: 1)"
    )
    parser.add_argument(
        "--pp-size", type=int, default=1,
        help="Pipeline parallel size (default: 1)"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--demo", type=str, default="all",
        choices=["all", "config", "attention", "mlp", "layer", "model", "training", "tp", "pp", "ddp", "specs", "checkpoint"],
        help="Which demo to run (default: all)"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=1000,
        help="Vocabulary size for model demos (default: 1000)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=64,
        help="Sequence length for demos (default: 64)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print_separator("Megatron-LM Hacker Script", char="*")
    print_rank_0(f"PyTorch version: {torch.__version__}")
    print_rank_0(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print_rank_0(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print_rank_0(f"CUDA devices: {torch.cuda.device_count()}")

    # Initialize distributed if needed
    try:
        is_distributed = initialize_distributed(
            tensor_model_parallel_size=args.tp_size,
            pipeline_model_parallel_size=args.pp_size,
        )
    except Exception as e:
        print_rank_0(f"Warning: Could not initialize distributed: {e}")
        print_rank_0("Continuing in single-GPU mode...")
        is_distributed = False

    try:
        if args.interactive:
            interactive_mode()
        else:
            # Run selected demos
            demos = {
                "config": demo_transformer_config,
                "attention": demo_attention_module,
                "mlp": demo_mlp_module,
                "layer": demo_transformer_layer,
                "model": lambda: demo_gpt_model(vocab_size=args.vocab_size, seq_len=args.seq_len),
                "training": lambda: demo_training_step(vocab_size=args.vocab_size, seq_len=args.seq_len),
                "tp": demo_tensor_parallelism,
                "pp": demo_pipeline_parallelism,
                "ddp": demo_ddp,
                "specs": demo_layer_specs,
                "checkpoint": demo_checkpointing,
            }

            if args.demo == "all":
                # Run all demos except ddp and pp which require distributed
                demo_order = ["config", "attention", "mlp", "layer", "model", "training", "specs", "checkpoint"]
                if is_distributed:
                    demo_order.extend(["tp", "pp", "ddp"])
                else:
                    demo_order.append("tp")  # TP works in single-GPU mode too

                for demo_name in demo_order:
                    try:
                        demos[demo_name]()
                    except Exception as e:
                        print_rank_0(f"\nError in {demo_name} demo: {e}")
            else:
                demos[args.demo]()

        print_separator("Done", char="*")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
