import os
import torch
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_layer import TransformerLayer

def hack_mcore():
    """
    A minimalist script to instantiate a Megatron Core Transformer Layer 
    and run a forward pass on a single GPU.
    """
    print("Initializing Hacker Mode...")

    # 1. Mock Distributed Setup for Single GPU
    # In a real run, these are set by torchrun
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6000'
    os.environ['LOCAL_RANK'] = '0'
    
    if not torch.cuda.is_available():
        print("Error: CUDA not available. Megatron Core requires a GPU.")
        return

    torch.cuda.set_device(0)
    
    # Initialize torch.distributed and MCore parallel state
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')
    
    # TP=1, PP=1
    parallel_state.initialize_model_parallel(1, 1)

    # 2. Configure a minimal Transformer
    # Most components in MCore expect a TransformerConfig
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        layernorm_epsilon=1e-5
    )

    # 3. Instantiate a single Transformer Layer
    # MCore uses 'specs' to build modules. We use the 'local' (standard PyTorch) spec.
    spec = get_gpt_layer_local_spec()
    
    # spec.submodules contains the specific implementations for Attention, MLP, etc.
    layer = TransformerLayer(
        config=config, 
        submodules=spec.submodules, 
        layer_number=1
    ).cuda()

    print(f"Hacker status: Layer initialized with {sum(p.numel() for p in layer.parameters())} params")

    # 4. Dummy Forward Pass
    # Input shape: [sequence_length, batch_size, hidden_size]
    seq_len = 32
    batch_size = 2
    hidden_states = torch.randn(seq_len, batch_size, config.hidden_size).cuda()
    
    # Simple causal mask or None for standard attention
    attention_mask = None 
    
    print(f"Running forward pass with input shape {hidden_states.shape}...")
    output = layer(hidden_states, attention_mask)
    
    print(f"Output shape: {output.shape}")
    print("Hack successful. You are now exploring the Core.")

if __name__ == "__main__":
    try:
        hack_mcore()
    except Exception as e:
        print(f"Hack failed: {e}")
        print("\nTip: Make sure you have installed Megatron-LM in editable mode: pip install -e .")
