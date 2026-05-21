# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from pathlib import Path

EXAMPLES_LLAMA = Path("examples/llama").resolve()
if str(EXAMPLES_LLAMA) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_LLAMA))

from llama_config import UserInfo, build_docker_command, load_config


def test_llama3_8b_yaml_preset_renders_fp8_long_run_shape() -> None:
    config = load_config(Path("examples/llama/configs/llama3_8b_long.yaml"))

    command = build_docker_command(
        config, Path("/repo"), UserInfo(uid=1, gid=2, name="user")
    )
    shell_command = command[-1]

    assert "-d" in command
    assert "torchrun --nproc_per_node=8 pretrain_gpt.py" in shell_command
    assert "--num-layers 32" in shell_command
    assert "--hidden-size 4096" in shell_command
    assert "--ffn-hidden-size 14336" in shell_command
    assert "--num-attention-heads 32" in shell_command
    assert "--num-query-groups 8" in shell_command
    assert "--seq-length 8192" in shell_command
    assert "--bf16" in shell_command
    assert "--transformer-impl transformer_engine" in shell_command
    assert "--fp8-format hybrid" in shell_command
    assert "--fp8-param-gather" in shell_command
    assert "--global-batch-size 128" in shell_command
    assert "--exit-duration-in-mins 235" in shell_command
    assert "--save /outputs/checkpoints" in shell_command
    assert "--load /outputs/checkpoints" not in shell_command
    assert "--mock-data" in shell_command
    assert "--vocab-size 128256" in shell_command
    assert "/tmp" not in " ".join(command)


def test_llama3_8b_preflight_yaml_checks_fp8_and_prints_training_command() -> None:
    config = load_config(Path("examples/llama/configs/llama3_8b_preflight.yaml"))

    command = build_docker_command(
        config, Path("/repo"), UserInfo(uid=1, gid=2, name="user")
    )
    shell_command = command[-1]

    assert "-d" not in command
    assert "torch.cuda.device_count() >= 8" in shell_command
    assert "torch.cuda.get_device_capability(0)" in shell_command
    assert "FP8 requires Hopper, Ada, or Blackwell GPUs" in shell_command
    assert (
        "torchrun --nproc_per_node=8 /outputs/work/preflight_dist.py" in shell_command
    )
    assert "preflight: pretrain_gpt.py found" in shell_command
    assert "preflight: training command:" in shell_command
    assert shell_command.count("torchrun --nproc_per_node=8 pretrain_gpt.py") == 1
