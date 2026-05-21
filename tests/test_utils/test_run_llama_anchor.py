# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from pathlib import Path

EXAMPLES_LLAMA = Path("examples/llama").resolve()
if str(EXAMPLES_LLAMA) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_LLAMA))

from llama_config import UserInfo, build_docker_command, load_config


def test_anchor_yaml_preset_renders_small_mock_data_shape() -> None:
    config = load_config(Path("examples/llama/configs/llama_anchor.yaml"))

    shell_command = build_docker_command(
        config, Path("/repo"), UserInfo(uid=1, gid=2, name="user")
    )[-1]

    assert "torchrun --nproc_per_node=1 pretrain_gpt.py" in shell_command
    assert "--num-layers 4" in shell_command
    assert "--hidden-size 256" in shell_command
    assert "--ffn-hidden-size 896" in shell_command
    assert "--num-query-groups 4" in shell_command
    assert "--train-iters 500000" in shell_command
    assert "--lr-decay-iters 500000" in shell_command
    assert "--exit-duration-in-mins 130" in shell_command
    assert "--mock-data" in shell_command
    assert "--tokenizer-type NullTokenizer" in shell_command
    assert "--vocab-size 8192" in shell_command
    assert "--data-cache-path /outputs/cache/data" in shell_command


def test_anchor_preflight_yaml_runs_checks_before_mini_train() -> None:
    config = load_config(Path("examples/llama/configs/llama_anchor_preflight.yaml"))

    command = build_docker_command(
        config, Path("/repo"), UserInfo(uid=1, gid=2, name="user")
    )
    shell_command = command[-1]

    assert "-d" not in command
    assert "/repo:/workspace:ro" in command
    assert "/repo/local/llama-anchor-preflight:/outputs" in command
    assert shell_command.index("preflight: torch=") < shell_command.index(
        "pretrain_gpt.py"
    )
    assert (
        "torchrun --nproc_per_node=1 /outputs/work/preflight_dist.py" in shell_command
    )
    assert "--train-iters 2" in shell_command
    assert "--exit-duration-in-mins 5" in shell_command
    assert "/tmp" not in " ".join(command)
