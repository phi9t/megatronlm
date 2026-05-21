# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import sys
from pathlib import Path

import tomllib

MODULE_PATH = Path("examples/llama/run_llama_anchor.py")
SPEC = importlib.util.spec_from_file_location("run_llama_anchor", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
llama_anchor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = llama_anchor
SPEC.loader.exec_module(llama_anchor)


def test_docker_command_sets_user_identity_for_host_uid() -> None:
    config = llama_anchor.LlamaAnchorConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/local/llama"),
        container_name="llama",
        image="megatron-lm:smoke",
    )
    user = llama_anchor.UserInfo(uid=1018, gid=100, name="philip")

    command = llama_anchor.build_docker_command(config, user, detach=True)

    assert "--user" in command
    assert "1018:100" in command
    assert "USER=philip" in command
    assert "LOGNAME=philip" in command
    assert "TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor" in command


def test_docker_command_keeps_source_readonly_and_runs_from_copy() -> None:
    config = llama_anchor.LlamaAnchorConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/local/llama"),
        container_name="llama",
        image="megatron-lm:smoke",
    )

    command = llama_anchor.build_docker_command(
        config, llama_anchor.UserInfo(uid=1, gid=2, name="user"), detach=True
    )
    shell_command = command[-1]

    assert "/repo:/workspace:ro" in command
    assert "/repo/local/llama:/outputs" in command
    assert "-w" in command
    assert "/outputs" in command
    assert "mktemp -d /outputs/work/megatron-lm.XXXXXXXX" in shell_command
    assert "--exclude=local" in shell_command
    assert "--exclude=.agents" in shell_command
    assert "--exclude=.cache" in shell_command
    assert "--exclude=.venv" in shell_command
    assert "--exclude=.git" in shell_command
    assert "-C /workspace -cf -" in shell_command
    assert 'tar -C "$run_dir" -xf -' in shell_command
    assert 'cd "$run_dir"' in shell_command


def test_training_command_uses_small_llama_mock_data_shape() -> None:
    config = llama_anchor.LlamaAnchorConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/local/llama"),
        container_name="llama",
        image="megatron-lm:smoke",
        duration_mins=130,
        train_iters=500_000,
    )

    command = llama_anchor.build_docker_command(
        config, llama_anchor.UserInfo(uid=1, gid=2, name="user"), detach=True
    )
    shell_command = command[-1]

    assert "--use-mcore-models" in shell_command
    assert "--position-embedding-type rope" in shell_command
    assert "--normalization RMSNorm" in shell_command
    assert "--swiglu" in shell_command
    assert "--group-query-attention" in shell_command
    assert "--num-query-groups 4" in shell_command
    assert "--mock-data" in shell_command
    assert "--data-cache-path /outputs/cache/data" in shell_command
    assert "--exit-duration-in-mins 130" in shell_command
    assert "--train-iters 500000" in shell_command
    assert "--lr-decay-iters 500000" in shell_command


def test_preflight_command_checks_training_environment_fundamentals() -> None:
    preflight_command = llama_anchor.build_preflight_command()

    assert "import torch" in preflight_command
    assert "import transformer_engine" in preflight_command
    assert "torch.cuda.is_available()" in preflight_command
    assert "torch.cuda.device_count() >= 1" in preflight_command
    assert "TORCHINDUCTOR_CACHE_DIR" in preflight_command
    assert "CUDA_DEVICE_MAX_CONNECTIONS" in preflight_command
    assert "preflight-write-check" in preflight_command
    assert "dist.init_process_group(backend='nccl')" in preflight_command
    assert "dist.all_reduce(value)" in preflight_command
    assert (
        "torchrun --nproc_per_node=1 /outputs/work/preflight_dist.py"
        in preflight_command
    )
    assert "/tmp" not in preflight_command


def test_preflight_docker_command_runs_checks_before_mini_train() -> None:
    config = llama_anchor.LlamaAnchorConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/local/llama-preflight"),
        container_name="llama-preflight",
        image="megatron-lm:smoke",
        duration_mins=5,
        train_iters=2,
        preflight=True,
    )

    command = llama_anchor.build_docker_command(
        config, llama_anchor.UserInfo(uid=1, gid=2, name="user"), detach=False
    )
    shell_command = command[-1]

    assert "-d" not in command
    assert "--name" in command
    assert "llama-preflight" in command
    assert "/repo/local/llama-preflight:/outputs" in command
    assert shell_command.index("preflight: torch=") < shell_command.index(
        "pretrain_gpt.py"
    )
    assert (
        "torchrun --nproc_per_node=1 /outputs/work/preflight_dist.py" in shell_command
    )
    assert "--train-iters 2" in shell_command
    assert "--exit-duration-in-mins 5" in shell_command
    assert "/tmp" not in " ".join(command)


def test_remove_container_command_targets_configured_name() -> None:
    assert llama_anchor.build_remove_container_command("llama") == [
        "docker",
        "rm",
        "-f",
        "llama",
    ]


def test_docker_command_does_not_use_tmp_paths() -> None:
    config = llama_anchor.LlamaAnchorConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/local/llama"),
        container_name="llama",
        image="megatron-lm:smoke",
    )

    command = llama_anchor.build_docker_command(
        config, llama_anchor.UserInfo(uid=1, gid=2, name="user"), detach=True
    )

    assert "/tmp" not in " ".join(command)


def test_mise_llama_anchor_tasks_launch_repo_local_runner() -> None:
    mise_config = tomllib.loads(Path("mise.toml").read_text())

    assert mise_config["tasks"]["llama-anchor"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama_anchor.py"
    )
    assert mise_config["tasks"]["llama-anchor-smoke"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama_anchor.py "
        "--name mcore-llama-anchor-smoke "
        "--output-dir local/llama-anchor-smoke --train-iters 2 --foreground"
    )
    assert mise_config["tasks"]["llama-anchor-preflight"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama_anchor.py "
        "--preflight --name mcore-llama-anchor-preflight "
        "--output-dir local/llama-anchor-preflight "
        "--duration-mins 5 --train-iters 2 --foreground"
    )
    assert "/tmp" not in mise_config["tasks"]["llama-anchor"]["run"]
    assert "/tmp" not in mise_config["tasks"]["llama-anchor-smoke"]["run"]
    assert "/tmp" not in mise_config["tasks"]["llama-anchor-preflight"]["run"]
