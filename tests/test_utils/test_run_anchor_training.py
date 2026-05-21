# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pathlib import Path

import tomllib

from tools.run_anchor_training import (
    AnchorRunConfig,
    UserInfo,
    build_docker_command,
    build_preflight_command,
    build_remove_container_command,
)


def test_docker_command_sets_user_identity_for_host_uid() -> None:
    config = AnchorRunConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/local/anchor"),
        container_name="anchor",
        image="megatron-lm:smoke",
    )
    user = UserInfo(uid=1018, gid=100, name="philip")

    command = build_docker_command(config, user, detach=True)

    assert "--user" in command
    assert "1018:100" in command
    assert "USER=philip" in command
    assert "LOGNAME=philip" in command
    assert "TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor" in command


def test_docker_command_keeps_source_readonly_and_runs_from_copy() -> None:
    config = AnchorRunConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/local/anchor"),
        container_name="anchor",
        image="megatron-lm:smoke",
    )

    command = build_docker_command(
        config, UserInfo(uid=1, gid=2, name="user"), detach=True
    )
    shell_command = command[-1]

    assert "/repo:/workspace:ro" in command
    assert "/repo/local/anchor:/outputs" in command
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


def test_training_command_uses_duration_limit_and_bounded_iters() -> None:
    config = AnchorRunConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/local/anchor"),
        container_name="anchor",
        image="megatron-lm:smoke",
        duration_mins=130,
        train_iters=500_000,
    )

    command = build_docker_command(
        config, UserInfo(uid=1, gid=2, name="user"), detach=True
    )
    shell_command = command[-1]

    assert "--exit-duration-in-mins 130" in shell_command
    assert "--train-iters 500000" in shell_command
    assert "--lr-decay-iters 500000" in shell_command


def test_preflight_command_checks_training_environment_fundamentals() -> None:
    preflight_command = build_preflight_command()

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
    config = AnchorRunConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/local/anchor-preflight"),
        container_name="anchor-preflight",
        image="megatron-lm:smoke",
        duration_mins=5,
        train_iters=2,
        preflight=True,
    )

    command = build_docker_command(
        config, UserInfo(uid=1, gid=2, name="user"), detach=False
    )
    shell_command = command[-1]

    assert "-d" not in command
    assert "--name" in command
    assert "anchor-preflight" in command
    assert "/repo/local/anchor-preflight:/outputs" in command
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
    assert build_remove_container_command("anchor") == ["docker", "rm", "-f", "anchor"]


def test_docker_command_does_not_use_tmp_paths() -> None:
    config = AnchorRunConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/local/anchor"),
        container_name="anchor",
        image="megatron-lm:smoke",
    )

    command = build_docker_command(
        config, UserInfo(uid=1, gid=2, name="user"), detach=True
    )

    assert "/tmp" not in " ".join(command)


def test_mise_anchor_task_launches_repo_local_runner() -> None:
    mise_config = tomllib.loads(Path("mise.toml").read_text())

    task = mise_config["tasks"]["anchor-train"]

    assert task["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python tools/run_anchor_training.py"
    )
    assert "/tmp" not in task["run"]


def test_mise_preflight_task_runs_foreground_mini_train() -> None:
    mise_config = tomllib.loads(Path("mise.toml").read_text())

    task = mise_config["tasks"]["anchor-train-preflight"]

    assert task["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python tools/run_anchor_training.py "
        "--preflight --name mcore-anchor-preflight "
        "--output-dir local/anchor-training-preflight "
        "--duration-mins 5 --train-iters 2 --foreground"
    )
    assert "/tmp" not in task["run"]
