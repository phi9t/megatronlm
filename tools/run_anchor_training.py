# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run the local Megatron-LM anchor training job in Docker."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnchorRunConfig:
    """Configuration for the local anchor training run."""

    repo_root: Path
    output_dir: Path
    container_name: str = "mcore-anchor-long"
    image: str = "megatron-lm:smoke"
    duration_mins: int = 130
    train_iters: int = 500_000


@dataclass(frozen=True)
class UserInfo:
    """Host user identity passed through to Docker."""

    uid: int
    gid: int
    name: str


def build_training_command(config: AnchorRunConfig) -> str:
    """Build the shell command executed inside the container."""

    args = [
        "torchrun",
        "--nproc_per_node=1",
        "pretrain_gpt.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "4",
        "--seq-length",
        "128",
        "--max-position-embeddings",
        "128",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "1",
        "--train-iters",
        str(config.train_iters),
        "--exit-duration-in-mins",
        str(config.duration_mins),
        "--lr-decay-iters",
        str(config.train_iters),
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "1024",
        "--mock-data",
        "--split",
        "949,50,1",
        "--distributed-backend",
        "nccl",
        "--lr",
        "1.0e-4",
        "--min-lr",
        "1.0e-5",
        "--lr-decay-style",
        "cosine",
        "--weight-decay",
        "0.01",
        "--clip-grad",
        "1.0",
        "--log-interval",
        "100",
        "--eval-interval",
        "10000",
        "--eval-iters",
        "5",
        "--save-interval",
        "1000000000",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--transformer-impl",
        "transformer_engine",
        "--use-mcore-models",
        "--bf16",
        "--tensorboard-dir",
        "/outputs/tensorboard",
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def build_docker_command(
    config: AnchorRunConfig, user: UserInfo, detach: bool
) -> list[str]:
    """Build the Docker command for the anchor run."""

    shell_command = (
        "set -euo pipefail; "
        "mkdir -p /outputs/work /outputs/home /outputs/cache/torchinductor; "
        "run_dir=$(mktemp -d /outputs/work/megatron-lm.XXXXXXXX); "
        "trap 'rm -rf \"$run_dir\"' EXIT; "
        "tar "
        "--exclude=.git "
        "--exclude=.venv "
        "--exclude=.agents "
        "--exclude=.cache "
        "--exclude=.pytest_cache "
        "--exclude=.ruff_cache "
        "--exclude=local "
        "--exclude='__pycache__' "
        "--exclude='*.pyc' "
        "--exclude='*.so' "
        '-C /workspace -cf - . | tar -C "$run_dir" -xf -; '
        'cd "$run_dir"; '
        f"{build_training_command(config)}"
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        config.container_name,
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--user",
        f"{user.uid}:{user.gid}",
        "-v",
        f"{config.repo_root}:/workspace:ro",
        "-v",
        f"{config.output_dir}:/outputs",
        "-w",
        "/outputs",
        "-e",
        "HOME=/outputs/home",
        "-e",
        f"USER={user.name}",
        "-e",
        f"LOGNAME={user.name}",
        "-e",
        "PIP_CONSTRAINT=",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor",
        "-e",
        "CUDA_DEVICE_MAX_CONNECTIONS=1",
        config.image,
        "bash",
        "-lc",
        shell_command,
    ]
    if detach:
        command.insert(2, "-d")
    return command


def build_remove_container_command(container_name: str) -> list[str]:
    """Build a command that removes a stale Docker container by name."""

    return ["docker", "rm", "-f", container_name]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="megatron-lm:smoke")
    parser.add_argument("--name", default="mcore-anchor-long")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("local/anchor-training")
    )
    parser.add_argument("--duration-mins", type=int, default=130)
    parser.add_argument("--train-iters", type=int, default=500_000)
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the anchor training Docker command."""

    args = parse_args()
    repo_root = Path.cwd().resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AnchorRunConfig(
        repo_root=repo_root,
        output_dir=output_dir,
        container_name=args.name,
        image=args.image,
        duration_mins=args.duration_mins,
        train_iters=args.train_iters,
    )
    user = UserInfo(
        uid=os.getuid(), gid=os.getgid(), name=os.environ.get("USER", "megatron")
    )
    command = build_docker_command(config, user, detach=not args.foreground)

    if args.dry_run:
        print(shlex.join(command))
        return

    subprocess.run(
        build_remove_container_command(config.container_name),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
