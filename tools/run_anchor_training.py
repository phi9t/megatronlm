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
    preflight: bool = False


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


def build_preflight_command() -> str:
    """Build environment checks that run before the preflight mini-train."""

    return (
        "python - <<'PY'\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "import torch\n"
        "import transformer_engine\n"
        "\n"
        "assert os.environ['HOME'] == '/outputs/home'\n"
        "assert os.environ['USER']\n"
        "assert os.environ['LOGNAME']\n"
        "assert os.environ['TORCHINDUCTOR_CACHE_DIR'] == '/outputs/cache/torchinductor'\n"
        "assert os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] == '1'\n"
        "assert torch.cuda.is_available()\n"
        "assert torch.cuda.device_count() >= 1\n"
        "\n"
        "for path in ('/outputs/home', '/outputs/cache/torchinductor'):\n"
        "    Path(path).mkdir(parents=True, exist_ok=True)\n"
        "    marker = Path(path) / 'preflight-write-check'\n"
        "    marker.write_text('ok', encoding='utf-8')\n"
        "    marker.unlink()\n"
        "\n"
        "te_version = getattr(transformer_engine, '__version__', 'unknown')\n"
        "print(f'preflight: torch={torch.__version__} cuda_devices={torch.cuda.device_count()}')\n"
        "print(f'preflight: transformer_engine={te_version}')\n"
        "PY\n"
        "cat > /outputs/work/preflight_dist.py <<'PY'\n"
        "import torch\n"
        "import torch.distributed as dist\n"
        "\n"
        "assert torch.cuda.is_available()\n"
        "dist.init_process_group(backend='nccl')\n"
        "rank = dist.get_rank()\n"
        "device = rank % torch.cuda.device_count()\n"
        "torch.cuda.set_device(device)\n"
        "value = torch.ones(1, device='cuda')\n"
        "dist.all_reduce(value)\n"
        "assert value.item() == dist.get_world_size()\n"
        "dist.destroy_process_group()\n"
        "print('preflight: torch.distributed nccl ok')\n"
        "PY\n"
        "torchrun --nproc_per_node=1 /outputs/work/preflight_dist.py"
    )


def build_inner_shell_command(config: AnchorRunConfig) -> str:
    """Build the shell command executed by Docker."""

    commands = [build_training_command(config)]
    if config.preflight:
        commands.insert(0, build_preflight_command())

    return (
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
        'cd "$run_dir"; ' + "; ".join(commands)
    )


def build_docker_command(
    config: AnchorRunConfig, user: UserInfo, detach: bool
) -> list[str]:
    """Build the Docker command for the anchor run."""

    shell_command = build_inner_shell_command(config)
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
    parser.add_argument("--preflight", action="store_true")
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
        preflight=args.preflight,
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
