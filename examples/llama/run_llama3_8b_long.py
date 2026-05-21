# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run a long local Llama-3 8B FP8 training job in Docker."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Llama3LongConfig:
    """Configuration for the local long-running Llama-3 8B run."""

    repo_root: Path
    output_dir: Path
    container_name: str = "mcore-llama3-8b-long"
    image: str = "megatron-lm:smoke"
    duration_mins: int = 235
    preflight: bool = False
    preflight_only: bool = False
    resume: bool = False
    tokenizer_model: str | None = None
    data_path: str | None = None


@dataclass(frozen=True)
class UserInfo:
    """Host user identity passed through to Docker."""

    uid: int
    gid: int
    name: str


def build_training_command(config: Llama3LongConfig) -> str:
    """Build the Llama-3 8B FP8 training command executed inside the container."""

    args = [
        "torchrun",
        "--nproc_per_node=8",
        "pretrain_gpt.py",
        "--use-mcore-models",
        "--num-layers",
        "32",
        "--hidden-size",
        "4096",
        "--ffn-hidden-size",
        "14336",
        "--num-attention-heads",
        "32",
        "--group-query-attention",
        "--num-query-groups",
        "8",
        "--kv-channels",
        "128",
        "--seq-length",
        "8192",
        "--max-position-embeddings",
        "8192",
        "--position-embedding-type",
        "rope",
        "--rotary-base",
        "1000000",
        "--rotary-percent",
        "1.0",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--swiglu",
        "--normalization",
        "RMSNorm",
        "--init-method-std",
        "0.0134",
        "--attention-backend",
        "fused",
        "--apply-layernorm-1p",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "128",
        "--train-samples",
        "1953125000",
        "--lr-decay-samples",
        "1949218748",
        "--lr-warmup-samples",
        "3906252",
        "--lr",
        "0.00015",
        "--min-lr",
        "0.00001",
        "--decoupled-lr",
        "5.0e-4",
        "--decoupled-min-lr",
        "4.5e-5",
        "--lr-decay-style",
        "cosine",
        "--clip-grad",
        "1.0",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--grad-reduce-in-bf16",
        "--cross-entropy-loss-fusion",
        "--calculate-per-token-loss",
        "--manual-gc",
        "--empty-unused-memory-level",
        "1",
        "--exit-duration-in-mins",
        str(config.duration_mins),
        "--fp8-format",
        "hybrid",
        "--fp8-amax-history-len",
        "1024",
        "--fp8-amax-compute-algo",
        "max",
        "--fp8-param-gather",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--sequence-parallel",
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        "--distributed-backend",
        "nccl",
        "--log-interval",
        "1",
        "--eval-iters",
        "32",
        "--eval-interval",
        "100",
        "--save-interval",
        "1000",
        "--log-throughput",
        "--profile",
        "--profile-step-start",
        "4",
        "--profile-step-end",
        "6",
        "--ckpt-format",
        "torch_dist",
        "--distributed-timeout-minutes",
        "60",
        "--save",
        "/outputs/checkpoints",
        "--tensorboard-dir",
        "/outputs/tensorboard",
    ]
    if config.resume:
        args.extend(["--load", "/outputs/checkpoints"])

    args.extend(build_data_args(config))
    return " ".join(shlex.quote(arg) for arg in args)


def build_data_args(config: Llama3LongConfig) -> list[str]:
    """Build mock-data or tokenizer/data-path arguments for training."""

    if config.tokenizer_model and config.data_path:
        return [
            "--data-path",
            config.data_path,
            "--tokenizer-type",
            "HuggingFaceTokenizer",
            "--tokenizer-model",
            config.tokenizer_model,
            "--data-cache-path",
            "/outputs/cache/data",
            "--split",
            "99,1,0",
            "--no-create-attention-mask-in-dataloader",
            "--no-mmap-bin-files",
            "--num-workers",
            "1",
            "--vocab-size",
            "128256",
        ]

    return [
        "--mock-data",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "128256",
        "--data-cache-path",
        "/outputs/cache/data",
        "--tiktoken-pattern",
        "v2",
        "--split",
        "99,1,0",
        "--no-create-attention-mask-in-dataloader",
        "--no-mmap-bin-files",
        "--num-workers",
        "1",
    ]


def build_preflight_command() -> str:
    """Build environment checks for the Llama-3 8B FP8 run."""

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
        "assert torch.cuda.device_count() >= 8\n"
        "\n"
        "for path in (\n"
        "    '/outputs/home',\n"
        "    '/outputs/cache/torchinductor',\n"
        "    '/outputs/cache/data',\n"
        "    '/outputs/checkpoints',\n"
        "    '/outputs/tensorboard',\n"
        "):\n"
        "    Path(path).mkdir(parents=True, exist_ok=True)\n"
        "    marker = Path(path) / 'preflight-write-check'\n"
        "    marker.write_text('ok', encoding='utf-8')\n"
        "    marker.unlink()\n"
        "\n"
        "major, minor = torch.cuda.get_device_capability(0)\n"
        "assert (major, minor) >= (8, 9), 'FP8 requires Hopper, Ada, or Blackwell GPUs'\n"
        "te_version = getattr(transformer_engine, '__version__', 'unknown')\n"
        "print(f'preflight: torch={torch.__version__} cuda_devices={torch.cuda.device_count()}')\n"
        "print(f'preflight: cuda_capability={major}.{minor}')\n"
        "print(f'preflight: transformer_engine={te_version}')\n"
        "PY\n"
        "cat > /outputs/work/preflight_dist.py <<'PY'\n"
        "import torch\n"
        "import torch.distributed as dist\n"
        "\n"
        "assert torch.cuda.is_available()\n"
        "dist.init_process_group(backend='nccl')\n"
        "rank = dist.get_rank()\n"
        "torch.cuda.set_device(rank % torch.cuda.device_count())\n"
        "value = torch.ones(1, device='cuda')\n"
        "dist.all_reduce(value)\n"
        "assert value.item() == dist.get_world_size()\n"
        "dist.destroy_process_group()\n"
        "print('preflight: torch.distributed nccl ok')\n"
        "PY\n"
        "torchrun --nproc_per_node=8 /outputs/work/preflight_dist.py"
    )


def build_dry_training_validation_command(config: Llama3LongConfig) -> str:
    """Build a lightweight command that records the training command without launching it."""

    training_command = build_training_command(config)
    return (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "\n"
        "assert Path('pretrain_gpt.py').is_file()\n"
        "print('preflight: pretrain_gpt.py found')\n"
        "PY\n"
        f"printf '%s\\n' {shlex.quote('preflight: training command: ' + training_command)}"
    )


def build_inner_shell_command(config: Llama3LongConfig) -> str:
    """Build the shell command executed by Docker."""

    commands = [build_training_command(config)]
    if config.preflight:
        if config.preflight_only:
            commands = [build_dry_training_validation_command(config)]
        commands.insert(0, build_preflight_command())

    return (
        "set -euo pipefail; "
        "mkdir -p "
        "/outputs/work "
        "/outputs/home "
        "/outputs/cache/torchinductor "
        "/outputs/cache/data "
        "/outputs/checkpoints "
        "/outputs/tensorboard; "
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
    config: Llama3LongConfig, user: UserInfo, detach: bool
) -> list[str]:
    """Build the Docker command for the Llama-3 8B FP8 run."""

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
    parser.add_argument("--name", default="mcore-llama3-8b-long")
    parser.add_argument("--output-dir", type=Path, default=Path("local/llama3-8b-long"))
    parser.add_argument("--duration-mins", type=int, default=235)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--data-path")
    return parser.parse_args()


def main() -> None:
    """Run the Llama-3 8B FP8 training Docker command."""

    args = parse_args()
    repo_root = Path.cwd().resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Llama3LongConfig(
        repo_root=repo_root,
        output_dir=output_dir,
        container_name=args.name,
        image=args.image,
        duration_mins=args.duration_mins,
        preflight=args.preflight,
        preflight_only=args.preflight_only,
        resume=args.resume,
        tokenizer_model=args.tokenizer_model,
        data_path=args.data_path,
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
