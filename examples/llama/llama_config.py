# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pydantic models and command builders for Llama example runs."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Literal, Sequence

import pydantic
import yaml


class LlamaConfigError(ValueError):
    """Raised when a Llama YAML config cannot be loaded."""


class StrictModel(pydantic.BaseModel):
    """Base model that rejects unknown YAML fields."""

    model_config = pydantic.ConfigDict(extra="forbid")


def reject_tmp_path(value: str) -> str:
    """Reject runtime paths under /tmp."""

    if value == "/tmp" or value.startswith("/tmp/"):
        raise ValueError("use repo-local paths instead of /tmp")
    return value


def add_arg(args: list[str], name: str, value: object | None) -> None:
    """Append a Megatron argument when the value is present."""

    if value is not None:
        args.extend([name, str(value)])


def add_bool_flag(args: list[str], name: str, enabled: bool) -> None:
    """Append a Megatron boolean flag when enabled."""

    if enabled:
        args.append(name)


class RuntimeConfig(StrictModel):
    """Host/container launch settings."""

    image: str = "megatron-lm:smoke"
    container_name: str
    output_dir: str
    foreground: bool = False
    dry_run: bool = False
    resume: bool = False

    @pydantic.field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: str) -> str:
        """Validate output directory placement."""

        return reject_tmp_path(value)


class DockerConfig(StrictModel):
    """Docker runtime shape shared by local Llama examples."""

    work_dir: str = "/outputs/work"
    home_dir: str = "/outputs/home"
    torchinductor_cache_dir: str = "/outputs/cache/torchinductor"
    data_cache_path: str = "/outputs/cache/data"
    checkpoints_path: str | None = None
    tensorboard_dir: str = "/outputs/tensorboard"
    source_mount: str = "/workspace"
    outputs_mount: str = "/outputs"
    cuda_device_max_connections: str = "1"
    source_excludes: list[str] = pydantic.Field(
        default_factory=lambda: [
            ".git",
            ".venv",
            ".agents",
            ".cache",
            ".pytest_cache",
            ".ruff_cache",
            "local",
            "__pycache__",
            "*.pyc",
            "*.so",
        ]
    )

    @pydantic.field_validator(
        "work_dir",
        "home_dir",
        "torchinductor_cache_dir",
        "data_cache_path",
        "checkpoints_path",
        "tensorboard_dir",
        "source_mount",
        "outputs_mount",
    )
    @classmethod
    def validate_paths(cls, value: str | None) -> str | None:
        """Validate container paths."""

        return reject_tmp_path(value) if value is not None else value

    @property
    def writable_paths(self) -> list[str]:
        """Return output-backed paths that must be created before training."""

        paths = [
            self.work_dir,
            self.home_dir,
            self.torchinductor_cache_dir,
            self.data_cache_path,
            self.tensorboard_dir,
        ]
        if self.checkpoints_path is not None:
            paths.append(self.checkpoints_path)
        return paths


class DistributedConfig(StrictModel):
    """Torch distributed and environment validation settings."""

    nproc_per_node: int = pydantic.Field(gt=0)
    backend: str = "nccl"
    timeout_minutes: int | None = pydantic.Field(default=None, gt=0)
    min_cuda_devices: int = pydantic.Field(default=1, gt=0)
    min_compute_capability: tuple[int, int] | None = None


class ModelConfig(StrictModel):
    """Transformer architecture settings."""

    use_mcore_models: bool = True
    num_layers: int = pydantic.Field(gt=0)
    hidden_size: int = pydantic.Field(gt=0)
    ffn_hidden_size: int = pydantic.Field(gt=0)
    num_attention_heads: int = pydantic.Field(gt=0)
    group_query_attention: bool = True
    num_query_groups: int | None = pydantic.Field(default=None, gt=0)
    kv_channels: int | None = pydantic.Field(default=None, gt=0)
    seq_length: int = pydantic.Field(gt=0)
    max_position_embeddings: int = pydantic.Field(gt=0)
    position_embedding_type: str = "rope"
    rotary_base: int | None = pydantic.Field(default=1_000_000, gt=0)
    rotary_percent: float | None = pydantic.Field(default=1.0, ge=0.0)
    attention_dropout: float = pydantic.Field(default=0.0, ge=0.0)
    hidden_dropout: float = pydantic.Field(default=0.0, ge=0.0)
    swiglu: bool = True
    normalization: str = "RMSNorm"
    init_method_std: float | None = pydantic.Field(default=0.0134, gt=0.0)
    attention_backend: str | None = "fused"
    apply_layernorm_1p: bool = True
    untie_embeddings_and_output_weights: bool = True
    disable_bias_linear: bool = True

    def to_args(self) -> list[str]:
        """Render model settings into Megatron arguments."""

        args: list[str] = []
        add_bool_flag(args, "--use-mcore-models", self.use_mcore_models)
        add_arg(args, "--num-layers", self.num_layers)
        add_arg(args, "--hidden-size", self.hidden_size)
        add_arg(args, "--ffn-hidden-size", self.ffn_hidden_size)
        add_arg(args, "--num-attention-heads", self.num_attention_heads)
        add_bool_flag(args, "--group-query-attention", self.group_query_attention)
        add_arg(args, "--num-query-groups", self.num_query_groups)
        add_arg(args, "--kv-channels", self.kv_channels)
        add_arg(args, "--seq-length", self.seq_length)
        add_arg(args, "--max-position-embeddings", self.max_position_embeddings)
        add_arg(args, "--position-embedding-type", self.position_embedding_type)
        add_arg(args, "--rotary-base", self.rotary_base)
        add_arg(args, "--rotary-percent", self.rotary_percent)
        add_arg(args, "--attention-dropout", self.attention_dropout)
        add_arg(args, "--hidden-dropout", self.hidden_dropout)
        add_bool_flag(args, "--swiglu", self.swiglu)
        add_arg(args, "--normalization", self.normalization)
        add_arg(args, "--init-method-std", self.init_method_std)
        add_arg(args, "--attention-backend", self.attention_backend)
        add_bool_flag(args, "--apply-layernorm-1p", self.apply_layernorm_1p)
        add_bool_flag(
            args,
            "--untie-embeddings-and-output-weights",
            self.untie_embeddings_and_output_weights,
        )
        add_bool_flag(args, "--disable-bias-linear", self.disable_bias_linear)
        return args


class TrainingConfig(StrictModel):
    """Batching, schedule, precision, and runtime training settings."""

    micro_batch_size: int = pydantic.Field(gt=0)
    global_batch_size: int = pydantic.Field(gt=0)
    train_iters: int | None = pydantic.Field(default=None, gt=0)
    train_samples: int | None = pydantic.Field(default=None, gt=0)
    lr_decay_iters: int | None = pydantic.Field(default=None, gt=0)
    lr_decay_samples: int | None = pydantic.Field(default=None, gt=0)
    lr_warmup_samples: int | None = pydantic.Field(default=None, ge=0)
    exit_duration_in_mins: int | None = pydantic.Field(default=None, gt=0)
    bf16: bool = True
    transformer_impl: str | None = "transformer_engine"
    grad_reduce_in_bf16: bool = False
    cross_entropy_loss_fusion: bool = False
    calculate_per_token_loss: bool = False
    manual_gc: bool = False
    empty_unused_memory_level: int | None = pydantic.Field(default=None, ge=0)

    @pydantic.model_validator(mode="after")
    def validate_training_horizon(self) -> TrainingConfig:
        """Require one primary training horizon."""

        if self.train_iters is None and self.train_samples is None:
            raise ValueError(
                "set either training.train_iters or training.train_samples"
            )
        return self

    def to_args(self) -> list[str]:
        """Render training settings into Megatron arguments."""

        args: list[str] = []
        add_arg(args, "--micro-batch-size", self.micro_batch_size)
        add_arg(args, "--global-batch-size", self.global_batch_size)
        add_arg(args, "--train-iters", self.train_iters)
        add_arg(args, "--train-samples", self.train_samples)
        add_arg(args, "--lr-decay-iters", self.lr_decay_iters)
        add_arg(args, "--lr-decay-samples", self.lr_decay_samples)
        add_arg(args, "--lr-warmup-samples", self.lr_warmup_samples)
        add_arg(args, "--exit-duration-in-mins", self.exit_duration_in_mins)
        add_bool_flag(args, "--bf16", self.bf16)
        add_arg(args, "--transformer-impl", self.transformer_impl)
        add_bool_flag(args, "--grad-reduce-in-bf16", self.grad_reduce_in_bf16)
        add_bool_flag(
            args, "--cross-entropy-loss-fusion", self.cross_entropy_loss_fusion
        )
        add_bool_flag(args, "--calculate-per-token-loss", self.calculate_per_token_loss)
        add_bool_flag(args, "--manual-gc", self.manual_gc)
        add_arg(args, "--empty-unused-memory-level", self.empty_unused_memory_level)
        return args


class OptimizerConfig(StrictModel):
    """Optimizer and learning-rate settings."""

    lr: str
    min_lr: str
    lr_decay_style: str = "cosine"
    weight_decay: str = "0.1"
    clip_grad: str = "1.0"
    adam_beta1: str | None = None
    adam_beta2: str | None = None
    decoupled_lr: str | None = None
    decoupled_min_lr: str | None = None

    def to_args(self) -> list[str]:
        """Render optimizer settings into Megatron arguments."""

        args: list[str] = []
        add_arg(args, "--lr", self.lr)
        add_arg(args, "--min-lr", self.min_lr)
        add_arg(args, "--decoupled-lr", self.decoupled_lr)
        add_arg(args, "--decoupled-min-lr", self.decoupled_min_lr)
        add_arg(args, "--lr-decay-style", self.lr_decay_style)
        add_arg(args, "--weight-decay", self.weight_decay)
        add_arg(args, "--clip-grad", self.clip_grad)
        add_arg(args, "--adam-beta1", self.adam_beta1)
        add_arg(args, "--adam-beta2", self.adam_beta2)
        return args


class Fp8Config(StrictModel):
    """FP8 settings."""

    enabled: bool = False
    format: str = "hybrid"
    amax_history_len: int = pydantic.Field(default=1024, gt=0)
    amax_compute_algo: str = "max"
    param_gather: bool = True

    def to_args(self) -> list[str]:
        """Render FP8 settings into Megatron arguments."""

        if not self.enabled:
            return []
        args: list[str] = []
        add_arg(args, "--fp8-format", self.format)
        add_arg(args, "--fp8-amax-history-len", self.amax_history_len)
        add_arg(args, "--fp8-amax-compute-algo", self.amax_compute_algo)
        add_bool_flag(args, "--fp8-param-gather", self.param_gather)
        return args


class ParallelismConfig(StrictModel):
    """Model-parallel and distributed optimizer settings."""

    tensor_model_parallel_size: int = pydantic.Field(default=1, gt=0)
    pipeline_model_parallel_size: int = pydantic.Field(default=1, gt=0)
    context_parallel_size: int | None = pydantic.Field(default=None, gt=0)
    sequence_parallel: bool = False
    use_distributed_optimizer: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False

    def to_args(self) -> list[str]:
        """Render parallelism settings into Megatron arguments."""

        args: list[str] = []
        add_arg(args, "--tensor-model-parallel-size", self.tensor_model_parallel_size)
        add_arg(
            args, "--pipeline-model-parallel-size", self.pipeline_model_parallel_size
        )
        add_arg(args, "--context-parallel-size", self.context_parallel_size)
        add_bool_flag(args, "--sequence-parallel", self.sequence_parallel)
        add_bool_flag(
            args, "--use-distributed-optimizer", self.use_distributed_optimizer
        )
        add_bool_flag(args, "--overlap-grad-reduce", self.overlap_grad_reduce)
        add_bool_flag(args, "--overlap-param-gather", self.overlap_param_gather)
        return args


class DataConfig(StrictModel):
    """Mock-data or real-data settings."""

    mode: Literal["mock", "real"] = "mock"
    tokenizer_type: str | None = None
    tokenizer_model: str | None = None
    data_path: str | None = None
    vocab_size: int = pydantic.Field(gt=0)
    split: str
    tiktoken_pattern: str | None = None
    no_create_attention_mask_in_dataloader: bool = True
    no_mmap_bin_files: bool = True
    num_workers: int = pydantic.Field(default=1, ge=0)

    @pydantic.model_validator(mode="after")
    def validate_real_data(self) -> DataConfig:
        """Require tokenizer and data paths together in real-data mode."""

        if self.mode == "real" and (not self.tokenizer_model or not self.data_path):
            raise ValueError(
                "real data mode requires data.tokenizer_model and data.data_path"
            )
        if self.mode == "mock" and (self.tokenizer_model or self.data_path):
            raise ValueError("mock data mode must not set tokenizer_model or data_path")
        return self

    def to_args(self, docker: DockerConfig) -> list[str]:
        """Render data settings into Megatron arguments."""

        args: list[str] = []
        if self.mode == "mock":
            args.append("--mock-data")
            add_arg(args, "--tokenizer-type", self.tokenizer_type or "NullTokenizer")
        else:
            add_arg(args, "--data-path", self.data_path)
            add_arg(
                args, "--tokenizer-type", self.tokenizer_type or "HuggingFaceTokenizer"
            )
            add_arg(args, "--tokenizer-model", self.tokenizer_model)
        add_arg(args, "--vocab-size", self.vocab_size)
        add_arg(args, "--data-cache-path", docker.data_cache_path)
        add_arg(args, "--tiktoken-pattern", self.tiktoken_pattern)
        add_arg(args, "--split", self.split)
        add_bool_flag(
            args,
            "--no-create-attention-mask-in-dataloader",
            self.no_create_attention_mask_in_dataloader,
        )
        add_bool_flag(args, "--no-mmap-bin-files", self.no_mmap_bin_files)
        add_arg(args, "--num-workers", self.num_workers)
        return args


class LoggingConfig(StrictModel):
    """Evaluation, logging, profiling, and checkpoint settings."""

    log_interval: int = pydantic.Field(gt=0)
    eval_iters: int | None = pydantic.Field(default=None, ge=0)
    eval_interval: int | None = pydantic.Field(default=None, gt=0)
    save_interval: int | None = pydantic.Field(default=None, gt=0)
    log_throughput: bool = False
    profile: bool = False
    profile_step_start: int | None = pydantic.Field(default=None, ge=0)
    profile_step_end: int | None = pydantic.Field(default=None, ge=0)
    ckpt_format: str | None = None

    def to_args(self, config: LlamaRunConfig) -> list[str]:
        """Render logging settings into Megatron arguments."""

        args: list[str] = []
        add_arg(args, "--log-interval", self.log_interval)
        add_arg(args, "--eval-iters", self.eval_iters)
        add_arg(args, "--eval-interval", self.eval_interval)
        add_arg(args, "--save-interval", self.save_interval)
        add_bool_flag(args, "--log-throughput", self.log_throughput)
        add_bool_flag(args, "--profile", self.profile)
        add_arg(args, "--profile-step-start", self.profile_step_start)
        add_arg(args, "--profile-step-end", self.profile_step_end)
        add_arg(args, "--ckpt-format", self.ckpt_format)
        add_arg(
            args, "--distributed-timeout-minutes", config.distributed.timeout_minutes
        )
        if config.docker.checkpoints_path is not None:
            add_arg(args, "--save", config.docker.checkpoints_path)
            if config.runtime.resume:
                add_arg(args, "--load", config.docker.checkpoints_path)
        add_arg(args, "--tensorboard-dir", config.docker.tensorboard_dir)
        return args


class PreflightConfig(StrictModel):
    """Environment probes that can run before training."""

    enabled: bool = False
    only: bool = False
    require_transformer_engine: bool = True
    require_fp8: bool = False


class LlamaRunConfig(StrictModel):
    """Complete YAML-backed Llama run configuration."""

    runtime: RuntimeConfig
    docker: DockerConfig = pydantic.Field(default_factory=DockerConfig)
    distributed: DistributedConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    fp8: Fp8Config = pydantic.Field(default_factory=Fp8Config)
    parallelism: ParallelismConfig = pydantic.Field(default_factory=ParallelismConfig)
    data: DataConfig
    logging: LoggingConfig
    preflight: PreflightConfig = pydantic.Field(default_factory=PreflightConfig)

    @pydantic.model_validator(mode="after")
    def validate_fp8_requirements(self) -> LlamaRunConfig:
        """Require a compute-capability guard when FP8 is enabled."""

        if self.fp8.enabled and self.distributed.min_compute_capability is None:
            raise ValueError("FP8 configs must set distributed.min_compute_capability")
        return self

    def training_args(self) -> list[str]:
        """Render the full pretrain_gpt.py argument list."""

        args = [
            "torchrun",
            f"--nproc_per_node={self.distributed.nproc_per_node}",
            "pretrain_gpt.py",
        ]
        args.extend(self.model.to_args())
        args.extend(self.training.to_args())
        args.extend(self.optimizer.to_args())
        args.extend(self.fp8.to_args())
        args.extend(self.parallelism.to_args())
        add_arg(args, "--distributed-backend", self.distributed.backend)
        args.extend(self.logging.to_args(self))
        args.extend(self.data.to_args(self.docker))
        return args

    def training_command(self) -> str:
        """Render the training command for a POSIX shell."""

        return " ".join(shlex.quote(arg) for arg in self.training_args())


class UserInfo(StrictModel):
    """Host user identity passed through to Docker."""

    uid: int
    gid: int
    name: str


def load_config(path: Path) -> LlamaRunConfig:
    """Load and validate a YAML-backed Llama run config."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LlamaConfigError(f"failed to read {path}: {exc}") from exc
    if raw is None:
        raw = {}
    try:
        return LlamaRunConfig.model_validate(raw)
    except pydantic.ValidationError as exc:
        raise LlamaConfigError(f"invalid Llama config {path}:\n{exc}") from exc


def resolve_config_path(cli_path: str | None, default_path: Path | None = None) -> Path:
    """Resolve the config path from CLI, environment, or wrapper default."""

    selected = cli_path or os.environ.get("LLAMA_CONFIG")
    if selected is None:
        if default_path is None:
            raise LlamaConfigError("set --config or LLAMA_CONFIG")
        selected = str(default_path)
    return Path(selected).expanduser().resolve()


def with_operational_overrides(
    config: LlamaRunConfig,
    *,
    foreground: bool = False,
    dry_run: bool = False,
    preflight_only: bool = False,
) -> LlamaRunConfig:
    """Apply the only CLI-supported overrides."""

    runtime_updates: dict[str, Any] = {}
    preflight_updates: dict[str, Any] = {}
    if foreground:
        runtime_updates["foreground"] = True
    if dry_run:
        runtime_updates["dry_run"] = True
    if preflight_only:
        preflight_updates["only"] = True
    if not runtime_updates and not preflight_updates:
        return config
    return config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update=runtime_updates),
            "preflight": config.preflight.model_copy(update=preflight_updates),
        }
    )


def build_preflight_command(config: LlamaRunConfig) -> str:
    """Build environment checks for a Llama run."""

    write_checks = "\n".join(
        [
            "for path in (",
            *[f"    {path!r}," for path in config.docker.writable_paths],
            "):",
            "    Path(path).mkdir(parents=True, exist_ok=True)",
            "    marker = Path(path) / 'preflight-write-check'",
            "    marker.write_text('ok', encoding='utf-8')",
            "    marker.unlink()",
        ]
    )
    transformer_engine_lines = (
        "import transformer_engine\n"
        "te_version = getattr(transformer_engine, '__version__', 'unknown')\n"
        "print(f'preflight: transformer_engine={te_version}')\n"
        if config.preflight.require_transformer_engine
        else ""
    )
    capability_lines = ""
    if config.distributed.min_compute_capability is not None:
        capability = config.distributed.min_compute_capability
        message = "FP8 requires Hopper, Ada, or Blackwell GPUs"
        if config.preflight.require_fp8:
            capability_lines = (
                "major, minor = torch.cuda.get_device_capability(0)\n"
                f"assert (major, minor) >= {capability!r}, {message!r}\n"
                "print(f'preflight: cuda_capability={major}.{minor}')\n"
            )
    return (
        "python - <<'PY'\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "import torch\n"
        f"{transformer_engine_lines}"
        f"assert os.environ['HOME'] == {config.docker.home_dir!r}\n"
        "assert os.environ['USER']\n"
        "assert os.environ['LOGNAME']\n"
        f"assert os.environ['TORCHINDUCTOR_CACHE_DIR'] == {config.docker.torchinductor_cache_dir!r}\n"
        f"assert os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] == {config.docker.cuda_device_max_connections!r}\n"
        "assert torch.cuda.is_available()\n"
        f"assert torch.cuda.device_count() >= {config.distributed.min_cuda_devices}\n"
        f"{write_checks}\n"
        f"{capability_lines}"
        "print(f'preflight: torch={torch.__version__} cuda_devices={torch.cuda.device_count()}')\n"
        "PY\n"
        f"cat > {config.docker.work_dir}/preflight_dist.py <<'PY'\n"
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
        f"torchrun --nproc_per_node={config.distributed.nproc_per_node} "
        f"{config.docker.work_dir}/preflight_dist.py"
    )


def build_dry_training_validation_command(config: LlamaRunConfig) -> str:
    """Build a command that validates the copied source and records training args."""

    training_command = config.training_command()
    return (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "\n"
        "assert Path('pretrain_gpt.py').is_file()\n"
        "print('preflight: pretrain_gpt.py found')\n"
        "PY\n"
        f"printf '%s\\n' {shlex.quote('preflight: training command: ' + training_command)}"
    )


def build_inner_shell_command(config: LlamaRunConfig) -> str:
    """Build the shell command executed by Docker."""

    commands = [config.training_command()]
    if config.preflight.enabled:
        if config.preflight.only:
            commands = [build_dry_training_validation_command(config)]
        commands.insert(0, build_preflight_command(config))
    mkdir_paths = " ".join(shlex.quote(path) for path in config.docker.writable_paths)
    tar_excludes = " ".join(
        f"--exclude={shlex.quote(exclude)}" for exclude in config.docker.source_excludes
    )
    return (
        "set -euo pipefail; "
        f"mkdir -p {mkdir_paths}; "
        f"run_dir=$(mktemp -d {config.docker.work_dir}/megatron-lm.XXXXXXXX); "
        "trap 'rm -rf \"$run_dir\"' EXIT; "
        f"tar {tar_excludes} "
        f"-C {shlex.quote(config.docker.source_mount)} -cf - . | "
        'tar -C "$run_dir" -xf -; '
        'cd "$run_dir"; ' + "; ".join(commands)
    )


def build_docker_command(
    config: LlamaRunConfig, repo_root: Path, user: UserInfo
) -> list[str]:
    """Build the Docker command for a validated Llama run."""

    output_dir = resolve_output_dir(config.runtime.output_dir, repo_root)
    shell_command = build_inner_shell_command(config)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        config.runtime.container_name,
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
        f"{repo_root}:{config.docker.source_mount}:ro",
        "-v",
        f"{output_dir}:{config.docker.outputs_mount}",
        "-w",
        config.docker.outputs_mount,
        "-e",
        f"HOME={config.docker.home_dir}",
        "-e",
        f"USER={user.name}",
        "-e",
        f"LOGNAME={user.name}",
        "-e",
        "PIP_CONSTRAINT=",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        f"TORCHINDUCTOR_CACHE_DIR={config.docker.torchinductor_cache_dir}",
        "-e",
        f"CUDA_DEVICE_MAX_CONNECTIONS={config.docker.cuda_device_max_connections}",
        config.runtime.image,
        "bash",
        "-lc",
        shell_command,
    ]
    if not config.runtime.foreground:
        command.insert(2, "-d")
    return command


def build_remove_container_command(container_name: str) -> list[str]:
    """Build a command that removes a stale Docker container by name."""

    return ["docker", "rm", "-f", container_name]


def resolve_output_dir(output_dir: str, repo_root: Path) -> Path:
    """Resolve a YAML output directory relative to the repo root."""

    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def quoted_command(command: Sequence[str]) -> str:
    """Render a command list for display or dry-run output."""

    return shlex.join(command)
