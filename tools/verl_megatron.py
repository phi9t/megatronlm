#!/usr/bin/env python3
"""Setup and preflight checks for the vendored verl + Megatron-Bridge path."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

VERL_COMMIT = "7dc39fec1c37da4098b50e68840e78906b69f16a"
MEGATRON_BRIDGE_COMMIT = "94dc04baf65157463181eef0c19549d5a6e4ccec"
MEGATRON_CORE_COMMIT = "38986a98aae6a0cc4c8ae7b435db3288a890b0cb"
DEFAULT_IMAGE = "verl-megatron-fsdp:local"
DEFAULT_BASE_IMAGE = "verlai/verl:vllm011.dev7"


@dataclass(frozen=True)
class VerLMegatronConfig:
    repo_root: Path
    image: str = DEFAULT_IMAGE
    base_image: str = DEFAULT_BASE_IMAGE
    min_gpus: int = 8
    dry_run: bool = False

    @property
    def verl_dir(self) -> Path:
        return self.repo_root / "third_party/verl"

    @property
    def bridge_dir(self) -> Path:
        return self.repo_root / "third_party/Megatron-Bridge"

    @property
    def mcore_dir(self) -> Path:
        return self.bridge_dir / "3rdparty/Megatron-LM"

    @property
    def patch_file(self) -> Path:
        return self.repo_root / "tools/verl_megatron/patches/0001-megatron-bridge-fsdp-dtensor-local-shape.patch"

    @property
    def patch_file_from_bridge(self) -> Path:
        return Path("../../tools/verl_megatron/patches/0001-megatron-bridge-fsdp-dtensor-local-shape.patch")


def repo_root() -> Path:
    return Path(__file__).parent.parent


def build_pythonpath() -> str:
    paths = [
        "/workspace/third_party/verl",
        "/workspace/third_party/Megatron-Bridge/src",
        "/workspace/third_party/Megatron-Bridge/3rdparty/Megatron-LM",
    ]
    return ":".join(paths)


def build_image_command(config: VerLMegatronConfig) -> list[str]:
    return [
        "docker",
        "build",
        "--build-arg",
        f"BASE_IMAGE={config.base_image}",
        "-t",
        config.image,
        "-f",
        "tools/verl_megatron/Dockerfile",
        ".",
    ]


def build_docker_run_command(
    config: VerLMegatronConfig,
    inner_command: str,
    *,
    workdir: str = "/workspace/third_party/verl",
) -> list[str]:
    env = {
        "PYTHONPATH": build_pythonpath(),
        "HF_HOME": "/workspace/local/hf-home",
        "HYDRA_FULL_ERROR": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        "TORCH_CUDA_ARCH_LIST": "10.0",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
        "UV_CACHE_DIR": "/workspace/local/uv-cache",
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--mount",
        "type=bind,source=.,target=/workspace",
        "-w",
        workdir,
        "--entrypoint",
        "bash",
    ]
    for key, value in env.items():
        command.extend(["-e", f"{key}={value}"])
    command.extend([config.image, "-lc", inner_command])
    return command


def build_import_preflight_command(min_gpus: int) -> str:
    return textwrap.dedent(
        f"""
        python3 - <<'PY'
        import importlib
        import torch

        modules = [
            "verl",
            "verl.trainer.sft_trainer",
            "verl.workers.engine.megatron.transformer_impl",
            "megatron.bridge.models.conversion.param_mapping",
            "megatron.core",
        ]
        for module in modules:
            importlib.import_module(module)

        gpu_count = torch.cuda.device_count()
        print(f"imports ok; cuda_device_count={{gpu_count}}")
        if gpu_count < {min_gpus}:
            raise SystemExit(f"expected at least {min_gpus} CUDA devices, found {{gpu_count}}")
        PY
        """
    ).strip()


def build_data_prep_command() -> str:
    return textwrap.dedent(
        """
        python3 examples/data_preprocess/gsm8k_multiturn_sft.py --local_save_dir /workspace/local/verl-data/gsm8k_sft
        """
    ).strip()


def build_model_download_command() -> str:
    return "hf download Qwen/Qwen2.5-Math-7B"


def build_sft_smoke_command() -> str:
    args = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
        "-m",
        "verl.trainer.sft_trainer",
        "data.train_files=/workspace/local/verl-data/gsm8k_sft/train.parquet",
        "data.val_files=/workspace/local/verl-data/gsm8k_sft/test.parquet",
        "data.messages_key=messages",
        "data.train_batch_size=8",
        "data.use_dynamic_bsz=True",
        "data.max_token_len_per_gpu=1024",
        "data.pad_mode=no_padding",
        "data.truncation=error",
        "model=hf_model",
        "model.path=Qwen/Qwen2.5-Math-7B",
        "model.trust_remote_code=True",
        "model.use_remove_padding=True",
        "engine=megatron",
        "engine.tensor_model_parallel_size=4",
        "engine.pipeline_model_parallel_size=1",
        "engine.expert_model_parallel_size=1",
        "engine.use_mbridge=True",
        "engine.vanilla_mbridge=False",
        "engine.use_megatron_fsdp=True",
        "+engine.override_transformer_config.gradient_accumulation_fusion=False",
        "optim=megatron",
        "optim.lr=1e-5",
        "optim.lr_warmup_steps_ratio=0.2",
        "optim.weight_decay=0.1",
        "optim.betas=[0.9,0.95]",
        "optim.clip_grad=1.0",
        "optim.lr_warmup_init=0",
        "optim.lr_decay_style=cosine",
        "optim.min_lr=1e-6",
        "trainer.default_local_dir=/workspace/local/verl-runs/sft-qwen-mfsdp-smoke-model-only",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=1",
        "trainer.project_name=verl_megatron_smoke",
        "trainer.experiment_name=sft_qwen_mfsdp_smoke",
        "trainer.logger=['console']",
        'checkpoint.save_contents=["model"]',
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def build_production_data_prep_command() -> str:
    return " && ".join(
        [
            "python3 examples/data_preprocess/gsm8k_multiturn_sft.py --local_save_dir /workspace/local/verl-data/gsm8k_sft",
            "python3 examples/data_preprocess/gsm8k.py --local_save_dir /workspace/local/verl-data/gsm8k",
            "python3 examples/data_preprocess/math_dataset.py --local_save_dir /workspace/local/verl-data/math",
        ]
    )


def build_production_sft_command(*, train_path: str, val_path: str, output_dir: str, total_steps: int) -> str:
    args = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
        "-m",
        "verl.trainer.sft_trainer",
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        "data.messages_key=messages",
        "data.train_batch_size=8",
        "data.use_dynamic_bsz=True",
        "data.max_token_len_per_gpu=1024",
        "data.pad_mode=no_padding",
        "data.truncation=error",
        "model=hf_model",
        "model.path=Qwen/Qwen2.5-Math-7B",
        "model.trust_remote_code=True",
        "model.use_remove_padding=True",
        "engine=megatron",
        "engine.tensor_model_parallel_size=4",
        "engine.pipeline_model_parallel_size=1",
        "engine.expert_model_parallel_size=1",
        "engine.use_mbridge=True",
        "engine.vanilla_mbridge=False",
        "engine.use_megatron_fsdp=True",
        "+engine.override_transformer_config.gradient_accumulation_fusion=False",
        "optim=megatron",
        "optim.lr=1e-5",
        "optim.lr_warmup_steps_ratio=0.2",
        "optim.weight_decay=0.1",
        "optim.betas=[0.9,0.95]",
        "optim.clip_grad=1.0",
        "optim.lr_warmup_init=0",
        "optim.lr_decay_style=cosine",
        "optim.min_lr=1e-6",
        f"trainer.default_local_dir={output_dir}",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={total_steps}",
        "trainer.project_name=verl_megatron_production_validation",
        "trainer.experiment_name=sft_b200_tp4",
        "trainer.logger=['console']",
        'checkpoint.save_contents=["model"]',
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def build_production_rl_command(
    *,
    gsm8k_train_path: str,
    gsm8k_test_path: str,
    math_train_path: str,
    math_test_path: str,
    output_dir: str,
    total_steps: int,
) -> str:
    train_files = f"['{gsm8k_train_path}','{math_train_path}']"
    test_files = f"['{gsm8k_test_path}','{math_test_path}']"
    args = [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
        "--config-path=config",
        "--config-name=ppo_megatron_trainer.yaml",
        f"data.train_files={train_files}",
        f"data.val_files={test_files}",
        "data.return_raw_chat=True",
        "data.train_batch_size=32",
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-7B",
        "actor_rollout_ref.model.use_fused_kernels=False",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1",
        "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4",
        "actor_rollout_ref.actor.megatron.use_mbridge=True",
        "actor_rollout_ref.actor.megatron.vanilla_mbridge=False",
        "actor_rollout_ref.actor.megatron.use_megatron_fsdp=True",
        "++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1",
        "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4",
        "actor_rollout_ref.ref.megatron.use_mbridge=True",
        "actor_rollout_ref.ref.megatron.vanilla_mbridge=False",
        "actor_rollout_ref.ref.megatron.use_megatron_fsdp=True",
        "++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=False",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console']",
        "trainer.project_name=verl_megatron_production_validation",
        "trainer.experiment_name=grpo_b200_tp4",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.save_freq=1",
        "trainer.test_freq=1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={total_steps}",
        f"trainer.default_local_dir={output_dir}",
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, cwd: Path, dry_run: bool = False) -> None:
    print(format_command(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def check_output(command: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def ensure_submodules(config: VerLMegatronConfig) -> None:
    run_command(
        ["git", "submodule", "update", "--init", "--recursive", "third_party/verl", "third_party/Megatron-Bridge"],
        cwd=config.repo_root,
        dry_run=config.dry_run,
    )


def verify_commit(path: Path, expected: str, label: str, *, dry_run: bool = False) -> None:
    if dry_run:
        print(f"check {label} commit == {expected}")
        return
    actual = check_output(["git", "rev-parse", "HEAD"], cwd=path)
    if actual != expected:
        raise SystemExit(f"{label} is at {actual}, expected {expected}")
    print(f"{label} commit ok: {actual}")


def verify_pins(config: VerLMegatronConfig) -> None:
    verify_commit(config.verl_dir, VERL_COMMIT, "verl", dry_run=config.dry_run)
    verify_commit(config.bridge_dir, MEGATRON_BRIDGE_COMMIT, "Megatron-Bridge", dry_run=config.dry_run)
    verify_commit(config.mcore_dir, MEGATRON_CORE_COMMIT, "Megatron-Core nested submodule", dry_run=config.dry_run)


def patch_state(config: VerLMegatronConfig) -> str:
    check = subprocess.run(
        ["git", "apply", "--check", str(config.patch_file_from_bridge)],
        cwd=config.bridge_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode == 0:
        return "missing"
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(config.patch_file_from_bridge)],
        cwd=config.bridge_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if reverse.returncode == 0:
        return "applied"
    return "conflict"


def apply_bridge_patch(config: VerLMegatronConfig) -> None:
    if config.dry_run:
        print(f"apply patch {config.patch_file_from_bridge} in third_party/Megatron-Bridge")
        return
    state = patch_state(config)
    if state == "applied":
        print("Megatron-Bridge compatibility patch already applied")
        return
    if state != "missing":
        raise SystemExit("Megatron-Bridge compatibility patch is neither cleanly applicable nor already applied")
    run_command(["git", "apply", str(config.patch_file_from_bridge)], cwd=config.bridge_dir)


def ensure_image(config: VerLMegatronConfig) -> None:
    if config.dry_run:
        run_command(build_image_command(config), cwd=config.repo_root, dry_run=True)
        return
    inspect = subprocess.run(["docker", "image", "inspect", config.image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if inspect.returncode == 0:
        print(f"Docker image exists: {config.image}")
        return
    run_command(build_image_command(config), cwd=config.repo_root)


def require_image(config: VerLMegatronConfig) -> None:
    if config.dry_run:
        print(f"check docker image exists: {config.image}")
        return
    inspect = subprocess.run(["docker", "image", "inspect", config.image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if inspect.returncode != 0:
        raise SystemExit(f"Docker image {config.image} not found; run setup or pass --image")
    print(f"Docker image ok: {config.image}")


def run_setup(args: argparse.Namespace) -> None:
    config = VerLMegatronConfig(
        repo_root=repo_root(),
        image=args.image,
        base_image=args.base_image,
        dry_run=args.dry_run,
    )
    ensure_submodules(config)
    verify_pins(config)
    apply_bridge_patch(config)
    if not args.skip_image:
        ensure_image(config)
    if not args.skip_data:
        run_command(build_docker_run_command(config, build_data_prep_command()), cwd=config.repo_root, dry_run=config.dry_run)
    if args.download_model:
        run_command(build_docker_run_command(config, build_model_download_command()), cwd=config.repo_root, dry_run=config.dry_run)


def run_preflight(args: argparse.Namespace) -> None:
    config = VerLMegatronConfig(
        repo_root=repo_root(),
        image=args.image,
        min_gpus=args.min_gpus,
        dry_run=args.dry_run,
    )
    verify_pins(config)
    if not config.dry_run:
        state = patch_state(config)
        if state != "applied":
            raise SystemExit(f"Megatron-Bridge patch is {state}; run setup before preflight")
        print("Megatron-Bridge compatibility patch ok")
    else:
        print("check Megatron-Bridge compatibility patch is applied")
    require_image(config)
    command = build_sft_smoke_command() if args.smoke else build_import_preflight_command(args.min_gpus)
    run_command(build_docker_run_command(config, command), cwd=config.repo_root, dry_run=config.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Initialize vendored repos, apply the Bridge patch, and prepare runtime assets.")
    setup.add_argument("--image", default=DEFAULT_IMAGE)
    setup.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    setup.add_argument("--skip-image", action="store_true", help="Do not build or check the Docker image.")
    setup.add_argument("--skip-data", action="store_true", help="Do not prepare the GSM8K SFT parquet data.")
    setup.add_argument("--download-model", action="store_true", help="Pre-download Qwen/Qwen2.5-Math-7B into local/hf-home.")
    setup.add_argument("--dry-run", action="store_true")
    setup.set_defaults(func=run_setup)

    preflight = subparsers.add_parser("preflight", help="Run a quick import/GPU check or the one-step SFT smoke.")
    preflight.add_argument("--image", default=DEFAULT_IMAGE)
    preflight.add_argument("--min-gpus", type=int, default=8)
    preflight.add_argument("--smoke", action="store_true", help="Run the one-step 8-GPU Megatron-FSDP SFT smoke.")
    preflight.add_argument("--dry-run", action="store_true")
    preflight.set_defaults(func=run_preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
