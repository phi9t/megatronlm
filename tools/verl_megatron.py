#!/usr/bin/env python3
"""Setup and preflight checks for the vendored verl + Megatron-Bridge path."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

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


def load_evidence_helpers() -> ModuleType:
    module_path = Path(__file__).parent / "verl_megatron_evidence.py"
    spec = importlib.util.spec_from_file_location("verl_megatron_evidence", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evidence_helpers = load_evidence_helpers()


def default_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def workspace_run_dir(run_id: str) -> str:
    return f"/workspace/local/verl-runs/production-validation/{run_id}"


def host_run_dir(config: VerLMegatronConfig, run_id: str) -> Path:
    return config.repo_root / "local/verl-runs/production-validation" / run_id


def system_exit_code(exc: SystemExit) -> int:
    if isinstance(exc.code, int):
        return exc.code
    return 1


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
        from transformers import AutoTokenizer

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

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B", trust_remote_code=True)
        if not hasattr(tokenizer, "all_special_tokens_extended"):
            raise SystemExit(
                "Qwen tokenizer is incompatible with vLLM: missing all_special_tokens_extended"
            )
        print("tokenizer ok; all_special_tokens_extended available")
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


def build_logged_inner_command(inner_command: str, *, log_path: str, exit_code_path: str) -> str:
    grouped_command = textwrap.indent(inner_command, "  ")
    return textwrap.dedent(
        f"""
        mkdir -p {shlex.quote(str(Path(log_path).parent))}
        set +e
        {{
        {grouped_command}
        }} 2>&1 | tee {shlex.quote(log_path)}
        echo ${{PIPESTATUS[0]}} > {shlex.quote(exit_code_path)}
        exit $(cat {shlex.quote(exit_code_path)})
        """
    ).strip()


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


def run_validate_production(args: argparse.Namespace) -> None:
    run_id = args.run_id or default_run_id()
    config = VerLMegatronConfig(repo_root=repo_root(), image=args.image, min_gpus=8, dry_run=args.dry_run)
    run_root = host_run_dir(config, run_id)
    workspace_root = workspace_run_dir(run_id)
    evidence = evidence_helpers.ProductionEvidence(run_dir=run_root)
    evidence.record("run_id", run_id)
    evidence.record("workspace_run_dir", workspace_root)
    evidence.record("rl_model_source", "base_model")

    try:
        verify_pins(config)
    except SystemExit as exc:
        evidence.add_gate("submodule-pins", False, f"Submodule pin validation failed: {exc}")
        if not config.dry_run:
            evidence_helpers.write_evidence(evidence)
        raise SystemExit(system_exit_code(exc)) from exc
    evidence.add_gate("submodule-pins", True, "Expected submodule commits match")

    if not config.dry_run:
        state = patch_state(config)
        evidence.add_gate("bridge-patch", state == "applied", f"Bridge patch state: {state}")
        if state != "applied":
            evidence_helpers.write_evidence(evidence)
            raise SystemExit("Megatron-Bridge patch is not applied; run setup first")
    else:
        print("check Megatron-Bridge compatibility patch is applied")
        evidence.add_gate("bridge-patch", True, "Dry-run assumes setup applies the Bridge patch")

    try:
        require_image(config)
    except SystemExit as exc:
        evidence.add_gate("docker-image", False, f"Docker image validation failed: {exc}")
        if not config.dry_run:
            evidence_helpers.write_evidence(evidence)
        raise SystemExit(system_exit_code(exc)) from exc
    evidence.add_gate("docker-image", True, f"Docker image available: {config.image}")

    preflight_command = build_import_preflight_command(8)
    data_command = build_production_data_prep_command()
    sft_command = build_production_sft_command(
        train_path="/workspace/local/verl-data/gsm8k_sft/train.parquet",
        val_path="/workspace/local/verl-data/gsm8k_sft/test.parquet",
        output_dir=f"{workspace_root}/sft",
        total_steps=args.total_steps,
    )
    rl_command = build_production_rl_command(
        gsm8k_train_path="/workspace/local/verl-data/gsm8k/train.parquet",
        gsm8k_test_path="/workspace/local/verl-data/gsm8k/test.parquet",
        math_train_path="/workspace/local/verl-data/math/train.parquet",
        math_test_path="/workspace/local/verl-data/math/test.parquet",
        output_dir=f"{workspace_root}/rl",
        total_steps=args.total_steps,
    )

    evidence_dir = run_root / "evidence"
    evidence.record("sft_command", sft_command)
    evidence.record("rl_command", rl_command)

    if config.dry_run:
        print("container-preflight command")
        print(preflight_command)
        print("data command")
        print(data_command)
        print(f"write {evidence_dir / 'sft_command.sh'}")
        print(sft_command)
        print(f"write {evidence_dir / 'rl_command.sh'}")
        print(rl_command)
        print(f"Evidence would be written to {evidence_dir}")
        return

    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "sft_command.sh").write_text(sft_command + "\n", encoding="utf-8")
    (evidence_dir / "rl_command.sh").write_text(rl_command + "\n", encoding="utf-8")

    phases = [
        ("container-preflight", preflight_command),
        ("data", data_command),
        ("sft", sft_command),
        ("rl", rl_command),
    ]
    phase_commands = {phase: command for phase, command in phases}
    phase_artifacts = {
        phase: {
            "log_path": f"{workspace_root}/logs/{phase}.log",
            "exit_code_path": f"{workspace_root}/logs/{phase}.exitcode",
        }
        for phase, _ in phases
    }
    evidence.record("phase_commands", phase_commands)
    evidence.record("phase_artifacts", phase_artifacts)
    for phase, command in phases:
        artifacts = phase_artifacts[phase]
        logged = build_logged_inner_command(
            command,
            log_path=artifacts["log_path"],
            exit_code_path=artifacts["exit_code_path"],
        )
        try:
            run_command(build_docker_run_command(config, logged), cwd=config.repo_root, dry_run=config.dry_run)
        except subprocess.CalledProcessError as exc:
            evidence.add_gate(
                f"{phase}-exit-code",
                False,
                f"{phase} command failed with exit code {exc.returncode}",
            )
            evidence_helpers.write_evidence(evidence)
            raise SystemExit(exc.returncode) from exc
        evidence.add_gate(f"{phase}-exit-code", True, f"{phase} command completed")

    if not config.dry_run:
        sft_inventory = evidence_helpers.inventory_paths(run_root / "sft")
        rl_inventory = evidence_helpers.inventory_paths(run_root / "rl")
        evidence.record("sft_checkpoint_inventory", sft_inventory)
        evidence.record("rl_checkpoint_inventory", rl_inventory)
        evidence.add_gate(
            "sft-checkpoint-content",
            evidence_helpers.has_checkpoint_content(run_root / "sft"),
            f"{len(sft_inventory)} files under SFT output",
        )
        rl_log_path = run_root / "logs" / "rl.log"
        has_rl_update_step = evidence_helpers.has_rl_update_step(rl_log_path)
        evidence.add_gate(
            "rl-step-evidence",
            has_rl_update_step,
            f"training/global_step metric found in {rl_log_path.name}: {has_rl_update_step}",
        )
        fatal_matches = []
        post_completion_fatal_matches = []
        for log_path in sorted((run_root / "logs").glob("*.log")):
            fatal_matches.extend(
                evidence_helpers.scan_log_for_fatal_patterns(
                    log_path, ignore_after_completion=True
                )
            )
            post_completion_fatal_matches.extend(
                evidence_helpers.scan_log_for_post_completion_fatal_patterns(log_path)
            )
        evidence.record(
            "fatal_log_matches",
            [match.__dict__ for match in fatal_matches],
        )
        evidence.record(
            "post_completion_fatal_log_matches",
            [match.__dict__ for match in post_completion_fatal_matches],
        )
        evidence.add_gate(
            "fatal-log-scan",
            not fatal_matches,
            (
                f"{len(fatal_matches)} fatal log matches; "
                f"{len(post_completion_fatal_matches)} post-completion matches ignored"
            ),
        )

    evidence.add_gate("evidence-written", True, "Evidence files written")
    evidence_helpers.write_evidence(evidence)
    print(f"Evidence written to {run_root / 'evidence'}")


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

    validate = subparsers.add_parser(
        "validate-production",
        help="Run bounded SFT and RL production validation with evidence capture.",
    )
    validate.add_argument("--image", default=DEFAULT_IMAGE)
    validate.add_argument(
        "--run-id",
        default=None,
        help="Run directory name under local/verl-runs/production-validation.",
    )
    validate.add_argument(
        "--total-steps",
        type=int,
        default=2,
        help="Bounded step count for SFT and RL validation phases.",
    )
    validate.add_argument("--dry-run", action="store_true")
    validate.set_defaults(func=run_validate_production)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
