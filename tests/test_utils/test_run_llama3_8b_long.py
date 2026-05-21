# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import sys
from pathlib import Path

import tomllib

MODULE_PATH = Path("examples/llama/run_llama3_8b_long.py")
SPEC = importlib.util.spec_from_file_location("run_llama3_8b_long", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
llama3_long = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = llama3_long
SPEC.loader.exec_module(llama3_long)


def build_config(**overrides: object) -> object:
    """Build a test config with stable paths."""

    values = {
        "repo_root": Path("/repo"),
        "output_dir": Path("/repo/local/llama3"),
        "container_name": "llama3",
        "image": "megatron-lm:smoke",
    }
    values.update(overrides)
    return llama3_long.Llama3LongConfig(**values)


def test_docker_command_sets_host_user_and_repo_local_outputs() -> None:
    config = build_config()
    user = llama3_long.UserInfo(uid=1018, gid=100, name="philip")

    command = llama3_long.build_docker_command(config, user, detach=True)

    assert "-d" in command
    assert "--user" in command
    assert "1018:100" in command
    assert "USER=philip" in command
    assert "LOGNAME=philip" in command
    assert "/repo:/workspace:ro" in command
    assert "/repo/local/llama3:/outputs" in command
    assert "TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor" in command
    assert "CUDA_DEVICE_MAX_CONNECTIONS=1" in command


def test_docker_command_copies_source_to_writable_output_workdir() -> None:
    command = llama3_long.build_docker_command(
        build_config(), llama3_long.UserInfo(uid=1, gid=2, name="user"), detach=True
    )
    shell_command = command[-1]

    assert "mktemp -d /outputs/work/megatron-lm.XXXXXXXX" in shell_command
    assert "--exclude=local" in shell_command
    assert "--exclude=.agents" in shell_command
    assert "--exclude=.cache" in shell_command
    assert "--exclude=.venv" in shell_command
    assert "--exclude=.git" in shell_command
    assert "-C /workspace -cf -" in shell_command
    assert 'tar -C "$run_dir" -xf -' in shell_command
    assert 'cd "$run_dir"' in shell_command


def test_training_command_uses_llama3_8b_fp8_mock_data_shape() -> None:
    command = llama3_long.build_docker_command(
        build_config(duration_mins=235),
        llama3_long.UserInfo(uid=1, gid=2, name="user"),
        detach=True,
    )
    shell_command = command[-1]

    assert "torchrun --nproc_per_node=8 pretrain_gpt.py" in shell_command
    assert "--num-layers 32" in shell_command
    assert "--hidden-size 4096" in shell_command
    assert "--ffn-hidden-size 14336" in shell_command
    assert "--num-attention-heads 32" in shell_command
    assert "--num-query-groups 8" in shell_command
    assert "--seq-length 8192" in shell_command
    assert "--position-embedding-type rope" in shell_command
    assert "--normalization RMSNorm" in shell_command
    assert "--swiglu" in shell_command
    assert "--bf16" in shell_command
    assert "--transformer-impl transformer_engine" in shell_command
    assert "--fp8-format hybrid" in shell_command
    assert "--fp8-param-gather" in shell_command
    assert "--micro-batch-size 1" in shell_command
    assert "--global-batch-size 128" in shell_command
    assert "--exit-duration-in-mins 235" in shell_command
    assert "--mock-data" in shell_command
    assert "--tokenizer-type NullTokenizer" in shell_command
    assert "--vocab-size 128256" in shell_command
    assert "--data-cache-path /outputs/cache/data" in shell_command
    assert "--save /outputs/checkpoints" in shell_command
    assert "--load /outputs/checkpoints" not in shell_command


def test_resume_is_the_only_path_that_adds_load_checkpoint() -> None:
    fresh_command = llama3_long.build_training_command(build_config(resume=False))
    resume_command = llama3_long.build_training_command(build_config(resume=True))

    assert "--load /outputs/checkpoints" not in fresh_command
    assert "--load /outputs/checkpoints" in resume_command


def test_real_data_mode_uses_huggingface_tokenizer_and_data_path() -> None:
    command = llama3_long.build_training_command(
        build_config(tokenizer_model="/models/tokenizer", data_path="/data/prefix")
    )

    assert "--mock-data" not in command
    assert "--data-path /data/prefix" in command
    assert "--tokenizer-type HuggingFaceTokenizer" in command
    assert "--tokenizer-model /models/tokenizer" in command
    assert "--vocab-size 128256" in command


def test_partial_real_data_configuration_falls_back_to_mock_data() -> None:
    tokenizer_only = llama3_long.build_training_command(
        build_config(tokenizer_model="/models/tokenizer")
    )
    data_only = llama3_long.build_training_command(
        build_config(data_path="/data/prefix")
    )

    assert "--mock-data" in tokenizer_only
    assert "--mock-data" in data_only


def test_preflight_checks_fp8_and_eight_gpu_training_environment() -> None:
    preflight_command = llama3_long.build_preflight_command()

    assert "import torch" in preflight_command
    assert "import transformer_engine" in preflight_command
    assert "torch.cuda.device_count() >= 8" in preflight_command
    assert "torch.cuda.get_device_capability(0)" in preflight_command
    assert "FP8 requires Hopper, Ada, or Blackwell GPUs" in preflight_command
    assert "dist.init_process_group(backend='nccl')" in preflight_command
    assert "dist.all_reduce(value)" in preflight_command
    assert (
        "torchrun --nproc_per_node=8 /outputs/work/preflight_dist.py"
        in preflight_command
    )
    assert "/tmp" not in preflight_command


def test_preflight_only_runs_checks_and_prints_training_command() -> None:
    config = build_config(preflight=True, preflight_only=True)

    command = llama3_long.build_docker_command(
        config, llama3_long.UserInfo(uid=1, gid=2, name="user"), detach=False
    )
    shell_command = command[-1]

    assert "-d" not in command
    assert shell_command.index("preflight: torch=") < shell_command.index(
        "preflight: training command:"
    )
    assert "preflight: pretrain_gpt.py found" in shell_command
    assert shell_command.count("torchrun --nproc_per_node=8 pretrain_gpt.py") == 1
    assert "/tmp" not in " ".join(command)


def test_remove_container_command_targets_configured_name() -> None:
    assert llama3_long.build_remove_container_command("llama3") == [
        "docker",
        "rm",
        "-f",
        "llama3",
    ]


def test_mise_llama3_long_tasks_launch_repo_local_runner() -> None:
    mise_config = tomllib.loads(Path("mise.toml").read_text())

    assert mise_config["tasks"]["llama3-8b-long"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama3_8b_long.py"
    )
    assert mise_config["tasks"]["llama3-8b-long-preflight"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama3_8b_long.py "
        "--preflight --preflight-only --name mcore-llama3-8b-long-preflight "
        "--output-dir local/llama3-8b-long-preflight --foreground"
    )
    assert mise_config["tasks"]["llama3-8b-long-dry-run"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama3_8b_long.py --dry-run"
    )
    assert "/tmp" not in mise_config["tasks"]["llama3-8b-long"]["run"]
    assert "/tmp" not in mise_config["tasks"]["llama3-8b-long-preflight"]["run"]
    assert "/tmp" not in mise_config["tasks"]["llama3-8b-long-dry-run"]["run"]
