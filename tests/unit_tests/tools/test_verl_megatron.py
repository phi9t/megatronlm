import importlib.util
import sys
from pathlib import Path


def load_module():
    module_path = Path(__file__).parents[3] / "tools/verl_megatron.py"
    spec = importlib.util.spec_from_file_location("verl_megatron_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pythonpath_points_at_vendored_repos():
    tool = load_module()

    pythonpath = tool.build_pythonpath()

    assert "/workspace/third_party/verl" in pythonpath
    assert "/workspace/third_party/Megatron-Bridge/src" in pythonpath
    assert "/workspace/third_party/Megatron-Bridge/3rdparty/Megatron-LM" in pythonpath


def test_sft_smoke_command_uses_working_megatron_fsdp_shape():
    tool = load_module()

    command = tool.build_sft_smoke_command()

    assert "engine.use_megatron_fsdp=True" in command
    assert "engine.tensor_model_parallel_size=4" in command
    assert "checkpoint.save_contents=" in command
    assert "model" in command
    assert "/workspace/local/verl-data/gsm8k_sft/train.parquet" in command


def test_docker_run_command_mounts_repo_and_runtime_env():
    tool = load_module()
    config = tool.VerLMegatronConfig(repo_root=Path("."), image="test-image")

    command = tool.build_docker_run_command(config, "python3 -V")

    assert command[:2] == ["docker", "run"]
    assert "--entrypoint" in command
    assert "type=bind,source=.,target=/workspace" in command
    assert "PYTHONPATH=/workspace/third_party/verl:/workspace/third_party/Megatron-Bridge/src:/workspace/third_party/Megatron-Bridge/3rdparty/Megatron-LM" in command
    assert "HF_HOME=/workspace/local/hf-home" in command
    assert "test-image" in command
    assert "/workspace/third_party/verl" in command


def test_image_build_command_uses_repo_owned_dockerfile():
    tool = load_module()
    config = tool.VerLMegatronConfig(repo_root=Path("."), image="test-image", base_image="base-image")

    command = tool.build_image_command(config)

    assert command == [
        "docker",
        "build",
        "--build-arg",
        "BASE_IMAGE=base-image",
        "-t",
        "test-image",
        "-f",
        "tools/verl_megatron/Dockerfile",
        ".",
    ]
