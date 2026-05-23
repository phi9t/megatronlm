import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_production_sft_command_uses_bounded_megatron_fsdp_run():
    tool = load_module()

    command = tool.build_production_sft_command(
        train_path="/workspace/local/verl-data/gsm8k_sft/train.parquet",
        val_path="/workspace/local/verl-data/gsm8k_sft/test.parquet",
        output_dir="/workspace/local/verl-runs/production-validation/run/sft",
        total_steps=2,
    )

    assert "torchrun" in command
    assert "--nproc_per_node=8" in command
    assert "verl.trainer.sft_trainer" in command
    assert "engine=megatron" in command
    assert "engine.use_megatron_fsdp=True" in command
    assert "engine.tensor_model_parallel_size=4" in command
    assert "model.path=Qwen/Qwen2.5-Math-7B" in command
    assert "trainer.total_training_steps=2" in command
    assert "checkpoint.save_contents=" in command
    assert '"model"' in command


def test_production_rl_command_uses_upstream_grpo_megatron_fsdp_shape():
    tool = load_module()

    command = tool.build_production_rl_command(
        gsm8k_train_path="/workspace/local/verl-data/gsm8k/train.parquet",
        gsm8k_test_path="/workspace/local/verl-data/gsm8k/test.parquet",
        math_train_path="/workspace/local/verl-data/math/train.parquet",
        math_test_path="/workspace/local/verl-data/math/test.parquet",
        output_dir="/workspace/local/verl-runs/production-validation/run/rl",
        total_steps=2,
    )

    assert "python3 -m verl.trainer.main_ppo" in command
    assert "--config-name=ppo_megatron_trainer.yaml" in command
    assert "actor_rollout_ref.actor.megatron.use_megatron_fsdp=True" in command
    assert "actor_rollout_ref.ref.megatron.use_megatron_fsdp=True" in command
    assert "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4" in command
    assert "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4" in command
    assert "data.train_files=" in command
    assert "/workspace/local/verl-data/gsm8k/train.parquet" in command
    assert "/workspace/local/verl-data/math/train.parquet" in command
    assert "data.val_files=" in command
    assert "/workspace/local/verl-data/gsm8k/test.parquet" in command
    assert "/workspace/local/verl-data/math/test.parquet" in command
    assert "trainer.n_gpus_per_node=8" in command
    assert "trainer.total_training_steps=2" in command
    assert "trainer.default_local_dir=/workspace/local/verl-runs/production-validation/run/rl" in command


def test_production_data_prep_command_creates_sft_and_rl_data():
    tool = load_module()

    command = tool.build_production_data_prep_command()

    assert "gsm8k_multiturn_sft.py" in command
    assert "gsm8k.py" in command
    assert "math_dataset.py" in command
    assert "/workspace/local/verl-data/gsm8k_sft" in command
    assert "/workspace/local/verl-data/gsm8k" in command
    assert "/workspace/local/verl-data/math" in command


def test_build_logged_inner_command_writes_exit_code_and_log():
    tool = load_module()

    command = tool.build_logged_inner_command(
        "python3 -V",
        log_path="/workspace/local/verl-runs/production-validation/run/logs/preflight.log",
        exit_code_path="/workspace/local/verl-runs/production-validation/run/logs/preflight.exitcode",
    )

    assert "set +e" in command
    assert "python3 -V" in command
    assert "{\n  python3 -V\n}" in command
    assert "2>&1 | tee /workspace/local/verl-runs/production-validation/run/logs/preflight.log" in command
    assert "echo ${PIPESTATUS[0]} > /workspace/local/verl-runs/production-validation/run/logs/preflight.exitcode" in command
    assert "exit $(cat /workspace/local/verl-runs/production-validation/run/logs/preflight.exitcode)" in command


def test_build_logged_inner_command_groups_compound_commands(tmp_path):
    tool = load_module()
    log_path = tmp_path / "compound.log"
    exit_code_path = tmp_path / "compound.exitcode"

    command = tool.build_logged_inner_command(
        "printf first && printf second && false",
        log_path=str(log_path),
        exit_code_path=str(exit_code_path),
    )
    result = subprocess.run(["bash", "-lc", command], text=True, capture_output=True, check=False)

    assert result.returncode != 0
    assert log_path.read_text() == "firstsecond"
    assert exit_code_path.read_text().strip() == "1"


def test_parser_accepts_validate_production_subcommand():
    tool = load_module()

    args = tool.build_parser().parse_args(["validate-production", "--dry-run", "--total-steps", "2"])

    assert args.command == "validate-production"
    assert args.dry_run is True
    assert args.total_steps == 2


def test_validate_production_dry_run_prints_sft_and_rl_commands(capsys):
    tool = load_module()

    exit_code = tool.main(["validate-production", "--dry-run", "--run-id", "unit-test", "--total-steps", "2"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "production-validation/unit-test" in output
    assert "verl.trainer.sft_trainer" in output
    assert "verl.trainer.main_ppo" in output
    assert "evidence/sft_command.sh" in output
    assert "evidence/rl_command.sh" in output


def test_validate_production_writes_evidence_on_phase_failure(tmp_path, monkeypatch):
    tool = load_module()

    monkeypatch.setattr(tool, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tool, "verify_pins", lambda config: None)
    monkeypatch.setattr(tool, "patch_state", lambda config: "applied")
    monkeypatch.setattr(tool, "require_image", lambda config: None)

    def fail_run_command(command, *, cwd, dry_run=False):
        raise subprocess.CalledProcessError(23, command)

    monkeypatch.setattr(tool, "run_command", fail_run_command)

    with pytest.raises(SystemExit) as exc_info:
        tool.main(["validate-production", "--run-id", "phase-failure", "--total-steps", "2"])

    assert exc_info.value.code == 23
    evidence_path = tmp_path / "local/verl-runs/production-validation/phase-failure/evidence/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    gate = next(gate for gate in evidence["gates"] if gate["name"] == "container-preflight-exit-code")
    assert evidence["status"] == "fail"
    assert gate["passed"] is False
    assert "exit code 23" in gate["detail"]
    assert evidence["records"]["phase_commands"]["container-preflight"]
    assert evidence["records"]["phase_artifacts"]["container-preflight"]["log_path"].endswith(
        "/logs/container-preflight.log"
    )
    assert evidence["records"]["phase_artifacts"]["container-preflight"]["exit_code_path"].endswith(
        "/logs/container-preflight.exitcode"
    )


def test_validate_production_writes_evidence_on_image_failure(tmp_path, monkeypatch):
    tool = load_module()

    monkeypatch.setattr(tool, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tool, "verify_pins", lambda config: None)
    monkeypatch.setattr(tool, "patch_state", lambda config: "applied")

    def fail_require_image(config):
        raise SystemExit("Docker image missing")

    monkeypatch.setattr(tool, "require_image", fail_require_image)

    with pytest.raises(SystemExit) as exc_info:
        tool.main(["validate-production", "--run-id", "image-failure", "--total-steps", "2"])

    assert exc_info.value.code == 1
    evidence_path = tmp_path / "local/verl-runs/production-validation/image-failure/evidence/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    gate = next(gate for gate in evidence["gates"] if gate["name"] == "docker-image")
    assert evidence["status"] == "fail"
    assert gate["passed"] is False
    assert "Docker image missing" in gate["detail"]
