# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from pathlib import Path

import pytest
import tomllib
from pydantic import ValidationError

EXAMPLES_LLAMA = Path("examples/llama").resolve()
if str(EXAMPLES_LLAMA) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_LLAMA))

from llama_config import (
    LlamaConfigError,
    LlamaRunConfig,
    UserInfo,
    build_docker_command,
    load_config,
    resolve_config_path,
    with_operational_overrides,
)
from run_llama import parse_args

CONFIG_DIR = Path("examples/llama/configs")


def test_all_checked_in_yaml_presets_validate() -> None:
    for config_path in CONFIG_DIR.glob("*.yaml"):
        config = load_config(config_path)
        assert isinstance(config, LlamaRunConfig)


def test_unknown_yaml_fields_fail_validation() -> None:
    raw = load_config(CONFIG_DIR / "llama_anchor.yaml").model_dump(mode="json")
    raw["runtime"]["unknown"] = "value"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LlamaRunConfig.model_validate(raw)


def test_invalid_real_data_configuration_fails_validation() -> None:
    raw = load_config(CONFIG_DIR / "llama_anchor.yaml").model_dump(mode="json")
    raw["data"]["mode"] = "real"
    raw["data"]["tokenizer_model"] = "/models/tokenizer"

    with pytest.raises(ValidationError, match="real data mode requires"):
        LlamaRunConfig.model_validate(raw)


def test_env_var_selects_config_when_cli_path_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = (CONFIG_DIR / "llama_anchor.yaml").resolve()
    monkeypatch.setenv("LLAMA_CONFIG", str(config_path))

    assert resolve_config_path(None) == config_path


def test_missing_config_selection_reports_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLAMA_CONFIG", raising=False)

    with pytest.raises(LlamaConfigError, match="set --config or LLAMA_CONFIG"):
        resolve_config_path(None)


def test_default_config_is_used_when_cli_and_env_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = (CONFIG_DIR / "llama_anchor.yaml").resolve()
    monkeypatch.delenv("LLAMA_CONFIG", raising=False)

    assert resolve_config_path(None, default_path=config_path) == config_path


def test_yaml_launcher_accepts_only_operational_cli_flags() -> None:
    args = parse_args(
        [
            "--config",
            "examples/llama/configs/llama_anchor.yaml",
            "--foreground",
            "--dry-run",
            "--preflight-only",
        ]
    )

    assert args.config == "examples/llama/configs/llama_anchor.yaml"
    assert args.foreground is True
    assert args.dry_run is True
    assert args.preflight_only is True


def test_operational_overrides_do_not_change_training_configuration() -> None:
    config = load_config(CONFIG_DIR / "llama_anchor.yaml")
    overridden = with_operational_overrides(
        config, foreground=True, dry_run=True, preflight_only=True
    )

    assert overridden.runtime.foreground is True
    assert overridden.runtime.dry_run is True
    assert overridden.preflight.only is True
    assert overridden.training.model_dump() == config.training.model_dump()
    assert overridden.model.model_dump() == config.model.model_dump()


def test_resume_is_configured_in_yaml_not_a_training_override() -> None:
    config = load_config(CONFIG_DIR / "llama3_8b_long.yaml")
    resumed = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"resume": True})}
    )

    assert "--load /outputs/checkpoints" not in config.training_command()
    assert "--load /outputs/checkpoints" in resumed.training_command()


def test_real_data_mode_renders_huggingface_tokenizer_args() -> None:
    raw = load_config(CONFIG_DIR / "llama_anchor.yaml").model_dump(mode="json")
    raw["data"].update(
        {
            "mode": "real",
            "tokenizer_model": "/models/tokenizer",
            "data_path": "/data/prefix",
        }
    )
    config = LlamaRunConfig.model_validate(raw)

    command = config.training_command()

    assert "--mock-data" not in command
    assert "--data-path /data/prefix" in command
    assert "--tokenizer-type HuggingFaceTokenizer" in command
    assert "--tokenizer-model /models/tokenizer" in command


def test_docker_command_keeps_source_readonly_and_uses_local_outputs() -> None:
    config = load_config(CONFIG_DIR / "llama_anchor.yaml")

    command = build_docker_command(
        config, Path("/repo"), UserInfo(uid=1018, gid=100, name="philip")
    )
    shell_command = command[-1]

    assert "--user" in command
    assert "1018:100" in command
    assert "USER=philip" in command
    assert "LOGNAME=philip" in command
    assert "/repo:/workspace:ro" in command
    assert "/repo/local/llama-anchor:/outputs" in command
    assert "mktemp -d /outputs/work/megatron-lm.XXXXXXXX" in shell_command
    assert "--exclude=local" in shell_command
    assert "-C /workspace -cf -" in shell_command
    assert 'tar -C "$run_dir" -xf -' in shell_command
    assert "/tmp" not in " ".join(command)


def test_tmp_paths_fail_validation() -> None:
    raw = load_config(CONFIG_DIR / "llama_anchor.yaml").model_dump(mode="json")
    raw["runtime"]["output_dir"] = "/tmp/llama"

    with pytest.raises(ValidationError, match="repo-local paths"):
        LlamaRunConfig.model_validate(raw)


def test_mise_llama_tasks_launch_yaml_runner() -> None:
    mise_config = tomllib.loads(Path("mise.toml").read_text())

    assert mise_config["tasks"]["llama-anchor"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama.py "
        "--config examples/llama/configs/llama_anchor.yaml"
    )
    assert mise_config["tasks"]["llama-anchor-smoke"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama.py "
        "--config examples/llama/configs/llama_anchor_smoke.yaml"
    )
    assert mise_config["tasks"]["llama-anchor-preflight"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama.py "
        "--config examples/llama/configs/llama_anchor_preflight.yaml"
    )
    assert mise_config["tasks"]["llama3-8b-long"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama.py "
        "--config examples/llama/configs/llama3_8b_long.yaml"
    )
    assert mise_config["tasks"]["llama3-8b-long-preflight"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama.py "
        "--config examples/llama/configs/llama3_8b_preflight.yaml"
    )
    assert mise_config["tasks"]["llama3-8b-long-dry-run"]["run"] == (
        "UV_PROJECT_ENVIRONMENT=local/uv-anchor "
        "uv run python examples/llama/run_llama.py "
        "--config examples/llama/configs/llama3_8b_long.yaml --dry-run"
    )
    for task_name, task in mise_config["tasks"].items():
        if task_name.startswith("llama"):
            assert "/tmp" not in task["run"]
