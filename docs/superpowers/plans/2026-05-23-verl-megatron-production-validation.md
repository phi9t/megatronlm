# verl Megatron Production Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded production-validation command that runs verl Megatron-FSDP SFT and RL on one 8-GPU B200 node and writes structured evidence proving the run worked.

**Architecture:** Extend `tools/verl_megatron.py` with a `validate-production` subcommand that reuses the existing Docker, submodule, patch, and preflight primitives. Add focused evidence helpers in `tools/verl_megatron_evidence.py` so command construction and report writing stay testable without launching Docker. Keep all runtime outputs under `local/verl-runs/production-validation/<timestamp>/`.

**Tech Stack:** Python stdlib, Docker CLI, git CLI, verl vendored examples, Megatron-Bridge, Megatron-LM, `mise`, pytest-style unit tests.

---

## File Structure

Create:

* `tools/verl_megatron_evidence.py`: evidence dataclasses, JSON/Markdown writers, log scanner, dataset/checkpoint inventory helpers.
* `docs/superpowers/plans/2026-05-23-verl-megatron-production-validation.md`: this implementation plan.

Modify:

* `tools/verl_megatron.py`: production-validation command builders and orchestration.
* `tests/unit_tests/tools/test_verl_megatron.py`: command construction and CLI parser tests.
* Create `tests/unit_tests/tools/test_verl_megatron_evidence.py`: evidence helper tests.
* `mise.toml`: production-validation tasks.
* `docs/developer/verl-megatron.md`: production validation usage and evidence location.

Do not modify vendored files under `third_party/`. The setup tool applies the Bridge patch to the submodule worktree at runtime.

## Task 1: Evidence Helper Module

**Files:**
* Create: `tools/verl_megatron_evidence.py`
* Create: `tests/unit_tests/tools/test_verl_megatron_evidence.py`

- [ ] **Step 1: Write failing tests for fatal log scanning and evidence writing**

Add this file:

```python
import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    module_path = Path(__file__).parents[3] / "tools/verl_megatron_evidence.py"
    spec = importlib.util.spec_from_file_location("verl_megatron_evidence", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scan_log_flags_fatal_patterns(tmp_path):
    evidence = load_module()
    log_path = tmp_path / "rank0.log"
    log_path.write_text("step 1 ok\nRuntimeError: NCCL error\n", encoding="utf-8")

    matches = evidence.scan_log_for_fatal_patterns(log_path)

    assert matches == [
        evidence.FatalLogMatch(path=str(log_path), pattern="RuntimeError", line=2),
        evidence.FatalLogMatch(path=str(log_path), pattern="NCCL error", line=2),
    ]


def test_scan_log_ignores_benign_warnings(tmp_path):
    evidence = load_module()
    log_path = tmp_path / "rank0.log"
    log_path.write_text("FutureWarning: transformers experimental support\nstep 1 ok\n", encoding="utf-8")

    assert evidence.scan_log_for_fatal_patterns(log_path) == []


def test_write_evidence_files(tmp_path):
    evidence = load_module()
    run = evidence.ProductionEvidence(run_dir=tmp_path, status="pass")
    run.add_gate("container-gpu-count", True, "8 CUDA devices visible")
    run.add_gate("sft-exit-code", True, "SFT exited 0")
    run.record("versions", {"verl": "7dc39fec"})

    evidence.write_evidence(run)

    evidence_json = json.loads((tmp_path / "evidence/evidence.json").read_text(encoding="utf-8"))
    summary = (tmp_path / "evidence/summary.md").read_text(encoding="utf-8")
    assert evidence_json["status"] == "pass"
    assert evidence_json["gates"][0]["name"] == "container-gpu-count"
    assert evidence_json["records"]["versions"]["verl"] == "7dc39fec"
    assert "# verl Megatron Production Validation" in summary
    assert "- PASS: container-gpu-count - 8 CUDA devices visible" in summary
```

- [ ] **Step 2: Run tests and verify they fail because the module does not exist**

Run:

```bash
uv run python -m pytest tests/unit_tests/tools/test_verl_megatron_evidence.py -q
```

Expected: FAIL during import with `FileNotFoundError` for `tools/verl_megatron_evidence.py`.

- [ ] **Step 3: Implement the evidence helper module**

Create `tools/verl_megatron_evidence.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


FATAL_LOG_PATTERNS = (
    "Traceback",
    "RuntimeError",
    "CUDA out of memory",
    "NCCL error",
    "AssertionError",
    "KeyError",
)


@dataclass(frozen=True)
class FatalLogMatch:
    path: str
    pattern: str
    line: int


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ProductionEvidence:
    run_dir: Path
    status: str = "running"
    gates: list[GateResult] = field(default_factory=list)
    records: dict[str, Any] = field(default_factory=dict)

    def add_gate(self, name: str, passed: bool, detail: str) -> None:
        self.gates.append(GateResult(name=name, passed=passed, detail=detail))
        if not passed:
            self.status = "fail"

    def record(self, key: str, value: Any) -> None:
        self.records[key] = value

    def finish(self) -> None:
        if self.status != "fail":
            self.status = "pass" if all(gate.passed for gate in self.gates) else "fail"


def scan_log_for_fatal_patterns(path: Path) -> list[FatalLogMatch]:
    matches: list[FatalLogMatch] = []
    if not path.exists():
        return matches
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        for pattern in FATAL_LOG_PATTERNS:
            if pattern in line:
                matches.append(FatalLogMatch(path=str(path), pattern=pattern, line=line_number))
    return matches


def inventory_paths(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [
        {"path": str(path), "size_bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def write_evidence(evidence: ProductionEvidence) -> None:
    evidence.finish()
    evidence_dir = evidence.run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": evidence.status,
        "run_dir": str(evidence.run_dir),
        "gates": [asdict(gate) for gate in evidence.gates],
        "records": evidence.records,
    }
    (evidence_dir / "evidence.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (evidence_dir / "summary.md").write_text(render_summary(evidence), encoding="utf-8")


def render_summary(evidence: ProductionEvidence) -> str:
    lines = [
        "# verl Megatron Production Validation",
        "",
        f"Status: {evidence.status.upper()}",
        f"Run directory: `{evidence.run_dir}`",
        "",
        "## Gates",
        "",
    ]
    for gate in evidence.gates:
        status = "PASS" if gate.passed else "FAIL"
        lines.append(f"- {status}: {gate.name} - {gate.detail}")
    lines.extend(["", "## Records", ""])
    for key in sorted(evidence.records):
        lines.append(f"- `{key}`: `{evidence.records[key]}`")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run evidence helper tests**

Run:

```bash
uv run python -m pytest tests/unit_tests/tools/test_verl_megatron_evidence.py -q
```

Expected: PASS for all tests in `test_verl_megatron_evidence.py`.

- [ ] **Step 5: Commit evidence helper**

```bash
git add tools/verl_megatron_evidence.py tests/unit_tests/tools/test_verl_megatron_evidence.py
git commit -m "Add verl Megatron evidence helpers"
```

## Task 2: Production Command Builders

**Files:**
* Modify: `tools/verl_megatron.py`
* Modify: `tests/unit_tests/tools/test_verl_megatron.py`

- [ ] **Step 1: Add failing tests for SFT, RL, and data prep command builders**

Append these tests to `tests/unit_tests/tools/test_verl_megatron.py`:

```python
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
    assert "trainer.total_training_steps=2" in command
    assert "checkpoint.save_contents=" in command


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
```

- [ ] **Step 2: Run tests and verify the builders are missing**

Run:

```bash
uv run python -m pytest tests/unit_tests/tools/test_verl_megatron.py -q
```

Expected: FAIL with `AttributeError` for `build_production_sft_command`.

- [ ] **Step 3: Add production command builders**

In `tools/verl_megatron.py`, add these functions after `build_sft_smoke_command()`:

```python
def build_production_data_prep_command() -> str:
    return " && ".join(
        [
            "python3 examples/data_preprocess/gsm8k_multiturn_sft.py --local_save_dir /workspace/local/verl-data/gsm8k_sft",
            "python3 examples/data_preprocess/gsm8k.py --local_save_dir /workspace/local/verl-data/gsm8k",
            "python3 examples/data_preprocess/math_dataset.py --local_save_dir /workspace/local/verl-data/math",
        ]
    )


def build_production_sft_command(
    *,
    train_path: str,
    val_path: str,
    output_dir: str,
    total_steps: int,
) -> str:
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
```

- [ ] **Step 4: Run command builder tests**

Run:

```bash
uv run python -m pytest tests/unit_tests/tools/test_verl_megatron.py -q
```

Expected: PASS for the tests in `test_verl_megatron.py`, unless the repo-level torch import issue blocks collection. If collection is blocked by missing `torch`, run the direct assertions from Task 7.

- [ ] **Step 5: Commit command builders**

```bash
git add tools/verl_megatron.py tests/unit_tests/tools/test_verl_megatron.py
git commit -m "Add verl Megatron production command builders"
```

## Task 3: Evidence-Aware Process Runner

**Files:**
* Modify: `tools/verl_megatron.py`
* Modify: `tests/unit_tests/tools/test_verl_megatron.py`

- [ ] **Step 1: Add failing tests for run directory and log command construction**

Append this test to `tests/unit_tests/tools/test_verl_megatron.py`:

```python
def test_build_logged_inner_command_writes_exit_code_and_log():
    tool = load_module()

    command = tool.build_logged_inner_command(
        "python3 -V",
        log_path="/workspace/local/verl-runs/production-validation/run/logs/preflight.log",
        exit_code_path="/workspace/local/verl-runs/production-validation/run/logs/preflight.exitcode",
    )

    assert "set +e" in command
    assert "python3 -V" in command
    assert "2>&1 | tee /workspace/local/verl-runs/production-validation/run/logs/preflight.log" in command
    assert "echo ${PIPESTATUS[0]} > /workspace/local/verl-runs/production-validation/run/logs/preflight.exitcode" in command
    assert "exit $(cat /workspace/local/verl-runs/production-validation/run/logs/preflight.exitcode)" in command
```

- [ ] **Step 2: Run the test and verify the helper is missing**

Run:

```bash
uv run python -m pytest tests/unit_tests/tools/test_verl_megatron.py::test_build_logged_inner_command_writes_exit_code_and_log -q
```

Expected: FAIL with `AttributeError` for `build_logged_inner_command`.

- [ ] **Step 3: Implement logged command wrapper**

Add this function to `tools/verl_megatron.py` after `format_command()`:

```python
def build_logged_inner_command(inner_command: str, *, log_path: str, exit_code_path: str) -> str:
    return textwrap.dedent(
        f"""
        mkdir -p {shlex.quote(str(Path(log_path).parent))}
        set +e
        {inner_command} 2>&1 | tee {shlex.quote(log_path)}
        echo ${{PIPESTATUS[0]}} > {shlex.quote(exit_code_path)}
        exit $(cat {shlex.quote(exit_code_path)})
        """
    ).strip()
```

- [ ] **Step 4: Run the logged command test**

Run:

```bash
uv run python -m pytest tests/unit_tests/tools/test_verl_megatron.py::test_build_logged_inner_command_writes_exit_code_and_log -q
```

Expected: PASS, or use the direct assertion fallback if repo-level torch import blocks pytest collection.

- [ ] **Step 5: Commit logged command wrapper**

```bash
git add tools/verl_megatron.py tests/unit_tests/tools/test_verl_megatron.py
git commit -m "Add logged verl Megatron command wrapper"
```

## Task 4: Production Validation Orchestrator

**Files:**
* Modify: `tools/verl_megatron.py`
* Modify: `tests/unit_tests/tools/test_verl_megatron.py`

- [ ] **Step 1: Add failing parser and dry-run tests**

Append these tests to `tests/unit_tests/tools/test_verl_megatron.py`:

```python
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
```

- [ ] **Step 2: Run parser tests and verify missing subcommand**

Run:

```bash
uv run python -m pytest tests/unit_tests/tools/test_verl_megatron.py::test_parser_accepts_validate_production_subcommand -q
```

Expected: FAIL because `validate-production` is not registered.

- [ ] **Step 3: Add production validation runtime constants and helpers**

Add imports near the top of `tools/verl_megatron.py`:

```python
import datetime as dt
import importlib.util
```

Add this script-local import helper after `repo_root()` so `tools/verl_megatron.py`
works when executed directly:

```python

def load_evidence_helpers():
    module_path = Path(__file__).parent / "verl_megatron_evidence.py"
    spec = importlib.util.spec_from_file_location("verl_megatron_evidence", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evidence_helpers = load_evidence_helpers()
```

Add these helpers:

```python
def default_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def workspace_run_dir(run_id: str) -> str:
    return f"/workspace/local/verl-runs/production-validation/{run_id}"


def host_run_dir(config: VerLMegatronConfig, run_id: str) -> Path:
    return config.repo_root / "local/verl-runs/production-validation" / run_id
```

- [ ] **Step 4: Implement `run_validate_production`**

Add this function before `build_parser()`:

```python
def run_validate_production(args: argparse.Namespace) -> None:
    run_id = args.run_id or default_run_id()
    config = VerLMegatronConfig(
        repo_root=repo_root(),
        image=args.image,
        min_gpus=8,
        dry_run=args.dry_run,
    )
    run_root = host_run_dir(config, run_id)
    workspace_root = workspace_run_dir(run_id)
    evidence = evidence_helpers.ProductionEvidence(run_dir=run_root)
    evidence.record("run_id", run_id)
    evidence.record("workspace_run_dir", workspace_root)
    evidence.record("rl_model_source", "base_model")

    verify_pins(config)
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

    require_image(config)
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
    for phase, command in phases:
        logged = build_logged_inner_command(
            command,
            log_path=f"{workspace_root}/logs/{phase}.log",
            exit_code_path=f"{workspace_root}/logs/{phase}.exitcode",
        )
        run_command(build_docker_run_command(config, logged), cwd=config.repo_root, dry_run=config.dry_run)
        evidence.add_gate(f"{phase}-exit-code", True, f"{phase} command completed")

    evidence.add_gate("evidence-written", True, "Evidence files written")
    evidence_helpers.write_evidence(evidence)
    print(f"Evidence written to {run_root / 'evidence'}")
```

- [ ] **Step 5: Register CLI parser arguments**

In `build_parser()`, add:

```python
    validate = subparsers.add_parser("validate-production", help="Run bounded SFT and RL production validation with evidence capture.")
    validate.add_argument("--image", default=DEFAULT_IMAGE)
    validate.add_argument("--run-id", default=None, help="Run directory name under local/verl-runs/production-validation.")
    validate.add_argument("--total-steps", type=int, default=2, help="Bounded step count for SFT and RL validation phases.")
    validate.add_argument("--dry-run", action="store_true")
    validate.set_defaults(func=run_validate_production)
```

- [ ] **Step 6: Run dry-run tests**

Run:

```bash
uv run python -m pytest tests/unit_tests/tools/test_verl_megatron.py::test_parser_accepts_validate_production_subcommand tests/unit_tests/tools/test_verl_megatron.py::test_validate_production_dry_run_prints_sft_and_rl_commands -q
```

Expected: PASS, or use direct assertion fallback if repo-level torch import blocks pytest collection.

- [ ] **Step 7: Commit orchestrator**

```bash
git add tools/verl_megatron.py tests/unit_tests/tools/test_verl_megatron.py
git commit -m "Add verl Megatron production validation command"
```

## Task 5: Evidence Gates and Inventories

**Files:**
* Modify: `tools/verl_megatron.py`
* Modify: `tools/verl_megatron_evidence.py`
* Modify: `tests/unit_tests/tools/test_verl_megatron_evidence.py`

- [ ] **Step 1: Add failing tests for checkpoint and log evidence gates**

Append these tests to `tests/unit_tests/tools/test_verl_megatron_evidence.py`:

```python
def test_checkpoint_inventory_detects_model_content(tmp_path):
    evidence = load_module()
    ckpt = tmp_path / "global_step_2/model"
    ckpt.mkdir(parents=True)
    (ckpt / "distcp_metadata").write_text("metadata", encoding="utf-8")

    inventory = evidence.inventory_paths(tmp_path)

    assert any(item["path"].endswith("distcp_metadata") for item in inventory)
    assert evidence.has_checkpoint_content(tmp_path) is True


def test_checkpoint_inventory_fails_when_empty(tmp_path):
    evidence = load_module()

    assert evidence.has_checkpoint_content(tmp_path) is False
```

- [ ] **Step 2: Run tests and verify `has_checkpoint_content` is missing**

Run:

```bash
uv run python -m pytest tests/unit_tests/tools/test_verl_megatron_evidence.py::test_checkpoint_inventory_detects_model_content -q
```

Expected: FAIL with `AttributeError` for `has_checkpoint_content`.

- [ ] **Step 3: Add checkpoint helper**

Add this function to `tools/verl_megatron_evidence.py`:

```python
def has_checkpoint_content(root: Path) -> bool:
    if not root.exists():
        return False
    for path in root.rglob("*"):
        if path.is_file() and path.name in {"distcp_metadata", "metadata.json", "latest_checkpointed_iteration.txt"}:
            return True
    return any(path.is_file() for path in root.rglob("*.pt"))
```

- [ ] **Step 4: Add post-phase evidence checks to orchestrator**

In `run_validate_production`, after the phase loop, add checks:

```python
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
        evidence.add_gate(
            "rl-step-evidence",
            bool(list((run_root / "logs").glob("rl.log"))),
            "RL log exists; metric parsing is recorded in the log artifact",
        )
        fatal_matches = []
        for log_path in sorted((run_root / "logs").glob("*.log")):
            fatal_matches.extend(evidence_helpers.scan_log_for_fatal_patterns(log_path))
        evidence.record("fatal_log_matches", [match.__dict__ for match in fatal_matches])
        evidence.add_gate("fatal-log-scan", not fatal_matches, f"{len(fatal_matches)} fatal log matches")
```

Place this before the final `evidence-written` gate.

- [ ] **Step 5: Run evidence tests**

Run:

```bash
uv run python -m pytest tests/unit_tests/tools/test_verl_megatron_evidence.py -q
```

Expected: PASS for all evidence helper tests.

- [ ] **Step 6: Commit evidence gates**

```bash
git add tools/verl_megatron.py tools/verl_megatron_evidence.py tests/unit_tests/tools/test_verl_megatron_evidence.py
git commit -m "Add verl Megatron production evidence gates"
```

## Task 6: mise Task and Developer Docs

**Files:**
* Modify: `mise.toml`
* Modify: `docs/developer/verl-megatron.md`

- [ ] **Step 1: Add mise tasks**

Append to `mise.toml`:

```toml
[tasks.verl-megatron-production-validation]
description = "Run bounded SFT and RL production validation for verl Megatron-FSDP on one 8-GPU B200 node."
run = "UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py validate-production"

[tasks.verl-megatron-production-validation-dry-run]
description = "Print the bounded SFT and RL production validation commands without running them."
run = "UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py validate-production --dry-run"
```

- [ ] **Step 2: Update developer docs**

Add this section to `docs/developer/verl-megatron.md` after the “One-Step Smoke” section:

~~~markdown
## Production Validation

Run the bounded SFT plus RL production validation on a single 8-GPU B200 node:

```bash
mise run verl-megatron-production-validation
```

For command review without launching training:

```bash
mise run verl-megatron-production-validation-dry-run
```

Each run writes a timestamped evidence directory under:

```text
local/verl-runs/production-validation/<timestamp>/evidence
```

The evidence includes `evidence.json`, `summary.md`, `sft_command.sh`,
`rl_command.sh`, environment records, logs, checkpoint inventories, and pass/fail
gates. The first production validation uses the base Qwen/Qwen2.5-Math-7B model
for RL and verifies the SFT phase through its own checkpoint output.
~~~

- [ ] **Step 3: Run docs whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Commit docs and mise tasks**

```bash
git add mise.toml docs/developer/verl-megatron.md
git commit -m "Document verl Megatron production validation"
```

## Task 7: Dry-Run Verification

**Files:**
* No source files unless a verification failure requires a fix.

- [ ] **Step 1: Run import ordering after Python edits**

Run:

```bash
uv run isort tools/verl_megatron.py tools/verl_megatron_evidence.py tests/unit_tests/tools/test_verl_megatron.py tests/unit_tests/tools/test_verl_megatron_evidence.py
```

Expected: command exits 0. If isort rewrites files, inspect and commit those formatting changes with the relevant task commit.

- [ ] **Step 2: Compile Python files**

Run:

```bash
python -m py_compile tools/verl_megatron.py tools/verl_megatron_evidence.py tests/unit_tests/tools/test_verl_megatron.py tests/unit_tests/tools/test_verl_megatron_evidence.py
```

Expected: exit code 0.

- [ ] **Step 3: Run direct assertions if pytest collection is blocked by missing torch**

Run:

```bash
python - <<'PY'
import importlib.util
import sys
from pathlib import Path

module_path = Path("tools/verl_megatron.py")
spec = importlib.util.spec_from_file_location("verl_megatron_tool", module_path)
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)

args = tool.build_parser().parse_args(["validate-production", "--dry-run", "--run-id", "verify", "--total-steps", "2"])
assert args.command == "validate-production"
assert args.dry_run is True
sft = tool.build_production_sft_command(
    train_path="/workspace/local/verl-data/gsm8k_sft/train.parquet",
    val_path="/workspace/local/verl-data/gsm8k_sft/test.parquet",
    output_dir="/workspace/local/verl-runs/production-validation/verify/sft",
    total_steps=2,
)
rl = tool.build_production_rl_command(
    gsm8k_train_path="/workspace/local/verl-data/gsm8k/train.parquet",
    gsm8k_test_path="/workspace/local/verl-data/gsm8k/test.parquet",
    math_train_path="/workspace/local/verl-data/math/train.parquet",
    math_test_path="/workspace/local/verl-data/math/test.parquet",
    output_dir="/workspace/local/verl-runs/production-validation/verify/rl",
    total_steps=2,
)
assert "verl.trainer.sft_trainer" in sft
assert "engine.use_megatron_fsdp=True" in sft
assert "verl.trainer.main_ppo" in rl
assert "actor_rollout_ref.actor.megatron.use_megatron_fsdp=True" in rl
print("direct production command assertions passed")
PY
```

Expected: prints `direct production command assertions passed`.

- [ ] **Step 4: Run dry-run command**

Run:

```bash
UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py validate-production --dry-run --run-id verify --total-steps 2
```

Expected: prints submodule pin checks, Bridge patch check, Docker image check, container preflight command, data command, SFT command, RL command, and evidence path under `local/verl-runs/production-validation/verify/evidence`.

- [ ] **Step 5: Run git whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

## Task 8: Full B200 Production Validation Run

**Files:**
* Runtime outputs only under `local/verl-runs/production-validation/<timestamp>/`.
* No source files unless the run exposes a defect that must be fixed.

- [ ] **Step 1: Ensure setup is applied**

Run:

```bash
UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py setup --skip-data
```

Expected:

```text
verl commit ok: 7dc39fec1c37da4098b50e68840e78906b69f16a
Megatron-Bridge commit ok: 94dc04baf65157463181eef0c19549d5a6e4ccec
Megatron-Core nested submodule commit ok: 38986a98aae6a0cc4c8ae7b435db3288a890b0cb
Megatron-Bridge compatibility patch already applied
Docker image exists: verl-megatron-fsdp:local
```

- [ ] **Step 2: Run fast preflight**

Run:

```bash
UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py preflight
```

Expected:

```text
Megatron-Bridge compatibility patch ok
Docker image ok: verl-megatron-fsdp:local
imports ok; cuda_device_count=8
```

- [ ] **Step 3: Run the production validation**

Run:

```bash
UV_PROJECT_ENVIRONMENT=local/uv-anchor uv run python tools/verl_megatron.py validate-production --total-steps 2
```

Expected: exit code 0 and printed path to `local/verl-runs/production-validation/<timestamp>/evidence`.

- [ ] **Step 4: Inspect evidence files**

Run:

```bash
latest_run=$(ls -td local/verl-runs/production-validation/* | head -1)
test -f "$latest_run/evidence/evidence.json"
test -f "$latest_run/evidence/summary.md"
python -m json.tool "$latest_run/evidence/evidence.json" >/dev/null
sed -n '1,160p' "$latest_run/evidence/summary.md"
```

Expected:

```text
Status: PASS
- PASS: submodule-pins
- PASS: bridge-patch
- PASS: docker-image
- PASS: container-preflight-exit-code
- PASS: data-exit-code
- PASS: sft-exit-code
- PASS: rl-exit-code
- PASS: sft-checkpoint-content
- PASS: fatal-log-scan
```

- [ ] **Step 5: Commit any fixes required by the full run**

If the production validation exposes a code or command defect, fix it with the smallest scoped patch, rerun Tasks 7 and 8, then commit:

```bash
git add tools/verl_megatron.py tools/verl_megatron_evidence.py tests/unit_tests/tools/test_verl_megatron.py tests/unit_tests/tools/test_verl_megatron_evidence.py docs/developer/verl-megatron.md mise.toml
git commit -m "Fix verl Megatron production validation"
```

If the run passes without source changes, do not commit runtime files from `local/`.

## Task 9: Final Verification and Handoff

**Files:**
* No new files unless verification reveals a source defect.

- [ ] **Step 1: Run final status and whitespace checks**

Run:

```bash
git status --short
git diff --check
```

Expected: `git status --short` shows no source changes, except possibly a dirty `third_party/Megatron-Bridge` worktree from the applied runtime patch. `git diff --check` exits 0.

- [ ] **Step 2: Summarize evidence**

Run:

```bash
export LATEST_RUN=$(ls -td local/verl-runs/production-validation/* | head -1)
python - <<'PY'
import json
import os
from pathlib import Path

latest = Path(os.environ["LATEST_RUN"])
payload = json.loads((latest / "evidence/evidence.json").read_text(encoding="utf-8"))
print(payload["status"])
for gate in payload["gates"]:
    print(f"{gate['name']}: {gate['passed']} - {gate['detail']}")
PY
```

Expected: first line is `pass`; all required gates print `True`.

- [ ] **Step 3: Final response**

Report:

* commit IDs created during implementation
* exact verification commands run
* evidence directory path
* `summary.md` status
* any known caveats, including whether pytest collection was blocked by missing host `torch`
