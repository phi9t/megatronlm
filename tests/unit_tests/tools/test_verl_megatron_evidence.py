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
    log_path.write_text(
        "FutureWarning: transformers experimental support\nstep 1 ok\n",
        encoding="utf-8",
    )

    assert evidence.scan_log_for_fatal_patterns(log_path) == []


def test_write_evidence_files(tmp_path):
    evidence = load_module()
    run = evidence.ProductionEvidence(run_dir=tmp_path, status="pass")
    run.add_gate("container-gpu-count", True, "8 CUDA devices visible")
    run.add_gate("sft-exit-code", True, "SFT exited 0")
    run.record("versions", {"verl": "7dc39fec"})

    evidence.write_evidence(run)

    evidence_json = json.loads(
        (tmp_path / "evidence/evidence.json").read_text(encoding="utf-8")
    )
    summary = (tmp_path / "evidence/summary.md").read_text(encoding="utf-8")
    assert evidence_json["status"] == "pass"
    assert evidence_json["gates"][0]["name"] == "container-gpu-count"
    assert evidence_json["records"]["versions"]["verl"] == "7dc39fec"
    assert "# verl Megatron Production Validation" in summary
    assert "- PASS: container-gpu-count - 8 CUDA devices visible" in summary
