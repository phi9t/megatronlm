from __future__ import annotations

import json
import re
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
COMPLETION_LOG_PATTERNS = (
    "Training Progress: 100%",
    "Final validation metrics:",
)
CHECKPOINT_CONTENT_NAMES = {"distcp_metadata", "metadata.json", ".metadata"}
CHECKPOINT_CONTENT_SUFFIXES = {".distcp", ".pt"}
RL_UPDATE_STEP_PATTERN = re.compile(
    r"""["']?training/global_step["']?\s*[:=]\s*([0-9]+)"""
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


def line_has_completion_marker(line: str) -> bool:
    return any(pattern in line for pattern in COMPLETION_LOG_PATTERNS)


def scan_log_for_fatal_patterns(
    path: Path, *, ignore_after_completion: bool = False
) -> list[FatalLogMatch]:
    matches: list[FatalLogMatch] = []
    if not path.exists():
        return matches
    completion_seen = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if line_has_completion_marker(line):
            completion_seen = True
        if ignore_after_completion and completion_seen:
            continue
        for pattern in FATAL_LOG_PATTERNS:
            if pattern in line:
                matches.append(
                    FatalLogMatch(path=str(path), pattern=pattern, line=line_number)
                )
    return matches


def scan_log_for_post_completion_fatal_patterns(path: Path) -> list[FatalLogMatch]:
    matches: list[FatalLogMatch] = []
    if not path.exists():
        return matches
    completion_seen = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if line_has_completion_marker(line):
            completion_seen = True
            continue
        if not completion_seen:
            continue
        for pattern in FATAL_LOG_PATTERNS:
            if pattern in line:
                matches.append(
                    FatalLogMatch(path=str(path), pattern=pattern, line=line_number)
                )
    return matches


def inventory_paths(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [
        {"path": str(path), "size_bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def has_checkpoint_content(root: Path) -> bool:
    if not root.exists():
        return False
    for path in root.rglob("*"):
        if path.is_file() and path.name in CHECKPOINT_CONTENT_NAMES:
            return True
        if path.is_file() and path.suffix in CHECKPOINT_CONTENT_SUFFIXES:
            return True
    return False


def has_rl_update_step(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RL_UPDATE_STEP_PATTERN.search(line)
        if match and int(match.group(1)) > 0:
            return True
    return False


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
    (evidence_dir / "evidence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "summary.md").write_text(
        render_summary(evidence),
        encoding="utf-8",
    )


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
