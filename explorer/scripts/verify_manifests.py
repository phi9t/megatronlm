#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "explorer" / "public" / "data"
MAX_LABEL = 28


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_ref(ref: str) -> str | None:
    file = ref.split(":", 1)[0]
    raw = Path(file)
    if raw.is_absolute() or ".." in raw.parts:
        return f"source ref must be repo-relative: {ref}"
    candidate = (ROOT / raw).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return f"source ref escapes repo: {ref}"
    if not candidate.is_file():
        return f"missing source file: {ref}"
    return None


def check_graph(path: Path) -> list[str]:
    errors: list[str] = []
    data = load(path)
    for node in data.get("nodes", []):
        label = node.get("label", "")
        if len(label) > MAX_LABEL:
            errors.append(f"{path}: label too long: {label}")
        node_ref = node.get("ref") or node.get("file")
        if node_ref:
            err = check_ref(str(node_ref))
            if err:
                errors.append(f"{path}: {err}")
    return errors


def check_model(path: Path) -> list[str]:
    errors: list[str] = []
    data = load(path)
    blocks = list(data.get("prelude", [])) + list(data.get("head", []))
    for group in data.get("layers", []):
        for branch in group.get("branches", []):
            blocks.append(branch.get("preNorm", {}))
            blocks.extend(branch.get("steps", []))
    for block in blocks:
        label = block.get("label", "")
        if len(label) > 28:
            errors.append(f"{path}: block label too long: {label}")
        block_ref = block.get("ref")
        if block_ref:
            err = check_ref(str(block_ref))
            if err:
                errors.append(f"{path}: {err}")
    return errors


def main() -> None:
    errors: list[str] = []
    for graph in (DATA / "graphs").glob("*.json"):
        if graph.name != "index.json":
            errors.extend(check_graph(graph))
    for model in (DATA / "models").glob("*.json"):
        if model.name != "index.json":
            errors.extend(check_model(model))
    if not (DATA / "guide.md").is_file():
        errors.append("missing guide.md")
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("manifest verification passed")


if __name__ == "__main__":
    main()
