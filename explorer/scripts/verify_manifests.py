#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
    for field in ("model", "slug", "family", "source", "config", "prelude", "layers", "head"):
        if field not in data:
            errors.append(f"{path}: missing model field: {field}")
    if data.get("family") not in {"dense-qknorm", "moe-qknorm", "mla-moe"}:
        errors.append(f"{path}: unknown model family: {data.get('family')}")
    errors.extend(check_finite_numbers(path, data))
    blocks = list(data.get("prelude", [])) + list(data.get("head", []))
    for group in data.get("layers", []):
        if not isinstance(group.get("repeat"), int) or group.get("repeat", 0) < 0:
            errors.append(f"{path}: invalid layer repeat: {group.get('repeat')}")
        for branch in group.get("branches", []):
            if not branch.get("name"):
                errors.append(f"{path}: missing branch name")
            blocks.append(branch.get("preNorm", {}))
            blocks.extend(branch.get("steps", []))
    for block in blocks:
        if not block:
            continue
        for field in ("id", "type", "label", "symbol", "ref", "kind", "desc"):
            if field not in block:
                errors.append(f"{path}: block missing field {field}: {block.get('id', '<unknown>')}")
        label = block.get("label", "")
        if len(label) > 28:
            errors.append(f"{path}: block label too long: {label}")
        block_ref = block.get("ref")
        if block_ref:
            err = check_ref(str(block_ref))
            if err:
                errors.append(f"{path}: {err}")
    return errors


def check_parallelism(path: Path) -> list[str]:
    errors: list[str] = []
    data = load(path)
    strategies = data.get("strategies", [])
    presets = data.get("presets", [])
    if not strategies:
        errors.append(f"{path}: missing strategies")
    if not presets:
        errors.append(f"{path}: missing presets")
    for strategy in strategies:
        for field in ("id", "label", "objective", "bestFor", "flags", "sourceRefs", "desc"):
            if field not in strategy:
                errors.append(f"{path}: strategy missing field {field}: {strategy.get('id', '<unknown>')}")
        for source_ref in strategy.get("sourceRefs", []):
            err = check_ref(str(source_ref))
            if err:
                errors.append(f"{path}: {err}")
    for preset in presets:
        for field in ("id", "label", "worldSize", "tp", "pp", "cp", "ep", "note"):
            if field not in preset:
                errors.append(f"{path}: preset missing field {field}: {preset.get('id', '<unknown>')}")
        dims = [preset.get(key) for key in ("worldSize", "tp", "pp", "cp", "ep")]
        if not all(isinstance(value, int) and value > 0 for value in dims):
            errors.append(f"{path}: preset dimensions must be positive integers: {preset.get('id')}")
            continue
        world_size, tp, pp, cp, ep = dims
        factor = tp * pp * cp * ep
        if world_size % factor != 0:
            errors.append(f"{path}: preset {preset.get('id')} worldSize {world_size} not divisible by TP*PP*CP*EP {factor}")
    return errors


def check_finite_numbers(path: Path, value, trail: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            errors.extend(check_finite_numbers(path, child, f"{trail}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(check_finite_numbers(path, child, f"{trail}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path}: non-finite numeric value at {trail}")
    return errors


def main() -> None:
    errors: list[str] = []
    for graph in (DATA / "graphs").glob("*.json"):
        if graph.name != "index.json":
            errors.extend(check_graph(graph))
    for model in (DATA / "models").glob("*.json"):
        if model.name != "index.json":
            errors.extend(check_model(model))
    parallelism = DATA / "parallelism" / "strategies.json"
    if parallelism.is_file():
        errors.extend(check_parallelism(parallelism))
    else:
        errors.append("missing parallelism/strategies.json")
    if not (DATA / "guide.md").is_file():
        errors.append("missing guide.md")
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("manifest verification passed")


if __name__ == "__main__":
    main()
