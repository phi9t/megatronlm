from __future__ import annotations

from pathlib import Path


def _repo_file(repo_root: Path, file: str) -> Path | None:
    root = repo_root.resolve()
    raw = Path(file)
    if raw.is_absolute() or ".." in raw.parts:
        return None
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def resolve_line(repo_root: Path, file: str, symbol: str | None, fallback: int | None = None) -> int | None:
    path = _repo_file(repo_root, file)
    if path is None or not path.is_file() or not symbol:
        return fallback
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return fallback
    candidates = [symbol]
    parts = symbol.split()
    if len(parts) >= 2:
        candidates.append(f"{parts[0]} {parts[1]}")
    for candidate in candidates:
        for line_no, line in enumerate(lines, start=1):
            idx = line.find(candidate)
            if idx < 0:
                continue
            after = line[idx + len(candidate) : idx + len(candidate) + 1]
            if after and (after.isalnum() or after == "_"):
                continue
            return line_no
    return fallback


def ref(repo_root: Path, file: str, symbol: str | None, fallback: int | None = None) -> str:
    if _repo_file(repo_root, file) is None:
        raise ValueError(f"source file must be repo-relative: {file}")
    line = resolve_line(repo_root, file, symbol, fallback)
    return f"{file}:{line}" if line else file
