#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


SECTIONS = [
    ("README.md", "Megatron-LM and Megatron Core"),
    ("docs/get-started/overview.md", "Project Architecture"),
    ("docs/user-guide/parallelism-guide.md", "Parallelism Strategies"),
    ("docs/ultra-scale-playbook/megatron-mapping.md", "Ultra-Scale Playbook Mapping"),
    ("docs/user-guide/features/moe.md", "Mixture of Experts"),
    ("docs/user-guide/features/multi_latent_attention.md", "Multi-Latent Attention"),
    ("docs/user-guide/features/megatron_fsdp.md", "Megatron FSDP"),
    ("docs/user-guide/features/context_parallel.md", "Context Parallelism"),
]


def excerpt(text: str, max_lines: int = 90) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    in_code = False
    in_comment = False
    in_details = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if in_comment:
            if "-->" in line:
                in_comment = False
            continue

        if in_details:
            if "</details>" in lower:
                in_details = False
            continue

        if "<!--" in line:
            if "-->" not in line:
                in_comment = True
            continue

        if "<details" in lower:
            if "</details>" not in lower:
                in_details = True
            continue

        if not in_code and stripped.startswith("<") and stripped.endswith(">"):
            continue

        if len(kept) >= max_lines and not in_code:
            break
        kept.append(line)
        if line.startswith("```"):
            in_code = not in_code
    if in_code:
        kept.append("```")
    return "\n".join(kept).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default="explorer/public/data/guide.md")
    args = parser.parse_args()
    repo_root = Path(args.repo_root)
    chunks = [
        "# Megatron Explorer Guide",
        "",
        "This generated guide links the visual explorer back to Megatron-LM source, docs, and example configs.",
        "",
        "```mermaid",
        "flowchart LR",
        "  YAML[Example YAML] --- Launcher[Launcher]",
        "  Launcher --- Train[Training loop]",
        "  Train --- Core[Megatron Core]",
        "  Core --- Parallel[Parallelism groups]",
        "  Core --- Model[Transformer model]",
        "  Train --- Optim[Optimizer and checkpointing]",
        "```",
    ]
    for rel, title in SECTIONS:
        path = repo_root / rel
        if not path.is_file():
            chunks.extend([f"## {title}", "", f"Source `{rel}` was not found.", ""])
            continue
        chunks.extend([f"## {title}", "", f"_Source: `{rel}`_", "", excerpt(path.read_text(encoding="utf-8")), ""])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
