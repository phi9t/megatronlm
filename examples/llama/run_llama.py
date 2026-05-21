# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run a YAML-configured local Llama training job in Docker."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Sequence

from llama_config import (
    LlamaConfigError,
    UserInfo,
    build_docker_command,
    build_remove_container_command,
    load_config,
    quoted_command,
    resolve_config_path,
    resolve_output_dir,
    with_operational_overrides,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to a validated Llama YAML config.")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    default_config: Path | None = None,
) -> None:
    """Run a YAML-configured Llama training Docker command."""

    args = parse_args(argv)
    try:
        config_path = resolve_config_path(args.config, default_path=default_config)
        config = load_config(config_path)
        config = with_operational_overrides(
            config,
            foreground=args.foreground,
            dry_run=args.dry_run,
            preflight_only=args.preflight_only,
        )
    except LlamaConfigError as exc:
        raise SystemExit(str(exc)) from exc

    repo_root = Path.cwd().resolve()
    output_dir = resolve_output_dir(config.runtime.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    user = UserInfo(
        uid=os.getuid(), gid=os.getgid(), name=os.environ.get("USER", "megatron")
    )
    command = build_docker_command(config, repo_root, user)

    if config.runtime.dry_run:
        print(quoted_command(command))
        return

    subprocess.run(
        build_remove_container_command(config.runtime.container_name),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
