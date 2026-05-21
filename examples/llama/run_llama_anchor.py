# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility wrapper for the small YAML-configured Llama anchor run."""

from __future__ import annotations

from pathlib import Path

from run_llama import main

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "llama_anchor.yaml"


if __name__ == "__main__":
    main(default_config=DEFAULT_CONFIG)
