# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility wrapper for the YAML-configured Llama-3 8B FP8 long run."""

from __future__ import annotations

from pathlib import Path

from run_llama import main

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "llama3_8b_long.yaml"


if __name__ == "__main__":
    main(default_config=DEFAULT_CONFIG)
