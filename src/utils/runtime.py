"""Shared process setup for command-line ARS pipelines."""

from __future__ import annotations

import os

from src.utils.logging_utils import setup_quiet_logging


def configure_pipeline_runtime(*, verbose: bool = False) -> None:
    """Set allocator policy and Hugging Face logging before heavy imports."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    setup_quiet_logging(level="verbose" if verbose else "quiet")
