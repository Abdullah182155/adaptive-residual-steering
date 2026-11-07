"""
Centralized verbosity control for ARS pipelines.

Calling ``setup_quiet_logging()`` before any HuggingFace imports take effect
suppresses the four main sources of noise:

  1. transformers "Loading weights / Materializing param" tqdm bars
  2. datasets download/caching progress bars
  3. The "unauthenticated requests" HF Hub warning
  4. PyTorch / tokenizers minor warnings that clutter training output

Levels:
  quiet   – suppress all HF progress bars and INFO logs (default for scripts)
  normal  – keep WARNING+ from all libs, but silence tqdm weight bars
  verbose – leave everything enabled (useful for debugging load issues)
"""

from __future__ import annotations

import logging
import os
import warnings


def setup_quiet_logging(level: str = "quiet") -> None:
    """Configure library verbosity before heavy imports.

    Parameters
    ----------
    level:
        ``"quiet"``   – suppress HF tqdm bars, INFO, DEBUG (recommended default)
        ``"normal"``  – suppress HF tqdm bars but keep WARNING+
        ``"verbose"`` – no suppression; raw HF output passes through
    """
    if level == "verbose":
        return

    # ------------------------------------------------------------------ #
    # 1. HuggingFace Hub: silence the unauthenticated-token warning and
    #    all INFO-level download chatter.
    # ------------------------------------------------------------------ #
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # kills tqdm in hub
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")     # avoids fork warning

    # ------------------------------------------------------------------ #
    # 2. transformers: "Loading weights / Materializing param" bars come
    #    from transformers.modeling_utils at INFO level and via tqdm.
    #    set_verbosity_error() promotes the threshold to ERROR so none of
    #    those INFO messages are emitted.
    # ------------------------------------------------------------------ #
    try:
        import transformers
        if level == "quiet":
            transformers.logging.set_verbosity_error()
        else:  # normal
            transformers.logging.set_verbosity_warning()
        # Also disable the internal tqdm progress callback
        transformers.logging.disable_progress_bar()
    except Exception:
        pass

    # ------------------------------------------------------------------ #
    # 3. datasets: download / preparation progress bars and INFO spam.
    # ------------------------------------------------------------------ #
    try:
        import datasets
        if level == "quiet":
            datasets.logging.set_verbosity_error()
        else:
            datasets.logging.set_verbosity_warning()
        datasets.logging.disable_progress_bar()
    except Exception:
        pass

    # ------------------------------------------------------------------ #
    # 4. Standard Python logging: push third-party loggers to WARNING+
    #    so stray getLogger("transformers.xxx").info(...) calls are silent.
    # ------------------------------------------------------------------ #
    for noisy_logger in (
        "transformers",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "transformers.tokenization_utils_base",
        "datasets",
        "datasets.load",
        "datasets.arrow_dataset",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "filelock",
        "urllib3",
    ):
        logging.getLogger(noisy_logger).setLevel(
            logging.ERROR if level == "quiet" else logging.WARNING
        )

    # ------------------------------------------------------------------ #
    # 5. Suppress specific Python warnings that commonly appear.
    # ------------------------------------------------------------------ #
    warnings.filterwarnings("ignore", message=".*Pandas requires version.*")
    warnings.filterwarnings("ignore", message=".*torch.utils._pytree._register_pytree_node.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
