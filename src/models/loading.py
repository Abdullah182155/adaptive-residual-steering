"""Model construction helpers for the frozen Phi-2 backbone."""

from __future__ import annotations

import os
from typing import Any, Callable, Protocol

import torch
from src.configs.config import RSCConfig


class PretrainedLoader(Protocol):
    """Minimum Hugging Face loader interface used by the model factory."""

    def from_pretrained(self, model_name: str, **kwargs: Any) -> Any: ...


def build_frozen_phi2(
    cfg: RSCConfig,
    device: torch.device,
    *,
    tokenizer_loader: PretrainedLoader,
    config_loader: PretrainedLoader,
    model_loader: PretrainedLoader,
    wrapper_factory: Callable[..., torch.nn.Module],
) -> tuple[torch.nn.Module, Any]:
    """Load Phi-2 and its ARS wrapper with the backbone in evaluation mode.

    Loader dependencies are explicit so callers can retain their existing
    logging setup and tests can substitute lightweight model components.
    """
    attention_backend = os.environ.get("ARS_ATTENTION_IMPLEMENTATION", "").strip().lower()
    if attention_backend and attention_backend not in {"eager", "sdpa"}:
        raise ValueError(
            "ARS_ATTENTION_IMPLEMENTATION must be 'eager', 'sdpa', or unset; "
            f"got {attention_backend!r}"
        )

    tokenizer = tokenizer_loader.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("the tokenizer does not provide an EOS or padding token ID")

    model_config = config_loader.from_pretrained(cfg.model_name, trust_remote_code=True)
    model_config.pad_token_id = tokenizer.pad_token_id
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    load_kwargs = {
        "config": model_config,
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if attention_backend:
        load_kwargs["attn_implementation"] = attention_backend
    print(
        "[model] attention implementation: "
        f"{attention_backend or 'transformers default'}",
        flush=True,
    )

    backbone = model_loader.from_pretrained(cfg.model_name, **load_kwargs).to(device)
    model = wrapper_factory(backbone, cfg, device)
    # Keep adapter dropout disabled outside explicit training loops.
    model.eval()
    return model, tokenizer


def load_frozen_phi2(cfg: RSCConfig, device: torch.device) -> tuple[torch.nn.Module, Any]:
    """Load Phi-2 using the installed Transformers classes and ARS wrapper."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from src.models.wrapper import Phi2WithRSC

    return build_frozen_phi2(
        cfg,
        device,
        tokenizer_loader=AutoTokenizer,
        config_loader=AutoConfig,
        model_loader=AutoModelForCausalLM,
        wrapper_factory=Phi2WithRSC,
    )
