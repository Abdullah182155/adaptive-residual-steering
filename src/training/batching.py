"""Batch extraction and padding-safe tensor preparation for training."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def extract_batch_rows(data: Any, offset: int, batch_size: int) -> list[dict]:
    """Return up to ``batch_size`` ordered rows from a list or dataset split."""
    if isinstance(data, list):
        return data[offset : offset + batch_size]
    end = min(offset + batch_size, len(data))
    return [data[index] for index in range(offset, end)]


def trim_to_attended_tokens(
    batch: dict[str, torch.Tensor],
    extras: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Remove all-padding columns and apply the same slice to aligned masks."""
    extras = extras or {}
    attention_mask = batch["attention_mask"].bool()
    live_columns = attention_mask.any(dim=0).nonzero(as_tuple=False).flatten()
    if live_columns.numel() == 0:
        raise ValueError("batch contains no attended tokens")
    start = int(live_columns.min())
    stop = int(live_columns.max()) + 1
    trimmed_batch = {name: value[:, start:stop] for name, value in batch.items()}
    trimmed_extras = {
        name: value[:, start:stop]
        for name, value in extras.items()
        if value.ndim >= 2
        and (
            value.shape[1] == attention_mask.shape[1]
            or value.shape[-1] == attention_mask.shape[-1]
        )
    }
    return trimmed_batch, trimmed_extras


def prepare_batch(
    rows: Sequence[dict[str, torch.Tensor]], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Stack model inputs and auxiliary masks, then trim shared padding."""
    batch = {
        name: torch.stack([row[name] for row in rows]).to(device)
        for name in ("input_ids", "attention_mask", "labels")
        if name in rows[0]
    }
    extras = {
        name: torch.stack([row[name] for row in rows]).to(device)
        for name in ("prompt_shield_mask", "token_types", "calc_mask", "neg_ids")
        if name in rows[0]
    }
    return trim_to_attended_tokens(batch, extras)


def epoch_batches(
    lengths: Sequence[int],
    batch_size: int,
    epoch: int,
    seed: int,
    mega: int = 16,
) -> list[list[int]]:
    """Build length-bucketed batches and shuffle their order deterministically."""
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(lengths), generator=generator).tolist()
    batches = []
    for start in range(0, len(order), batch_size * mega):
        chunk = sorted(order[start : start + batch_size * mega], key=lambda index: lengths[index])
        batches.extend(
            chunk[offset : offset + batch_size]
            for offset in range(0, len(chunk), batch_size)
        )
    return [batches[index] for index in torch.randperm(len(batches), generator=generator).tolist()]
