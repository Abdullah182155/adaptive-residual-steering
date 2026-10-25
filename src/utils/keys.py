"""Canonical key generation for sample indexing and token label stores across ARS stages."""

from typing import Any, Tuple


def canonical_token_key(split: Any, qid: Any, n_shot: Any = 0) -> str:
    """Universal canonical key string for sample lookup in TokenLabelStore and CF tables.

    Format: '<split>:<clean_qid>:<n_shot>'
    Example: 'train:15:0', 'eval:42:0'
    """
    qid_str = str(qid).strip()
    if qid_str.startswith("train:"):
        qid_str = qid_str[len("train:"):]
    elif qid_str.startswith("eval:"):
        qid_str = qid_str[len("eval:"):]
    elif qid_str.startswith("test:"):
        qid_str = qid_str[len("test:"):]

    split_str = str(split).lower().strip() if split is not None else "train"
    if not split_str or split_str == "none":
        split_str = "train"
    try:
        shot_int = int(n_shot)
    except (TypeError, ValueError):
        shot_int = 0
    return f"{split_str}:{qid_str}:{shot_int}"


def parse_token_key(key: str) -> Tuple[str, str, int]:
    """Parse a canonical key back into (split, clean_qid, n_shot)."""
    parts = str(key).split(":")
    if len(parts) >= 3:
        try:
            return parts[0], parts[1], int(parts[2])
        except ValueError:
            return parts[0], parts[1], 0
    return "train", str(key), 0
