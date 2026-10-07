import re
import torch
import torch.nn.functional as F
from typing import Optional, Set

_CONTRAST_MATH_IDS: Optional[Set[int]] = None
_CONTRAST_FORMAT_IDS: Optional[Set[int]] = None
_CONTRAST_DISCOURSE_IDS: Optional[Set[int]] = None

def build_contrast_id_cache(tokenizer):
    global _CONTRAST_MATH_IDS, _CONTRAST_FORMAT_IDS, _CONTRAST_DISCOURSE_IDS
    if _CONTRAST_MATH_IDS is not None:
        return

    math_re = re.compile(r"^[\d\+\-\*\/\=\$\%\.]+$")
    format_re = re.compile(r"^(Question|Answer|#{1,4}|\n|<\|endoftext\|>|:)$")

    math_ids_set = set()
    format_ids_set = set()
    vocab = tokenizer.get_vocab()
    for tok_str, tok_id in vocab.items():
        clean = tok_str.lstrip("Ġ").lstrip("Ċ").lstrip("▁").lstrip("#").strip()
        if not clean:
            continue
        if math_re.match(clean):
            math_ids_set.add(tok_id)
        elif format_re.match(clean):
            format_ids_set.add(tok_id)

    discourse_re = re.compile(
        r"^(therefore|thus|hence|because|since|result|gives"
        r"|remaining|left|more|fewer|per|each|every|total|sum"
        r"|solve|find|determine|equals|answer)$",
        re.IGNORECASE,
    )
    discourse_ids_set = set()
    for tok_str, tok_id in vocab.items():
        clean = tok_str.lstrip("Ġ").lstrip("Ċ").lstrip("▁").strip()
        if not clean:
            continue
        if discourse_re.match(clean):
            discourse_ids_set.add(tok_id)

    unk = tokenizer.unk_token_id
    _CONTRAST_MATH_IDS = math_ids_set - {unk}
    _CONTRAST_FORMAT_IDS = format_ids_set - {unk}
    _CONTRAST_DISCOURSE_IDS = discourse_ids_set - {unk}

def compute_contrastive_gate_loss(model, ids: torch.Tensor, labels: torch.Tensor, tokenizer, device: torch.device) -> torch.Tensor:
    """Usefulness targets for gate activations balanced by token category."""
    build_contrast_id_cache(tokenizer)
    math_ids_t = torch.tensor(sorted(_CONTRAST_MATH_IDS), dtype=torch.long, device=device)
    format_ids_t = torch.tensor(sorted(_CONTRAST_FORMAT_IDS), dtype=torch.long, device=device)
    disc_ids_t = (
        torch.tensor(sorted(_CONTRAST_DISCOURSE_IDS), dtype=torch.long, device=device)
        if _CONTRAST_DISCOURSE_IDS else None
    )

    total = torch.tensor(0.0, device=device)
    count = 0

    for gate in model.gates.values():
        alpha = gate._alpha_for_aux
        if alpha is None:
            continue
        a = alpha.squeeze(-1).float()
        B_gate = a.size(0)
        T = min(a.size(1), ids.size(1))

        ids_clamped = ids[:, :T]
        target_positions = labels[:, :T] != -100
        vm = gate._valid_mask
        if vm is not None and vm.shape[1] >= T:
            vm = vm[:, :T]
        else:
            vm = None

        for b in range(B_gate):
            b_ids = ids_clamped[min(b, ids_clamped.size(0) - 1)]
            a_b = a[b, :T]
            valid_b = vm[b] if vm is not None else torch.ones_like(a_b, dtype=torch.bool)
            target_b = target_positions[min(b, target_positions.size(0) - 1)]

            math_mask = torch.isin(b_ids, math_ids_t) & valid_b & target_b
            format_mask = torch.isin(b_ids, format_ids_t) & valid_b
            prompt_mask = valid_b & ~target_b
            other_mask = ~math_mask & ~format_mask & valid_b & target_b

            if disc_ids_t is not None:
                discourse_mask = torch.isin(b_ids, disc_ids_t) & valid_b & target_b
                other_mask = other_mask & ~discourse_mask
            else:
                discourse_mask = torch.zeros_like(math_mask)

            if math_mask.any():
                total = total + (a_b[math_mask] - 0.85).square().mean()
                count += 1
            if format_mask.any():
                total = total + (a_b[format_mask] - 0.05).square().mean()
                count += 1
            if prompt_mask.any():
                total = total + (a_b[prompt_mask] - 0.05).square().mean()
                count += 1
            if discourse_mask.any():
                total = total + (a_b[discourse_mask] - 0.40).square().mean()
                count += 1
            if other_mask.any():
                total = total + (a_b[other_mask] - 0.40).square().mean()
                count += 1
            if math_mask.any() and other_mask.any():
                math_mean = a_b[math_mask].mean()
                non_math = (format_mask | discourse_mask | other_mask)
                other_mean = a_b[non_math].mean()
                margin_loss = F.relu(0.20 - (math_mean - other_mean))
                total = total + margin_loss
                count += 1

    return total / max(1, count)
