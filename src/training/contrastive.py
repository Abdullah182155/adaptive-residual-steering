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
    """Structural margin loss for gate activations — enforces relative ordering
    (math_alpha > discourse_alpha > format_alpha) without hardcoded absolute targets.
    
    Old design used MSE to fixed targets (0.85, 0.05, 0.40) which fought against
    the task loss when the backbone's optimal alpha differed. New design only
    enforces that the gate steers math tokens more than format tokens, and keeps
    format/prompt alpha below a soft ceiling.
    """
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

            if disc_ids_t is not None:
                discourse_mask = torch.isin(b_ids, disc_ids_t) & valid_b & target_b
            else:
                discourse_mask = torch.zeros_like(math_mask)

            # Soft ceiling: format/prompt alpha should stay low (penalty only if > 0.3)
            if format_mask.any():
                total = total + F.relu(a_b[format_mask] - 0.3).square().mean()
                count += 1
            if prompt_mask.any():
                total = total + F.relu(a_b[prompt_mask] - 0.3).square().mean()
                count += 1

            # Margin: math_alpha should exceed non-math_alpha by at least 0.15
            if math_mask.any():
                math_mean = a_b[math_mask].mean()
                non_math = format_mask | discourse_mask
                if non_math.any():
                    non_math_mean = a_b[non_math].mean()
                    total = total + F.relu(0.15 - (math_mean - non_math_mean))
                    count += 1

    return total / max(1, count)
