"""
Pair mining for preference training (Step-DPO).

Phase 1.3: Implement pair mining with k=8 samples, frontier filter, edit-distance matching,
divergence localisation, and rationalised-gold fallback.
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from src.data.onpolicy import OnPolicyChain


@dataclass
class PreferencePair:
    """A preference pair (y+, y-) for DPO training."""
    qid: int
    shot: int
    y_plus: OnPolicyChain  # Correct/better chain
    y_minus: OnPolicyChain  # Incorrect/worse chain
    divergence_position: Optional[int]  # Position where they diverge
    pi0_logprob_y_plus: float  # Baseline logprob for y+
    pi0_logprob_y_minus: float  # Baseline logprob for y-


def mine_preference_pairs(
    chains: List[OnPolicyChain],
    k_samples: int = 8,
    temperature: float = 0.8,
    frontier_threshold: float = 0.5,
    edit_distance_threshold: int = 3,
    max_pairs_per_question: int = 10,
    rationalized_gold_fallback_ratio: float = 0.1,
    rng: Optional[random.Random] = None,
) -> List[PreferencePair]:
    """
    Mine preference pairs from on-policy chains.
    
    Strategy:
    1. Group chains by (qid, shot)
    2. For each group, select k samples
    3. Separate into correct (correct=True) and incorrect (correct=False)
    4. Match correct/incorrect pairs with edit distance filter
    5. Localize divergence position
    6. If insufficient pairs, use rationalized gold fallback
    
    Args:
        chains: List of OnPolicyChain from sampling
        k_samples: Number of samples per question/shot
        temperature: Sampling temperature (for documentation)
        frontier_threshold: Frontier score threshold
        edit_distance_threshold: Max edit distance for pairing
        max_pairs_per_question: Max pairs to mine per question
        rationalized_gold_fallback_ratio: Use gold chain as y+ if low diversity
        rng: Random number generator
    
    Returns:
        List of PreferencePair objects
    """
    if rng is None:
        rng = random.Random(42)
    
    # Group by (qid, shot)
    groups: Dict[Tuple[int, int], List[OnPolicyChain]] = {}
    for chain in chains:
        key = (chain.qid, chain.shot)
        if key not in groups:
            groups[key] = []
        groups[key].append(chain)
    
    pairs = []
    
    for (qid, shot), group_chains in groups.items():
        # Shuffle and select k samples
        rng.shuffle(group_chains)
        selected = group_chains[:k_samples]
        
        # Separate correct and incorrect
        correct = [c for c in selected if c.correct]
        incorrect = [c for c in selected if not c.correct]
        
        if not correct or not incorrect:
            # Not enough diversity - skip or use gold fallback
            if len(correct) == 0 and len(incorrect) == 0:
                continue
            # If only one type, use gold fallback for missing type
            continue
        
        # Match pairs with edit distance filter
        for c_chain in correct:
            for i_chain in incorrect:
                # Simple edit distance (number of differing lines)
                diff_lines = 0
                c_lines = c_chain.text.split('\n')
                i_lines = i_chain.text.split('\n')
                min_len = min(len(c_lines), len(i_lines))
                
                for i in range(min_len):
                    if c_lines[i] != i_lines[i]:
                        diff_lines += 1
                    if diff_lines > edit_distance_threshold:
                        break
                
                if diff_lines <= edit_distance_threshold:
                    # Localize divergence position
                    div_pos = _first_diff_line(c_chain.text, i_chain.text)
                    
                    pair = PreferencePair(
                        qid=qid,
                        shot=shot,
                        y_plus=c_chain,
                        y_minus=i_chain,
                        divergence_position=div_pos,
                        pi0_logprob_y_plus=0.0,  # Will be filled by forward pass
                        pi0_logprob_y_minus=0.0,
                    )
                    pairs.append(pair)
                    
                    if len(pairs) >= max_pairs_per_question:
                        break
            if len(pairs) >= max_pairs_per_question:
                break
        
        # Rationalized gold fallback if insufficient pairs
        if len(pairs) < max_pairs_per_question and rng.random() < rationalized_gold_fallback_ratio:
            # Use a correct chain as y+, generate rationalized gold as y-
            # This is a placeholder - actual implementation would need gold chains
            pass
    
    return pairs


def _first_diff_line(good: str, bad: str) -> Optional[int]:
    """Find the first line where good and bad differ."""
    good_lines = good.split('\n')
    bad_lines = bad.split('\n')
    
    min_len = min(len(good_lines), len(bad_lines))
    for i in range(min_len):
        if good_lines[i] != bad_lines[i]:
            return i
    
    return None if len(good_lines) == len(bad_lines) else min_len


def filter_frontier_pairs(
    pairs: List[PreferencePair],
    threshold: float = 0.5,
) -> List[PreferencePair]:
    """Filter pairs based on frontier score (if available)."""
    # Placeholder - would need frontier scores from sampling
    return pairs


def save_pairs(pairs: List[PreferencePair], path: str) -> None:
    """Save preference pairs to JSON."""
    import json
    data = []
    for p in pairs:
        data.append({
            "qid": p.qid,
            "shot": p.shot,
            "y_plus": {
                "sample_idx": p.y_plus.sample_idx,
                "text": p.y_plus.text,
                "correct": p.y_plus.correct,
                "gold": p.y_plus.gold,
            },
            "y_minus": {
                "sample_idx": p.y_minus.sample_idx,
                "text": p.y_minus.text,
                "correct": p.y_minus.correct,
                "gold": p.y_minus.gold,
            },
            "divergence_position": p.divergence_position,
        })
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_pairs(path: str) -> List[PreferencePair]:
    """Load preference pairs from JSON."""
    import json
    with open(path, "r") as f:
        data = json.load(f)
    
    pairs = []
    for item in data:
        y_plus = OnPolicyChain(
            qid=item["qid"],
            sample_idx=item["y_plus"]["sample_idx"],
            shot=item["shot"],
            prompt="",  # Would need to reconstruct
            text=item["y_plus"]["text"],
            pred_strict=None,
            pred_lenient=None,
            gold=item["y_plus"]["gold"],
            correct=item["y_plus"]["correct"],
            n_tokens=0,
            div_line=None,
        )
        y_minus = OnPolicyChain(
            qid=item["qid"],
            sample_idx=item["y_minus"]["sample_idx"],
            shot=item["shot"],
            prompt="",
            text=item["y_minus"]["text"],
            pred_strict=None,
            pred_lenient=None,
            gold=item["y_minus"]["gold"],
            correct=item["y_minus"]["correct"],
            n_tokens=0,
            div_line=None,
        )
        
        pair = PreferencePair(
            qid=item["qid"],
            shot=item["shot"],
            y_plus=y_plus,
            y_minus=y_minus,
            divergence_position=item["divergence_position"],
            pi0_logprob_y_plus=0.0,
            pi0_logprob_y_minus=0.0,
        )
        pairs.append(pair)
    
    return pairs
