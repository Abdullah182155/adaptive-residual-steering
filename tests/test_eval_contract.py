from src.evaluation.answers import score_tiers
from src.evaluation.benchmark import mcnemar_exact


def test_score_tiers_distinguishes_strict_marker_and_lenient():
    strict = score_tiers("Final answer: 7", 7.0)
    marker = score_tiers("#### 7", 7.0)
    lenient = score_tiers("The result is 7", 7.0)
    assert strict["strict"] and strict["marker"] and strict["lenient"]
    assert not marker["strict"] and marker["marker"]
    assert not lenient["marker"] and lenient["lenient"]


def test_exact_mcnemar_handles_no_discordance_and_asymmetry():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == 0.0625

def test_build_prompt_with_shield_shields_fewshot_exemplars():
    from src.data.prompts import build_prompt_with_shield
    import torch

    class DummyTokenizer:
        def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
            words = text.split(' ')
            offsets = []
            cur = 0
            for w in words:
                start = text.find(w, cur)
                end = start + len(w)
                offsets.append((start, end))
                cur = end
            T = len(offsets)
            return {
                'input_ids': torch.arange(T).unsqueeze(0),
                'attention_mask': torch.ones(1, T),
                'offset_mapping': [torch.tensor(offsets)],
            }

    tok = DummyTokenizer()
    prompt_0, enc_0, shield_0 = build_prompt_with_shield(tok, 'How many apples?', n_shot=0)
    assert not bool(shield_0.any())

    prompt_2, enc_2, shield_2 = build_prompt_with_shield(tok, 'How many apples?', n_shot=2)
    assert bool(shield_2.any())
    assert not bool(shield_2[-3:].any())

