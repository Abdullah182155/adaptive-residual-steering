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
