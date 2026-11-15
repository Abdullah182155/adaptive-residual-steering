import torch
from types import SimpleNamespace

from src.training.rft_collect import strict_accept, tokenize_rft_example
from src.training.rft_trainer import _plain_nll_sum
from scripts.run_rft_pipeline import parse_args


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __init__(self):
        self.padding_side = "right"

    @staticmethod
    def _encode(text):
        ids = [10 + ord(char) % 70 for char in text]
        offsets = [(idx, idx + 1) for idx in range(len(text))]
        return ids, offsets

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=False,
        truncation=False,
        return_offsets_mapping=False,
    ):
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [self._encode(item) for item in texts]
        max_length = max(len(ids) for ids, _ in encoded)
        input_rows, mask_rows, offset_rows = [], [], []
        for ids, offsets in encoded:
            padding_length = max_length - len(ids)
            pad_ids = [self.pad_token_id] * padding_length
            pad_offsets = [(0, 0)] * padding_length
            if self.padding_side == "left":
                input_rows.append(pad_ids + ids)
                mask_rows.append([0] * padding_length + [1] * len(ids))
                offset_rows.append(pad_offsets + offsets)
            else:
                input_rows.append(ids + pad_ids)
                mask_rows.append([1] * len(ids) + [0] * padding_length)
                offset_rows.append(offsets + pad_offsets)
        result = {"input_ids": input_rows, "attention_mask": mask_rows}
        if return_offsets_mapping:
            result["offset_mapping"] = offset_rows
        if return_tensors == "pt":
            result = {name: torch.tensor(value) for name, value in result.items()}
        elif len(texts) == 1:
            result = {name: value[0] for name, value in result.items()}
        return result


def test_rft_accepts_only_strict_correct_verified_completions():
    good = "Step 1: 2 + 2 = 4.\nFinal answer: 4"
    bad_math = "Step 1: 2 + 2 = 5.\nFinal answer: 4"
    bad_format = "The answer is 4"

    assert strict_accept(good, "What is 2 plus 2?", 4.0) == good
    assert strict_accept(bad_math, "What is 2 plus 2?", 4.0) is None
    assert strict_accept(bad_format, "What is 2 plus 2?", 4.0) is None


def test_rft_tokenizer_targets_only_completion_and_shields_prompt():
    tokenizer = CharacterTokenizer()
    cfg = SimpleNamespace(max_tok_len=512, shield_policy="prompt")
    completion = "Step 1: 2 + 2 = 4.\nFinal answer: 4"

    row = tokenize_rft_example(
        tokenizer,
        "What is 2 plus 2?",
        completion,
        cfg,
        max_length=512,
    )

    assert row is not None
    labels = row["labels"]
    assert int(labels.ne(-100).sum()) == len(completion)
    attended = int(row["attention_mask"].sum())
    prompt_length = attended - len(completion)
    assert int(row["prompt_shield_mask"].sum()) == prompt_length - 1
    assert not row["prompt_shield_mask"][prompt_length - 1]
    assert tokenizer.padding_side == "right"


def test_plain_rft_nll_ignores_prompt_and_padding_labels():
    logits = torch.zeros(1, 4, 5)
    labels = torch.tensor([[-100, -100, 2, 3]])
    loss = _plain_nll_sum(logits, labels)

    expected = 2 * torch.log(torch.tensor(5.0))
    assert torch.allclose(loss, expected)


def test_rft_runner_exposes_explicit_stages_and_two_t4_sharding():
    args = parse_args(["--stage", "collect", "--shard", "1", "--n_shards", "2"])

    assert args.stage == "collect"
    assert args.shard == 1
    assert args.n_shards == 2


def test_rft_stage_question_count_is_optional_for_stage_specific_defaults():
    args = parse_args(["--stage", "gate0"])

    assert args.n_questions is None
    assert not args.gate0_nonblocking
    assert not args.allow_low_yield


def test_smoke_collection_can_continue_with_low_yield_but_records_quality():
    import pytest
    import scripts.run_rft_pipeline as runner

    stats = {"positive_sample_rate": 0.0}
    runner._enforce_collection_yield(stats, allow_low_yield=True)

    assert stats["minimum_yield_passed"] is False
    assert stats["low_yield_allowed"] is True

    with pytest.raises(RuntimeError, match="yield is below 15%"):
        runner._enforce_collection_yield({"positive_sample_rate": 0.0})


def test_gate0_uses_requested_sample_and_allows_diagnostic_smoke_mode(tmp_path, monkeypatch):
    import scripts.run_rft_pipeline as runner

    observed = {}

    def fake_collect(_model, _tokenizer, _rows, _device, **kwargs):
        observed["indices"] = kwargs["indices"]
        return {
            "n": len(kwargs["indices"]),
            "pass_at_k_lenient": 0.4,
            "pass_at_k_delta_vs_greedy_pp": -5.0,
            "greedy_lenient_accuracy": 0.45,
        }

    monkeypatch.setattr(runner, "collect_rft_data", fake_collect)
    stats = runner._gate0(
        object(),
        object(),
        list(range(20)),
        SimpleNamespace(seed=7, max_tok_len=768, shield_policy="prompt"),
        torch.device("cpu"),
        tmp_path,
        k=2,
        temperature=0.7,
        n_questions=6,
        nonblocking=True,
    )

    assert len(observed["indices"]) == 6
    assert stats["quality_gate_passed"] is False
    assert stats["passed"] is True
    assert stats["nonblocking"] is True


def test_gate0_strict_mode_still_fails_when_quality_threshold_is_missed(tmp_path, monkeypatch):
    import pytest
    import scripts.run_rft_pipeline as runner

    monkeypatch.setattr(
        runner,
        "collect_rft_data",
        lambda *_args, **_kwargs: {
            "n": 1,
            "pass_at_k_lenient": 0.5,
            "pass_at_k_delta_vs_greedy_pp": 0.0,
        },
    )

    with pytest.raises(RuntimeError, match="Gate 0 failed"):
        runner._gate0(
            object(),
            object(),
            [0],
            SimpleNamespace(seed=1, max_tok_len=768, shield_policy="prompt"),
            torch.device("cpu"),
            tmp_path,
            k=2,
            temperature=0.7,
            n_questions=1,
        )


def test_equivalence_stage_respects_requested_question_count(tmp_path, monkeypatch):
    import scripts.run_rft_pipeline as runner

    class Split(list):
        def with_format(self, _format):
            return self

    class Model:
        def set_routing_override(self, *_args):
            return None

    rows = [{"question": f"Question {index}", "answer": "1"} for index in range(30)]
    observed = {}
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "build_model", lambda _cfg, _device: (Model(), object()))
    monkeypatch.setattr(
        runner, "gsm8k_to_cot", lambda _tokenizer, _cfg: (Split(rows), Split(rows))
    )

    def fake_equivalence(_model, _tokenizer, selected_rows, _device, _cfg, **kwargs):
        observed["n"] = kwargs["n"]
        return {"n": kwargs["n"], "passed": True}

    monkeypatch.setattr(runner, "equivalence_test", fake_equivalence)
    args = parse_args(
        ["--stage", "equivalence", "--n_questions", "10", "--output_dir", str(tmp_path)]
    )

    result = runner.run(args)

    assert observed["n"] == 10
    assert result["n"] == 10
