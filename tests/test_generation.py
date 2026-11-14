import torch

from src.evaluation.generation import equivalence_test, generate_batch, generate_texts


class TinyTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token_id = 0
    pad_token = "<pad>"

    def __init__(self):
        self.padding_side = "right"

    def __call__(self, texts, return_tensors="pt", padding=False, truncation=False):
        if isinstance(texts, str):
            texts = [texts]
        rows = [[10 + ord(char) % 70 for char in text] for text in texts]
        width = max(map(len, rows))
        ids, masks = [], []
        for row in rows:
            padding_length = width - len(row)
            if self.padding_side == "left":
                ids.append([self.pad_token_id] * padding_length + row)
                masks.append([0] * padding_length + [1] * len(row))
            else:
                ids.append(row + [self.pad_token_id] * padding_length)
                masks.append([1] * len(row) + [0] * padding_length)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        mapping = {2: "A", 3: "B", 4: "C"}
        return "".join(mapping.get(int(token), "") for token in token_ids)

    def batch_decode(self, batch, skip_special_tokens=True):
        return [self.decode(tokens, skip_special_tokens) for tokens in batch]


class TinyGenerationModel:
    def __init__(self, vary=True):
        self.shield = None
        self.vary = vary

    def set_prompt_shield_mask(self, mask):
        self.shield = mask

    def generate(
        self,
        input_ids,
        attention_mask,
        logits_processor,
        stopping_criteria,
        num_return_sequences,
        **kwargs,
    ):
        sequences = input_ids.repeat_interleave(num_return_sequences, dim=0)
        scripted = (
            [[2, 99], [3, 4, 99]][: sequences.shape[0]]
            if self.vary
            else [[2, 99] for _ in range(sequences.shape[0])]
        )
        for step in range(4):
            scores = torch.zeros(sequences.shape[0], 100)
            for processor in logits_processor:
                scores = processor(sequences, scores)
            next_tokens = []
            for row, tokens in enumerate(scripted):
                if step < len(tokens):
                    next_tokens.append(tokens[step])
                else:
                    next_tokens.append(99)
            next_tokens = torch.tensor(next_tokens).unsqueeze(1)
            # Processors force EOS for any row that has already emitted one.
            forced = scores.argmax(dim=-1).unsqueeze(1)
            already_done = torch.stack(
                [sequences[row, input_ids.shape[1] :].eq(99).any() for row in range(len(scripted))]
            )
            next_tokens[already_done] = forced[already_done]
            sequences = torch.cat([sequences, next_tokens], dim=1)
            if all(criteria(sequences, scores) for criteria in stopping_criteria):
                break
        return sequences


def test_generate_batch_uses_left_padding_and_padded_prompt_width():
    tokenizer = TinyTokenizer()
    model = TinyGenerationModel()

    results = generate_batch(
        model,
        tokenizer,
        ["long", "xy"],
        "cpu",
        max_new_tokens=8,
    )

    assert tokenizer.padding_side == "left"
    assert [result.text for result in results] == ["A", "BC"]
    assert [result.n_new_tokens for result in results] == [2, 3]
    assert [result.stop_reason for result in results] == ["eos", "eos"]
    assert model.shield is None


def test_generate_texts_accepts_multiple_stop_token_ids():
    tokenizer = TinyTokenizer()
    model = TinyGenerationModel()
    results = generate_texts(
        model,
        tokenizer,
        ["one", "two"],
        "cpu",
        stop_token_ids=[98, 99],
    )
    assert results == ["A", "BC"]


def test_sampling_can_return_multiple_sequences_for_one_prompt():
    results = generate_batch(
        TinyGenerationModel(),
        TinyTokenizer(),
        ["one prompt"],
        "cpu",
        num_return_sequences=2,
        temperature=0.7,
    )

    assert [result.text for result in results] == ["A", "BC"]


def test_equivalence_harness_checks_paired_accuracy_and_identity(monkeypatch):
    import src.evaluation.generation as generation

    monkeypatch.setattr(
        generation,
        "build_prompts",
        lambda tokenizer, questions, question_indices, n_shot, policy: (
            ["prompt"] * len(questions),
            [None] * len(questions),
        ),
    )
    result = equivalence_test(
        TinyGenerationModel(vary=False),
        TinyTokenizer(),
        [{"question": "q1", "final_num": "0"}, {"question": "q2", "final_num": "0"}],
        "cpu",
        n=2,
        batch_size=2,
    )

    assert result["passed"]
    assert result["delta_lenient_pp"] == 0
    assert result["string_identity"] == 1
