"""Pluggable arithmetic out-of-distribution dataset loader."""
from datasets import load_dataset


OOD_DATASETS = {
    "svamp": ("ChilleD/SVAMP", None),
    "multiarith": ("ChilleD/MultiArith", None),
    "gsm_plus": ("qintongli/GSM-Plus", None),
}


def load_ood_rows(name: str, limit: int | None = None):
    """Load a normalized question/answer view of one supported OOD benchmark."""
    if name not in OOD_DATASETS:
        raise ValueError(f"unsupported OOD dataset: {name}")
    dataset_name, subset = OOD_DATASETS[name]
    dataset = load_dataset(dataset_name, subset, split="test")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    question_keys = ("question", "Body", "problem")
    answer_keys = ("answer", "Answer", "final_answer")
    rows = []
    for example in dataset:
        question = next((example[key] for key in question_keys if key in example), None)
        answer = next((example[key] for key in answer_keys if key in example), None)
        if question is not None and answer is not None:
            rows.append({"question": str(question), "answer": str(answer).split("####")[-1].strip()})
    return rows
