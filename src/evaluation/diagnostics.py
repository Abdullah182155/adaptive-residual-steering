"""Optional visual diagnostics for ARS training and paired evaluation."""
import json
from pathlib import Path


def plot_training(history, save_to):
    import matplotlib.pyplot as pyplot
    figure, axis = pyplot.subplots()
    axis.plot(history.get("train_nll", []), label="train NLL")
    axis.plot(history.get("validation_nll", []), label="validation NLL")
    axis.set_xlabel("epoch")
    axis.legend()
    figure.savefig(save_to, bbox_inches="tight")
    pyplot.close(figure)


def plot_paired(record, save_to):
    import matplotlib.pyplot as pyplot
    wins, losses = 0, 0
    for shot in record["shots"].values():
        paired = shot.get("paired", {})
        wins += paired.get("fixed_wins", 0)
        losses += paired.get("fixed_losses", 0)
    figure, axis = pyplot.subplots()
    axis.bar(["fixed wins", "fixed losses"], [wins, losses])
    figure.savefig(save_to, bbox_inches="tight")
    pyplot.close(figure)


def show_generations(jsonl_path, k=4, flips_only=True):
    rows = [json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()]
    if flips_only:
        rows = [row for row in rows if row.get("baseline", {}).get("lenient") != row.get("fixed", {}).get("lenient")]
    return rows[:k]


def token_injection_trace(model, tokenizer, prompt, device):
    model.set_injection_trace(True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model(**inputs)
    model.set_injection_trace(False)
    return model.injection_trace
