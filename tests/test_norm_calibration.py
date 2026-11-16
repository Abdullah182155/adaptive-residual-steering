import json
from pathlib import Path
from scripts.calibrate_norms import load_calibration_questions


def test_norm_calibration_reuses_unique_rollout_questions(tmp_path):
    shard0 = tmp_path / "shard0.jsonl"
    shard1 = tmp_path / "shard1.jsonl"
    shard0.write_text(
        json.dumps({"question": "question A"}) + "\n" + json.dumps({"question": "question B"}) + "\n",
        encoding="utf-8",
    )
    shard1.write_text(
        json.dumps({"question": "question B"}) + "\n" + json.dumps({"question": "question C"}) + "\n",
        encoding="utf-8",
    )

    first = load_calibration_questions(2, rollout_files=[str(shard0), str(shard1)], seed=9)
    repeated = load_calibration_questions(2, rollout_files=[str(shard0), str(shard1)], seed=9)

    assert first == repeated
    assert len(first) == len(set(first)) == 2
    assert set(first).issubset({"question A", "question B", "question C"})
