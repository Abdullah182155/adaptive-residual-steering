import json


def test_build_benchmark_summary_filters_by_run_and_writes_tables(tmp_path):
    from scripts.research_notebook_analysis import build_benchmark_summary

    record = {
        "suite": "gsm8k",
        "four_arm_complete": True,
        "provenance": {
            "evaluation_code_commit": "code-revision",
            "artifact_fingerprint": "artifact-123",
        },
        "shots": {
            "2": {
                "baseline": {"n": 2, "accuracy_pct": 50.0},
                "steer_only": {"n": 2, "accuracy_pct": 50.0},
                "static_bank": {"n": 2, "accuracy_pct": 100.0},
                "router": {"n": 2, "accuracy_pct": 50.0},
                "paired_vs_baseline": {
                    "steer_only": {"wins": 1, "losses": 1, "ties": 0, "delta_pp": 0.0},
                    "static_bank": {"wins": 1, "losses": 0, "ties": 1, "delta_pp": 50.0},
                    "router": {"wins": 0, "losses": 1, "ties": 1, "delta_pp": -50.0},
                },
                "router_route_usage": {
                    "route_counts": {"12": 2},
                    "k_counts": {"1": 2},
                    "layer_counts": {"12": 2},
                },
            }
        },
    }
    (tmp_path / "session3_gsm8k_2_seed42.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    output_dir = tmp_path / "analysis"
    summary = build_benchmark_summary(
        [tmp_path],
        output_dir,
        artifact_fingerprint="artifact-123",
        evaluation_code_commit="code-revision",
    )

    assert summary["record_count"] == 1
    assert len(summary["arm_metrics"]) == 4
    assert len(summary["paired_comparisons"]) == 3
    assert {row["arm"] for row in summary["arm_metrics"]} == {
        "baseline", "steer_only", "static_bank", "router"
    }
    for name in (
        "benchmark_metrics_summary.json",
        "benchmark_metrics_summary.csv",
        "benchmark_paired_comparisons.csv",
        "benchmark_route_usage.csv",
    ):
        assert (output_dir / name).is_file()
