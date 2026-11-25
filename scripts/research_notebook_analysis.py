"""Analysis helpers for saved ARS research-notebook artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

from scripts.research_notebook_utils import atomic_write_json, merge_result_bundle, pack_files, sha256_file


RESULT_BUNDLES = (
    "ars_gsm8k_results.zip",
    "ars_math_ood_results.zip",
    "ars_mcq_results.zip",
)


def merge_available_results(input_roots: Iterable[str | Path], workspace: str | Path) -> dict[str, Any]:
    """Merge attached benchmark bundles into one flat, conflict-checked folder."""
    from scripts.research_notebook_utils import locate_named_file

    destination = Path(workspace)
    destination.mkdir(parents=True, exist_ok=True)
    bundles = {}
    for name in RESULT_BUNDLES:
        path = locate_named_file(input_roots, name)
        if path is not None:
            merge_result_bundle(path, destination)
            bundles[name] = {"path": str(path), "sha256": sha256_file(path)}
    return {"workspace": str(destination), "bundles": bundles}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def collect_saved_records(roots: Iterable[str | Path]) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            if path.name.startswith("mcq_") or path.name.startswith("session3_"):
                record = _read_json(path)
                if not record or not isinstance(record.get("shots"), dict):
                    continue
                # Shot slices can share a filename, dataset, and seed. Keep each
                # distinct file while deduplicating an identical path if roots overlap.
                identity = str(path.resolve())
                if identity not in seen:
                    records.append((path, record))
                    seen.add(identity)
    return records


def _as_percent(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result * 100.0 if 0.0 <= result <= 1.0 else result


def _arm_accuracy(shot_data: dict[str, Any], arm: str) -> float | None:
    values = shot_data.get(arm)
    if not isinstance(values, dict):
        return None
    for key in ("accuracy_pct", "lenient_accuracy_pct", "lenient_accuracy", "accuracy"):
        if key in values:
            return _as_percent(values[key])
    return None


def _paired_data(shot_data: dict[str, Any], arm: str) -> dict[str, Any] | None:
    if isinstance(shot_data.get("paired_vs_baseline"), dict):
        value = shot_data["paired_vs_baseline"].get(arm)
        return value if isinstance(value, dict) else None
    value = shot_data.get(f"paired_{arm}")
    return value if isinstance(value, dict) else None


def _comparison_data(shot_data: dict[str, Any], reference: str, treatment: str) -> dict[str, Any] | None:
    if reference == "baseline":
        return _paired_data(shot_data, treatment)
    keys = {
        ("steer_only", "static_bank"): ("static_bank_vs_fixed_v2", "static_bank_vs_steer_only"),
        ("steer_only", "router"): ("router_vs_fixed_v2", "router_vs_steer_only"),
        ("static_bank", "router"): ("router_vs_static_bank",),
    }
    for key in keys.get((reference, treatment), ()):
        value = shot_data.get(key)
        if isinstance(value, dict):
            return value
    return None


def _route_usage(shot_data: dict[str, Any]) -> dict[str, Any]:
    value = shot_data.get("router_route_usage", {})
    return value if isinstance(value, dict) else {}


def _record_rows(records: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for path, record in records:
        suite = str(record.get("suite", record.get("dataset", path.stem)))
        for shot, values in record.get("shots", {}).items():
            if not isinstance(values, dict):
                continue
            for arm in ("baseline", "steer_only", "static_bank", "router"):
                accuracy = _arm_accuracy(values, arm)
                if accuracy is not None:
                    arm_data = values.get(arm, {})
                    paired = _comparison_data(values, "baseline", arm) if arm != "baseline" else None
                    route_usage = _route_usage(values)
                    rows.append({
                        "source_record": str(path),
                        "suite": suite,
                        "shot": str(shot),
                        "arm": arm,
                        "n": arm_data.get("n", record.get("test_n", record.get("test_n_requested"))),
                        "accuracy_pct": accuracy,
                        "strict_accuracy_pct": _as_percent(arm_data.get("strict_accuracy", arm_data.get("canonical_accuracy", arm_data.get("accuracy_pct")))),
                        "marker_accuracy_pct": _as_percent(arm_data.get("marker_accuracy", arm_data.get("accuracy_pct"))),
                        "lenient_accuracy_pct": _as_percent(arm_data.get("lenient_accuracy", arm_data.get("accuracy_pct"))),
                        "extraction_rate_pct": _as_percent(arm_data.get("extraction_rate_pct", arm_data.get("format_rate"))),
                        "n_timed": arm_data.get("n_timed"),
                        "timing_coverage_rate_pct": _as_percent(arm_data.get("timing_coverage_rate")),
                        "mean_generation_latency_seconds": arm_data.get("mean_generation_latency_seconds"),
                        "median_generation_latency_seconds": arm_data.get("median_generation_latency_seconds"),
                        "p90_generation_latency_seconds": arm_data.get("p90_generation_latency_seconds"),
                        "questions_per_second": arm_data.get("questions_per_second"),
                        "output_tokens_per_second": arm_data.get("output_tokens_per_second"),
                        "mean_new_tokens": arm_data.get("mean_new_tokens"),
                        "median_new_tokens": arm_data.get("median_new_tokens"),
                        "mean_prompt_tokens": arm_data.get("mean_prompt_tokens"),
                        "stop_reason_counts": arm_data.get("stop_reason_counts"),
                        "total_output_tokens": arm_data.get("total_output_tokens"),
                        "timed_output_tokens": arm_data.get("timed_output_tokens"),
                        "total_generation_seconds": arm_data.get("total_generation_seconds"),
                        "max_new_tokens_rate_pct": _as_percent(arm_data.get("max_new_tokens_rate")),
                        "guard_trip_rate_pct": _as_percent(arm_data.get("guard_trip_rate")),
                        "nonfinite_count": arm_data.get("nonfinite_count"),
                        "equation_coverage_rate_pct": _as_percent(arm_data.get("equation_coverage_rate")),
                        "mean_eaa_when_equations_present": arm_data.get("mean_eaa_when_equations_present"),
                        "clean_arithmetic_rate_pct": _as_percent(arm_data.get("clean_arithmetic_rate")),
                        "paired_wins_vs_baseline": paired.get("wins", paired.get("fixed_wins")) if paired else None,
                        "paired_losses_vs_baseline": paired.get("losses", paired.get("fixed_losses")) if paired else None,
                        "paired_ties_vs_baseline": paired.get("ties") if paired else None,
                        "paired_delta_pp_vs_baseline": paired.get("delta_pp") if paired else None,
                        "paired_ci95_pp_vs_baseline": paired.get("bootstrap_ci95_pp", paired.get("ci95_pp")) if paired else None,
                        "mcnemar_p_value_vs_baseline": paired.get("mcnemar_p_value") if paired else None,
                        "router_unique_routes": route_usage.get("unique_routes"),
                        "router_route_entropy": route_usage.get("route_entropy"),
                        "router_most_used_route_share": route_usage.get("most_used_route_share"),
                        "router_k_counts": route_usage.get("k_counts"),
                        "router_layer_counts": route_usage.get("layer_counts"),
                        "device_name": record.get("runtime_metadata", {}).get("device_name"),
                    })
    return rows


def build_benchmark_summary(
    roots: Iterable[str | Path],
    output_dir: str | Path,
    *,
    artifact_sha256: dict[str, Any] | None = None,
    artifact_fingerprint: str | None = None,
    evaluation_code_commit: str | None = None,
) -> dict[str, Any]:
    """Save the arm-level quality/performance table for one exact evaluation."""
    root_values = [Path(item) for item in roots]
    records = collect_saved_records(root_values)
    if artifact_sha256 is not None or artifact_fingerprint is not None:
        records = [
            item for item in records
            if (
                artifact_sha256 is not None
                and item[1].get("artifact_sha256") == artifact_sha256
            ) or (
                artifact_fingerprint is not None
                and item[1].get("provenance", {}).get("artifact_fingerprint") == artifact_fingerprint
            )
        ]
    if evaluation_code_commit is not None:
        records = [
            item for item in records
            if item[1].get("provenance", {}).get("evaluation_code_commit") == evaluation_code_commit
        ]
    summary = _summary_payload(records, root_values)
    definitions = {
        "generation_latency": "Synchronized wall time for one generate_batch call; includes input tokenization, generation, stopping, and decoding. One unscored warm-up call is run per arm and shot setting.",
        "questions_per_second": "Number of measured examples divided by the sum of their generation latencies; excludes dataset loading, scoring, and file writing.",
        "output_tokens_per_second": "Output tokens from timed generations divided by the sum of their measured generation latencies. Legacy rows without timing are excluded.",
        "mean_prompt_tokens": "Average input prompt width in tokenizer tokens for evaluated examples.",
        "max_new_tokens_rate_pct": "Fraction of generations that ended at the configured new-token cap; a higher value can indicate truncation.",
        "equation_quality": "Heuristic arithmetic-chain audit; equation coverage is shown to contextualize EAA and clean arithmetic rate.",
    }
    payload = {
        "schema": "ars_benchmark_quality_performance_v1",
        "source_roots": [str(root) for root in root_values],
        "record_count": len(records),
        "artifact_sha256_filter": artifact_sha256,
        "artifact_fingerprint_filter": artifact_fingerprint,
        "evaluation_code_commit_filter": evaluation_code_commit,
        "measure_definitions": definitions,
        "arm_metrics": summary["accuracy_rows"],
        "paired_comparisons": summary["paired_rows"],
        "route_usage": summary["route_rows"],
        "coverage": summary["coverage"],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "benchmark_metrics_summary.json", payload)
    _write_csv(output / "benchmark_metrics_summary.csv", payload["arm_metrics"])
    _write_csv(output / "benchmark_paired_comparisons.csv", payload["paired_comparisons"])
    _write_csv(output / "benchmark_route_usage.csv", payload["route_usage"])
    return payload


def _summary_payload(records: list[tuple[Path, dict[str, Any]]], roots: list[Path]) -> dict[str, Any]:
    accuracy_rows = _record_rows(records)
    paired_rows = []
    route_rows = []
    for path, record in records:
        suite = str(record.get("suite", record.get("dataset", path.stem)))
        for shot, values in record.get("shots", {}).items():
            if not isinstance(values, dict):
                continue
            comparisons = (
                ("baseline", "steer_only"), ("baseline", "static_bank"),
                ("baseline", "router"), ("steer_only", "static_bank"),
                ("steer_only", "router"), ("static_bank", "router"),
            )
            for reference, arm in comparisons:
                paired = _comparison_data(values, reference, arm)
                if paired:
                    paired_rows.append({
                        "source_record": str(path),
                        "suite": suite,
                        "shot": str(shot),
                        "reference_arm": reference,
                        "arm": arm,
                        "comparison": f"{arm}_vs_{reference}",
                        "n": paired.get("n"),
                        "wins": paired.get("wins", paired.get("fixed_wins")),
                        "losses": paired.get("losses", paired.get("fixed_losses")),
                        "ties": paired.get("ties"),
                        "delta_pp": paired.get("delta_pp"),
                        "ci95_pp": paired.get("bootstrap_ci95_pp", paired.get("ci95_pp")),
                        "mcnemar_p_value": paired.get("mcnemar_p_value"),
                    })
            usage = _route_usage(values)
            counts = usage.get("route_counts", {})
            if any((counts, usage.get("k_counts"), usage.get("layer_counts"), usage.get("alpha_counts"))):
                total = sum(int(value) for value in counts.values())
                probabilities = [int(value) / total for value in counts.values() if total]
                route_rows.append({
                    "source_record": str(path),
                    "suite": suite,
                    "shot": str(shot),
                    "route_counts": counts,
                    "k_counts": usage.get("k_counts", {}),
                    "layer_counts": usage.get("layer_counts", {}),
                    "alpha_counts": usage.get("alpha_counts", {}),
                    "unique_routes": int(usage.get("unique_routes", len(counts))),
                    "route_entropy": -sum(p * math.log(p) for p in probabilities) if probabilities else 0.0,
                    "most_used_route_share": float(usage.get("most_used_route_share", max(probabilities, default=0.0))),
                })
    return {
        "schema": "ars_saved_experiment_summary_v1",
        "source_roots": [str(root) for root in roots],
        "record_count": len(records),
        "accuracy_rows": accuracy_rows,
        "paired_rows": paired_rows,
        "route_rows": route_rows,
        "coverage": [
            {"record": str(path), "suite": record.get("suite", record.get("dataset")),
             "shots": sorted(record.get("shots", {}).keys(), key=str),
             "complete_four_arm": bool(record.get("four_arm_complete", record.get("evaluated_arms") == ["baseline", "steer_only", "static_bank", "router"]))}
            for path, record in records
        ],
    }


def build_router_diagnostics(roots: Iterable[str | Path], output_dir: str | Path) -> dict[str, Any]:
    """Write compact, copyable route-use and paired-accuracy diagnostics."""
    root_values = [Path(item) for item in roots]
    records = collect_saved_records(root_values)
    summary = _summary_payload(records, root_values)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = atomic_write_json(output / "router_diagnostics.json", summary)
    _write_csv(output / "arm_accuracy.csv", summary["accuracy_rows"])
    _write_csv(output / "paired_comparisons.csv", summary["paired_rows"])
    _write_csv(output / "router_usage.csv", summary["route_rows"])
    print({"records": summary["record_count"], "conditions": len(summary["accuracy_rows"]) // 4,
           "diagnostics": str(json_path), "paired_rows": len(summary["paired_rows"])})
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _collect_predictions(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    filename_pattern = re.compile(r"(?P<suite>.+)_(?P<shot>\d+)shot_(?P<arm>.+)\.jsonl$")
    for root in roots:
        for path in root.rglob("*.jsonl"):
            if path.parent.name not in {"generations", "predictions"}:
                continue
            match = filename_pattern.search(path.name)
            if not match:
                continue
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    arm = str(item.get("arm", match.group("arm")))
                    key = (match.group("suite"), match.group("shot"), int(item.get("source_index", item.get("idx", line_number))), arm)
                    if key in seen:
                        continue
                    seen.add(key)
                    correct = item.get("correct", item.get("lenient"))
                    if correct is None:
                        continue
                    rows.append({
                        "suite": key[0], "shot": key[1], "source_index": key[2],
                        "arm": arm, "correct": bool(correct),
                        "n_new_tokens": item.get("n_new_tokens"),
                        "prompt_tokens": item.get("prompt_tokens"),
                        "generation_latency_seconds": item.get("generation_latency_seconds"),
                        "stop_reason": item.get("stop_reason"),
                        "selected_layers": item.get("selected_layers", []),
                        "selected_alpha": item.get("selected_alpha"),
                    })
    return rows


def _theme(plt):
    black, white, orange = "#000000", "#FFFFFF", "#F28C28"
    plt.rcParams.update({
        "figure.facecolor": black, "axes.facecolor": black,
        "axes.edgecolor": white, "axes.labelcolor": white,
        "text.color": white, "xtick.color": white, "ytick.color": white,
        "savefig.facecolor": black, "font.size": 10,
    })
    return black, white, orange


def create_paper_assets(roots: Iterable[str | Path], output_dir: str | Path) -> dict[str, Any]:
    """Create shareable dark-theme plots, data tables, and one zip bundle."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    root_values = [Path(item) for item in roots]
    records = collect_saved_records(root_values)
    summary = _summary_payload(records, root_values)
    predictions = _collect_predictions(root_values)
    output = Path(output_dir)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    if figures.exists():
        shutil.rmtree(figures)
    for stale_name in (
        "arm_accuracy.csv", "paired_comparisons.csv", "router_usage.csv",
        "per_question_predictions.csv", "paper_results_summary.json", "figure_manifest.json",
    ):
        stale_path = output / stale_name
        if stale_path.exists():
            stale_path.unlink()
    figures.mkdir(parents=True, exist_ok=True)
    black, white, orange = _theme(plt)
    accents = [white, orange, white, orange]
    generated: list[str] = []

    def save(name: str, fig) -> None:
        fig.tight_layout()
        destination = figures / name
        fig.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(fig)
        generated.append(name)

    accuracy = summary["accuracy_rows"]
    if accuracy:
        groups = sorted({(row["suite"], row["shot"]) for row in accuracy})
        arms = ("baseline", "steer_only", "static_bank", "router")
        width = 0.18
        x = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(groups)), 5))
        for index, arm in enumerate(arms):
            values = [next((row["accuracy_pct"] for row in accuracy if (row["suite"], row["shot"], row["arm"]) == (suite, shot, arm)), np.nan) for suite, shot in groups]
            bars = ax.bar(x + (index - 1.5) * width, values, width, label=arm, color=accents[index], edgecolor=white)
            if arm in {"static_bank", "router"}:
                for bar in bars:
                    bar.set_hatch("//")
        ax.set_xticks(x, [f"{suite}\n{shot}-shot" for suite, shot in groups])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Four-arm accuracy by dataset and shot count")
        ax.legend(frameon=False, ncols=2)
        ax.grid(axis="y", color=white, alpha=0.15)
        save("01_accuracy_by_condition.png", fig)
        _write_csv(output / "arm_accuracy.csv", accuracy)

        performance_rows = [
            row for row in accuracy
            if row.get("median_generation_latency_seconds") is not None
            and row.get("output_tokens_per_second") is not None
        ]
        if performance_rows:
            performance_groups = sorted({(row["suite"], row["shot"]) for row in performance_rows})
            width = 0.18
            x = np.arange(len(performance_groups))
            fig, axes = plt.subplots(1, 2, figsize=(max(13, 2.0 * len(performance_groups)), 5))
            for index, arm in enumerate(arms):
                latency_values = [
                    next((float(row["median_generation_latency_seconds"]) for row in performance_rows
                          if (row["suite"], row["shot"], row["arm"]) == (suite, shot, arm)), np.nan)
                    for suite, shot in performance_groups
                ]
                throughput_values = [
                    next((float(row["output_tokens_per_second"]) for row in performance_rows
                          if (row["suite"], row["shot"], row["arm"]) == (suite, shot, arm)), np.nan)
                    for suite, shot in performance_groups
                ]
                offset = (index - 1.5) * width
                color = orange if index % 2 else white
                latency_bars = axes[0].bar(x + offset, latency_values, width, label=arm, color=color, edgecolor=white)
                throughput_bars = axes[1].bar(x + offset, throughput_values, width, label=arm, color=color, edgecolor=white)
                if arm in {"static_bank", "router"}:
                    for bar in (*latency_bars, *throughput_bars):
                        bar.set_hatch("//")
            labels = [f"{suite}\n{shot}-shot" for suite, shot in performance_groups]
            for axis in axes:
                axis.set_xticks(x, labels)
                axis.grid(axis="y", color=white, alpha=0.15)
            axes[0].set_ylabel("Median generation latency (seconds)")
            axes[0].set_title("Latency per example")
            axes[1].set_ylabel("Generated output tokens per second")
            axes[1].set_title("Output throughput")
            axes[0].legend(frameon=False, ncols=2)
            save("12_latency_and_throughput_by_condition.png", fig)

    paired = summary["paired_rows"]
    if paired:
        valid = [row for row in paired if row.get("delta_pp") is not None and row.get("ci95_pp")]
        if valid:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(valid))))
            y = np.arange(len(valid))
            deltas = np.array([float(row["delta_pp"]) for row in valid])
            lows = np.array([float(row["ci95_pp"][0]) for row in valid])
            highs = np.array([float(row["ci95_pp"][1]) for row in valid])
            ax.errorbar(deltas, y, xerr=[deltas - lows, highs - deltas], fmt="o", color=orange, ecolor=orange, capsize=3)
            ax.axvline(0, color=white, alpha=0.7)
            ax.set_yticks(y, [f"{row['suite']} {row['shot']}s {row['comparison']}" for row in valid])
            ax.set_xlabel("Paired accuracy delta (percentage points)")
            ax.set_title("Paired gains and 95% intervals")
            ax.grid(axis="x", color=white, alpha=0.15)
            save("02_paired_deltas_and_intervals.png", fig)
        _write_csv(output / "paired_comparisons.csv", paired)
        win_rows = [row for row in paired if row.get("wins") is not None]
        if win_rows:
            fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(win_rows)), 5))
            x = np.arange(len(win_rows)); width = 0.24
            for j, field in enumerate(("wins", "losses", "ties")):
                bars = ax.bar(x + (j - 1) * width, [row.get(field) or 0 for row in win_rows], width,
                              label=field, color=orange if field == "wins" else white,
                              edgecolor=orange if field == "losses" else white)
                if field == "losses":
                    for bar in bars: bar.set_hatch("//")
            ax.set_xticks(x, [f"{r['suite']}\n{r['shot']}s {r['comparison']}" for r in win_rows], rotation=30, ha="right")
            ax.set_ylabel("Questions")
            ax.set_title("Paired wins, losses, and ties")
            ax.legend(frameon=False)
            save("03_paired_wins_losses_ties.png", fig)

    if accuracy:
        groups = sorted({(row["suite"], row["shot"]) for row in accuracy})
        arms = ("baseline", "steer_only", "static_bank", "router")
        matrix = [[next((row["accuracy_pct"] for row in accuracy if (row["suite"], row["shot"], row["arm"]) == (suite, shot, arm)), np.nan) for arm in arms] for suite, shot in groups]
        cmap = LinearSegmentedColormap.from_list("ars_black_orange_white", [black, orange, white])
        fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * len(groups))))
        image = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_yticks(np.arange(len(groups)), [f"{suite} {shot}s" for suite, shot in groups])
        ax.set_xticks(np.arange(len(arms)), arms, rotation=25, ha="right")
        ax.set_title("Accuracy transfer matrix")
        colorbar = fig.colorbar(image, ax=ax); colorbar.set_label("Accuracy (%)", color=white); colorbar.ax.yaxis.set_tick_params(color=white)
        save("04_accuracy_transfer_matrix.png", fig)

    route_rows = summary["route_rows"]
    _write_csv(output / "router_usage.csv", route_rows)
    if route_rows:
        k_totals = Counter()
        route_totals = Counter()
        layer_totals = Counter()
        alpha_totals = Counter()
        diversity = []
        for row in route_rows:
            k_totals.update({str(k): int(v) for k, v in row["k_counts"].items()})
            route_totals.update({str(k): int(v) for k, v in row["route_counts"].items()})
            layer_totals.update({str(k): int(v) for k, v in row["layer_counts"].items()})
            alpha_totals.update({str(k): int(v) for k, v in row["alpha_counts"].items()})
            diversity.append(row)
        for name, counter, title, limit in (
            ("05_router_k_distribution.png", k_totals, "Router active-layer count", 6),
            ("06_most_used_routes.png", route_totals, "Most selected routes", 12),
            ("07_layer_selection_frequency.png", layer_totals, "Layer selection frequency", 14),
            ("08_router_alpha_frequency.png", alpha_totals, "Router intensity selection", 8),
        ):
            top = counter.most_common(limit)
            if not top:
                continue
            fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(top)), 4))
            ax.bar([key for key, _ in top], [value for _, value in top], color=orange, edgecolor=white)
            ax.set_title(title); ax.set_ylabel("Selections"); ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", color=white, alpha=0.15)
            save(name, fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter([row["unique_routes"] for row in diversity], [row["most_used_route_share"] for row in diversity], color=orange, edgecolors=white, s=85)
        for row in diversity:
            ax.annotate(f"{row['suite']} {row['shot']}s", (row["unique_routes"], row["most_used_route_share"]), color=white, fontsize=8)
        ax.set_xlabel("Unique routes"); ax.set_ylabel("Most-used route share")
        ax.set_title("Route diversity and concentration")
        ax.grid(color=white, alpha=0.15)
        save("09_route_diversity_concentration.png", fig)

    if predictions:
        by_arm = defaultdict(list)
        for row in predictions:
            if row.get("n_new_tokens") is not None:
                by_arm[row["arm"]].append(int(row["n_new_tokens"]))
        if by_arm:
            labels = sorted(by_arm)
            fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 4.5))
            bp = ax.boxplot([by_arm[label] for label in labels], labels=labels, patch_artist=True)
            for index, box in enumerate(bp["boxes"]):
                box.set_facecolor(orange if index % 2 else white); box.set_edgecolor(white)
            for element in ("whiskers", "caps", "medians"):
                for line in bp[element]: line.set_color(white)
            ax.set_ylabel("Generated tokens"); ax.set_title("Completion length by arm")
            ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", color=white, alpha=0.15)
            save("10_completion_length_by_arm.png", fig)
        grouped_predictions: dict[tuple[str, str, int], dict[str, bool]] = defaultdict(dict)
        for row in predictions:
            grouped_predictions[(row["suite"], row["shot"], row["source_index"])][row["arm"]] = row["correct"]
        outcome_counts = Counter()
        for values in grouped_predictions.values():
            baseline = values.get("baseline")
            if baseline is None:
                continue
            for arm in ("steer_only", "static_bank", "router"):
                if arm not in values:
                    continue
                key = f"{arm}_{'win' if values[arm] and not baseline else 'loss' if baseline and not values[arm] else 'tie'}"
                outcome_counts[key] += 1
        if outcome_counts:
            labels = list(outcome_counts)
            colors = [orange if label.endswith("win") else white for label in labels]
            fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(labels)), 4))
            ax.bar(labels, [outcome_counts[label] for label in labels], color=colors, edgecolor=white)
            ax.set_ylabel("Questions"); ax.set_title("Per-question benefit, harm, and ties")
            ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", color=white, alpha=0.15)
            save("11_per_question_benefit_harm.png", fig)

    _write_csv(output / "per_question_predictions.csv", predictions)
    summary_path = atomic_write_json(output / "paper_results_summary.json", summary)
    manifest = {
        "schema": "ars_paper_assets_v1",
        "source_records": [str(path) for path, _ in records],
        "prediction_rows": len(predictions),
        "figure_count": len(generated),
        "figures": generated,
        "theme": {"background": black, "text": white, "accent": orange},
        "interpretation_note": "Route diversity is descriptive; evaluate it together with paired accuracy and compute cost.",
    }
    if generated:
        columns = 3
        rows = math.ceil(len(generated) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), squeeze=False)
        for axis, name in zip(axes.flat, generated):
            axis.imshow(plt.imread(figures / name))
            axis.set_title(name, color=white, fontsize=9)
            axis.axis("off")
        for axis in axes.flat[len(generated):]:
            axis.axis("off")
        fig.suptitle("ARS figure overview", color=white)
        fig.tight_layout()
        contact_path = figures / "00_figure_contact_sheet.png"
        fig.savefig(contact_path, dpi=180, bbox_inches="tight", facecolor=black)
        plt.close(fig)
        generated.insert(0, contact_path.name)
        manifest["figures"] = generated
        manifest["figure_count"] = len(generated)
    manifest_path = atomic_write_json(output / "figure_manifest.json", manifest)
    files = {path: str(path.relative_to(output)) for path in output.rglob("*") if path.is_file() and path.name != "ars_paper_assets.zip"}
    bundle = pack_files(files, output / "ars_paper_assets.zip") if files else None
    result = {**manifest, "summary": str(summary_path), "manifest": str(manifest_path), "bundle": str(bundle) if bundle else None}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
