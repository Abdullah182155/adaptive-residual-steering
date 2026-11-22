"""Paired GSM8K evaluation for the supported ARS benchmark arms."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from tqdm import tqdm
from src.evaluation.answers import extract_numeric_answer, score_tiers
from src.evaluation.generation import build_prompts, build_question_feature_masks, generate_batch
from src.evaluation.chain import (
    audit_chain,
    classify_outcome,
    paired_bootstrap_ci,
)


def _gold(value) -> float:
    return float(str(value).replace(",", "").replace("$", "").strip())


def _set_gate_mode(model, enabled: bool) -> None:
    """Enable or disable the per-token gate on the model.

    When disabled the gate is frozen at alpha=1.0 (pass-through, no selectivity).
    When enabled the learned gate is active and applies per-token steering weights.
    """
    if not hasattr(model, "gates") or not model.gates:
        return
    for gate in model.gates.values():
        gate._freeze_gate = not enabled
        if not enabled:
            gate._freeze_gate_value = 1.0


def _configure_arm(model, arm: str, cfg, gate_provider=None) -> None:
    """Set routing, gain, and gate state explicitly for one benchmark arm."""
    if hasattr(model, "set_route_experts"):
        model.set_route_experts(None)
    if hasattr(model, "set_prompt_strength_override"):
        model.set_prompt_strength_override(None)
    if hasattr(model, "set_route_gains"):
        model.set_route_gains(None)
    if hasattr(model, "set_intervention_gains"):
        model.set_intervention_gains(None)
    gate_enabled = arm in {"gate", "full"}
    if hasattr(cfg, "use_gate"):
        cfg.use_gate = gate_enabled
    if hasattr(model, "set_gate_provider"):
        model.set_gate_provider(gate_provider if gate_enabled else None)
    if hasattr(model, "cfg"):
        model.cfg.use_gate = gate_enabled

    if arm == "baseline":
        model.set_routing_override("none", cfg.fixed_layers)
        if hasattr(model, "set_route_selector"):
            model.set_route_selector(None)
        if hasattr(model, "set_steer_scale"):
            model.set_steer_scale(1.0)
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(1.0)
        if hasattr(model, "set_control"):
            model.set_control(None)
        _set_gate_mode(model, enabled=False)
    elif arm == "identity":
        model.set_routing_override("only", cfg.fixed_layers)
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(0.0)
        if hasattr(model, "set_control"):
            model.set_control(None)
        _set_gate_mode(model, enabled=False)
    elif arm in {"fixed", "steer_only"}:
        # Pure Phase 1 steering: pinned weights, static route, unit scale/gain,
        # and no gate provider or route selector can influence the arm.
        model.set_routing_override("only", cfg.fixed_layers)
        if hasattr(model, "set_route_selector"):
            model.set_route_selector(None)
        if hasattr(model, "set_steer_scale"):
            model.set_steer_scale(1.0)
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(1.0)
        if hasattr(model, "set_control"):
            model.set_control(None)
        _set_gate_mode(model, enabled=False)
    elif arm == "static_bank":
        route = list(getattr(cfg, "static_bank_route", ()))
        alpha = float(getattr(cfg, "static_bank_alpha", 1.0))
        model.set_routing_override("only", route)
        if hasattr(model, "set_route_experts"):
            model.set_route_experts([route])
        if hasattr(model, "set_route_selector"):
            model.set_route_selector(None)
        if hasattr(model, "set_steer_scale"):
            model.set_steer_scale(1.0)
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(1.0)
        if hasattr(model, "set_route_gains"):
            model.set_route_gains(alpha)
        if hasattr(model, "set_control"):
            model.set_control(None)
        _set_gate_mode(model, enabled=False)
    elif arm.startswith("steer_gain:"):
        try:
            gain = float(arm.split(":", 1)[1])
        except ValueError as error:
            raise ValueError(f"invalid steering gain arm: {arm}") from error
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("steering gain must be finite and nonnegative")
        model.set_routing_override("only", cfg.fixed_layers)
        if hasattr(model, "set_route_selector"):
            model.set_route_selector(None)
        if hasattr(model, "set_steer_scale"):
            model.set_steer_scale(1.0)
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(1.0)
        if hasattr(model, "set_intervention_gains"):
            model.set_intervention_gains(torch.tensor([gain], dtype=torch.float32))
        _set_gate_mode(model, enabled=False)
    elif arm.startswith("strength_fixed:"):
        try:
            strength = float(arm.split(":", 1)[1])
        except ValueError as error:
            raise ValueError(f"invalid fixed prompt-strength arm: {arm}") from error
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("fixed prompt strength must be finite and nonnegative")
        if not hasattr(model, "set_prompt_strength_override"):
            raise ValueError(
                "strength_fixed arms require the prompt-conditioned steering architecture"
            )
        model.set_routing_override("only", cfg.fixed_layers)
        if hasattr(model, "set_route_selector"):
            model.set_route_selector(None)
        if hasattr(model, "set_steer_scale"):
            model.set_steer_scale(1.0)
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(1.0)
        model.set_prompt_strength_override(strength)
        if hasattr(model, "set_control"):
            model.set_control(None)
        _set_gate_mode(model, enabled=False)
    elif arm == "random":
        model.set_routing_override("only", cfg.fixed_layers)
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(1.0)
        if hasattr(model, "set_control"):
            model.set_control("random")
        _set_gate_mode(model, enabled=False)
    elif arm == "router":
        model.set_routing_override(None)
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(1.0)
        if hasattr(model, "set_control"):
            model.set_control(None)
        _set_gate_mode(model, enabled=False)
    elif arm == "gate":
        model.set_routing_override("only", cfg.fixed_layers)
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(1.0)
        if hasattr(model, "set_control"):
            model.set_control(None)
        _set_gate_mode(model, enabled=True)
    elif arm == "full":
        model.set_routing_override(None)
        if hasattr(model, "set_intervention_gain"):
            model.set_intervention_gain(1.0)
        if hasattr(model, "set_control"):
            model.set_control(None)
        _set_gate_mode(model, enabled=True)


@torch.no_grad()
def _generate(
    model,
    tokenizer,
    question: str,
    n_shot: int,
    device: torch.device,
    max_new_tokens: int,
    policy: str = "prompt",
    route_prefill_fn=None,
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    template_id: int = 0,
    **prompt_kw,
) -> Tuple[str, Dict[str, Any]]:
    prompts, shields = build_prompts(
        tokenizer,
        [question],
        [int(prompt_kw.get("question_idx", 0))],
        n_shot,
        policy=policy,
        template_ids=[template_id],
    )
    router_feature_kwargs = {}
    if route_prefill_fn is not None and getattr(route_prefill_fn, "uses_question_features", False):
        feature_masks = build_question_feature_masks(tokenizer, prompts, [question])
        question_tokens = int(feature_masks[0].sum())
        router_feature_kwargs = {
            "route_feature_masks": feature_masks,
            "route_context_features": torch.tensor(
                [[min(max(float(n_shot), 0.0), 4.0) / 4.0, min(question_tokens, 256) / 256.0]],
                dtype=torch.float32,
            ),
        }
    result = generate_batch(
        model,
        tokenizer,
        prompts,
        device,
        prompt_shields=shields,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_prompt_len=max(getattr(getattr(model, "cfg", None), "max_tok_len", 768), 1024),
        route_prefill_fn=route_prefill_fn,
        **router_feature_kwargs,
    )[0]
    prompt_strengths = {}
    if hasattr(model, "get_prompt_strengths"):
        prompt_strengths = {
            str(layer): float(values.flatten()[0])
            for layer, values in model.get_prompt_strengths().items()
            if values.numel()
        }
    return result.text, {
        "stop_reason": result.stop_reason,
        "n_new_tokens": result.n_new_tokens,
        "prompt_strength_by_layer": prompt_strengths,
    }


def _score(text: str, gold: float) -> Dict[str, bool]:
    return score_tiers(text, gold)


def _summary(results: List[Dict[str, object]]) -> Dict[str, float]:
    total = len(results)
    strict = sum(item["strict"] for item in results)
    marker = sum(item["marker"] for item in results)
    lenient = sum(item["lenient"] for item in results)
    guard_trips = sum(item.get("guard_trip", False) for item in results)
    nonfinite = sum(item.get("nonfinite", False) for item in results)
    return {
        "n": total,
        "strict_accuracy": 100.0 * strict / max(total, 1),
        "canonical_accuracy": 100.0 * strict / max(total, 1),
        "marker_accuracy": 100.0 * marker / max(total, 1),
        "format_rate": sum(item["format_ok"] for item in results) / max(total, 1),
        "lenient_accuracy": 100.0 * lenient / max(total, 1),
        "guard_trip_rate": guard_trips / max(total, 1),
        "nonfinite_count": nonfinite,
    }


def _paired_counts(
    baseline: List[Dict[str, object]], fixed: List[Dict[str, object]]
) -> Dict[str, Any]:
    if len(baseline) != len(fixed):
        raise ValueError(f"paired arms have different row counts: {len(baseline)} vs {len(fixed)}")
    baseline_ids = [item.get("source_index") for item in baseline]
    fixed_ids = [item.get("source_index") for item in fixed]
    if any(value is not None for value in baseline_ids + fixed_ids) and baseline_ids != fixed_ids:
        raise ValueError("paired arms have different source question IDs or ordering")
    wins = losses = ties = 0
    for control, treatment in zip(baseline, fixed):
        if treatment["lenient"] and not control["lenient"]:
            wins += 1
        elif control["lenient"] and not treatment["lenient"]:
            losses += 1
        else:
            ties += 1
    b_flags = [bool(item["lenient"]) for item in baseline]
    f_flags = [bool(item["lenient"]) for item in fixed]
    delta_pp, ci_lo, ci_hi = paired_bootstrap_ci(b_flags, f_flags, b=10000, seed=0)
    return {
        "fixed_wins": wins,
        "fixed_losses": losses,
        "ties": ties,
        "delta_pp": delta_pp,
        "ci95_pp": [ci_lo, ci_hi],
    }


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def run_benchmark(
    model,
    tokenizer,
    rows: Iterable[dict],
    cfg,
    device: torch.device,
    shots: Sequence[int],
    arms: Sequence[str],
    contract: Dict[str, object] | None = None,
    suite: str = "gsm8k",
    router_prefill_fn=None,
    arm_setup=None,
    test_seed: int | None = None,
) -> Dict[str, object]:
    """Run greedy paired baseline and steering evaluations into a serializable record."""
    # Ensure SteerNet dropout is off during generation (SteerNet starts in train mode).
    model.eval()
    rows = list(rows)
    record: Dict[str, object] = {
        "model": cfg.model_name,
        "route": list(cfg.fixed_layers),
        "suite": suite,
        "test_seed": test_seed,
        "test_source_indices": [
            int(row.get("source_index", index)) for index, row in enumerate(rows)
        ],
        "shots": {},
    }
    shield_policy = getattr(cfg, "shield_policy", "context")
    # Keep the configured provider available for gate/full arms after any
    # baseline or pure-steering arm explicitly detaches it.
    gate_provider = getattr(model, "_gate_provider", None)
    for n_shot in shots:
        arm_results: Dict[str, List[Dict[str, object]]] = {}
        for arm in arms:
            supported_arm = arm in {
                "baseline", "fixed", "steer_only", "identity", "random",
                "router", "static_bank", "gate", "full",
            } or arm.startswith(("steer_gain:", "strength_fixed:"))
            if not supported_arm:
                raise ValueError(f"unsupported benchmark arm: {arm}")
            if arm_setup is not None:
                arm_setup(arm)
            _configure_arm(model, arm, cfg, gate_provider)

            results = []
            router_route_counts: Dict[str, int] = {}
            router_k_counts: Dict[str, int] = {}
            router_layer_counts: Dict[str, int] = {}
            router_alpha_counts: Dict[str, int] = {}
            progress = tqdm(
                rows,
                desc=f"{suite} {n_shot}-shot {arm}",
                unit="question",
                leave=True,
                file=sys.stdout,
                disable=False,
                mininterval=0.5,
            )
            for row_idx, row in enumerate(progress):
                if hasattr(model, "set_control") and arm == "random":
                    model.set_control("random", seed=row_idx)

                selected_alpha = (
                    float(arm.split(":", 1)[1]) if arm.startswith("steer_gain:")
                    else float(getattr(cfg, "static_bank_alpha", 1.0)) if arm == "static_bank"
                    else 1.0
                )

                text, meta = _generate(
                    model,
                    tokenizer,
                    row["question"],
                    n_shot,
                    device,
                    cfg.generation_max_new_tokens,
                    policy=shield_policy,
                    question_idx=row_idx,
                    route_prefill_fn=router_prefill_fn if arm in {"router", "full"} else None,
                    temperature=getattr(cfg, "generation_temperature", 0.0),
                    repetition_penalty=getattr(cfg, "generation_repetition_penalty", 1.1),
                    no_repeat_ngram_size=getattr(cfg, "generation_no_repeat_ngram", 6),
                    template_id=row_idx % 3 if getattr(cfg, "diversify_prompts", False) else 0,
                )
                selected_route: List[int] = []
                if arm in {"router", "static_bank"} and hasattr(model, "current_routes"):
                    routes_now = model.current_routes()
                    if routes_now:
                        selected_route = [int(layer) for layer in routes_now[0]]
                    gains = getattr(model, "_route_gains", None)
                    if gains is not None and arm == "router":
                        gains = torch.as_tensor(gains).detach().flatten()
                        if gains.numel():
                            selected_alpha = float(gains[0].cpu())
                    route_key = "+".join(map(str, selected_route)) if selected_route else "k0"
                    router_route_counts[route_key] = router_route_counts.get(route_key, 0) + 1
                    k_key = str(len(selected_route))
                    router_k_counts[k_key] = router_k_counts.get(k_key, 0) + 1
                    for layer in selected_route:
                        key = str(layer)
                        router_layer_counts[key] = router_layer_counts.get(key, 0) + 1
                    alpha_key = str(selected_alpha)
                    router_alpha_counts[alpha_key] = router_alpha_counts.get(alpha_key, 0) + 1
                guard = dict(model.guard_state)
                cap = (
                    None
                    if contract is None
                    else contract.get("envelope", {}).get("relative_norm_cap")
                )
                guard_trip = cap is not None and float(
                    guard.get("rel_max", 0.0)
                ) > float(cap)
                if (
                    (
                        arm in {"fixed", "steer_only", "random", "static_bank", "gate"}
                        or arm.startswith(("steer_gain:", "strength_fixed:"))
                    )
                    and cfg.guard_revert_to_k0
                    and (guard_trip or guard.get("nonfinite", False))
                ):
                    model.set_routing_override("none", cfg.fixed_layers)
                    text, meta = _generate(
                        model,
                        tokenizer,
                        row["question"],
                        n_shot,
                        device,
                        cfg.generation_max_new_tokens,
                        policy=shield_policy,
                        question_idx=row_idx,
                        temperature=getattr(cfg, "generation_temperature", 0.0),
                        repetition_penalty=getattr(cfg, "generation_repetition_penalty", 1.1),
                        no_repeat_ngram_size=getattr(cfg, "generation_no_repeat_ngram", 6),
                        template_id=row_idx % 3 if getattr(cfg, "diversify_prompts", False) else 0,
                    )
                    fallback_layers = (
                        getattr(cfg, "static_bank_route", [])
                        if arm == "static_bank"
                        else cfg.fixed_layers
                    )
                    model.set_routing_override("only", fallback_layers)

                gold_num = _gold(row["answer"])
                scored = _score(text, gold_num)
                pred_marker = extract_numeric_answer(text)
                pred_lenient = extract_numeric_answer(text, lenient=True)
                audit = audit_chain(text, row["question"])
                results.append(
                    {
                        "source_index": int(row.get("source_index", row_idx)),
                        **scored,
                        "text": text,
                        "pred_marker": pred_marker,
                        "pred_lenient": pred_lenient,
                        "stop_reason": meta["stop_reason"],
                        "n_new_tokens": meta["n_new_tokens"],
                        "rel_max": float(guard.get("rel_max", 0.0)),
                        "guard_trip": bool(guard_trip),
                        "nonfinite": bool(guard.get("nonfinite", False)),
                        "car": audit["car"],
                        "eaa": audit["eaa"],
                        "selected_layers": selected_route if arm in {"router", "static_bank"} else None,
                        "selected_k": len(selected_route) if arm in {"router", "static_bank"} else None,
                        "selected_alpha": selected_alpha if (arm in {"router", "static_bank"} or arm.startswith("steer_gain:")) else None,
                        "prompt_strength_by_layer": meta.get("prompt_strength_by_layer", {}),
                    }
                )
                progress.set_postfix(correct=sum(item["lenient"] for item in results))
                sys.stdout.flush()
            arm_results[arm] = results

            # Save individual arm generation file: report/generations/{suite}_{n_shot}shot_{arm}.jsonl
            generation_dir = Path(cfg.report_dir) / "generations"
            generation_dir.mkdir(parents=True, exist_ok=True)
            safe_arm_name = arm.replace(":", "_")
            arm_output_path = generation_dir / f"{suite}_{n_shot}shot_{safe_arm_name}.jsonl"
            arm_records = [
                {
                    "idx": i,
                    "source_index": int(row.get("source_index", i)),
                    "question": row["question"],
                    "gold": row["answer"],
                    **res,
                }
                for i, (row, res) in enumerate(zip(rows, results))
            ]
            arm_output_path.write_text(
                "".join(json.dumps(item) + "\n" for item in arm_records),
                encoding="utf-8",
            )

        shot_record: Dict[str, object] = {
            arm: _summary(values) for arm, values in arm_results.items()
        }
        for arm, values in arm_results.items():
            strengths: Dict[str, list[float]] = {}
            for item in values:
                for layer, value in item.get("prompt_strength_by_layer", {}).items():
                    strengths.setdefault(str(layer), []).append(float(value))
            if strengths:
                shot_record[arm]["prompt_strength_by_layer"] = {
                    layer: {
                        "mean": sum(layer_values) / len(layer_values),
                        "std": (
                            sum(
                                (value - sum(layer_values) / len(layer_values)) ** 2
                                for value in layer_values
                            )
                            / len(layer_values)
                        ) ** 0.5,
                        "min": min(layer_values),
                        "max": max(layer_values),
                    }
                    for layer, layer_values in sorted(strengths.items())
                }
        if "router" in arm_results:
            shot_record["router_route_usage"] = {
                "route_counts": router_route_counts,
                "k_counts": router_k_counts,
                "layer_counts": router_layer_counts,
                "alpha_counts": router_alpha_counts,
            }
        paired_arm = next(
            (name for name in ("steer_only", "fixed") if name in arm_results), None
        )
        if paired_arm is None:
            paired_arm = next((name for name in arms if name.startswith("steer_gain:")), None)
        if "baseline" in arm_results and paired_arm is not None:
            paired_stats = _paired_counts(arm_results["baseline"], arm_results[paired_arm])
            paired_key = (
                "paired_steer_only" if paired_arm == "steer_only"
                else "paired_" + paired_arm.replace(":", "_") if paired_arm.startswith("steer_gain:")
                else "paired"
            )
            shot_record[paired_key] = paired_stats
            shot_record[paired_key]["mcnemar_p_value"] = mcnemar_exact(
                paired_stats["fixed_wins"], paired_stats["fixed_losses"]
            )
            identical = sum(
                baseline["text"] == fixed["text"]
                for baseline, fixed in zip(arm_results["baseline"], arm_results[paired_arm])
            )
            shot_record[paired_key]["identical_text_rate"] = identical / max(len(rows), 1)
            shot_record[paired_key]["null_intervention_warning"] = (
                shot_record[paired_key]["identical_text_rate"] > 0.95
            )
            # Add failure classification taxonomy for discordant questions
            taxonomy_counts: Dict[str, int] = {}
            for row, b_res, f_res in zip(rows, arm_results["baseline"], arm_results[paired_arm]):
                cls_tag = classify_outcome(b_res, f_res, _gold(row["answer"]), question=row["question"])
                taxonomy_counts[cls_tag] = taxonomy_counts.get(cls_tag, 0) + 1
            shot_record[paired_key]["taxonomy"] = taxonomy_counts

            # Save paired generation file for diagnostics and legacy tools: report/generations/{suite}_{n_shot}shot.jsonl
            paired_rows = [
                {
                    "source_index": int(row.get("source_index", i)),
                    "question": row["question"],
                    "gold": row["answer"],
                    "baseline": b,
                    paired_arm: f,
                }
                for i, (row, b, f) in enumerate(
                    zip(rows, arm_results["baseline"], arm_results[paired_arm])
                )
            ]
            paired_output_path = generation_dir / f"{suite}_{n_shot}shot.jsonl"
            paired_output_path.write_text(
                "".join(json.dumps(item) + "\n" for item in paired_rows),
                encoding="utf-8",
            )

        # Every calibrated intensity gets its own paired result against the same
        # baseline rows; do not leave secondary gains with summary-only metrics.
        if "baseline" in arm_results:
            for gain_arm in (name for name in arms if name.startswith("steer_gain:")):
                paired = _paired_counts(arm_results["baseline"], arm_results[gain_arm])
                key = "paired_" + gain_arm.replace(":", "_")
                paired["mcnemar_p_value"] = mcnemar_exact(
                    paired["fixed_wins"], paired["fixed_losses"]
                )
                paired["identical_text_rate"] = sum(
                    base["text"] == steered["text"]
                    for base, steered in zip(arm_results["baseline"], arm_results[gain_arm])
                ) / max(len(rows), 1)
                shot_record[key] = paired
                gain_suffix = gain_arm.replace(":", "_")
                gain_pairs = [
                    {
                        "source_index": int(row.get("source_index", index)),
                        "question": row["question"],
                        "gold": row["answer"],
                        "baseline": baseline,
                        gain_arm: steered,
                    }
                    for index, (row, baseline, steered) in enumerate(
                        zip(rows, arm_results["baseline"], arm_results[gain_arm])
                    )
                ]
                (generation_dir / f"{suite}_{n_shot}shot_{gain_suffix}_paired.jsonl").write_text(
                    "".join(json.dumps(item) + "\n" for item in gain_pairs),
                    encoding="utf-8",
                )

            for strength_arm in (
                name for name in arms if name.startswith("strength_fixed:")
            ):
                paired = _paired_counts(arm_results["baseline"], arm_results[strength_arm])
                safe_name = strength_arm.replace(":", "_").replace(".", "_")
                paired["mcnemar_p_value"] = mcnemar_exact(
                    paired["fixed_wins"], paired["fixed_losses"]
                )
                paired["identical_text_rate"] = sum(
                    base["text"] == steered["text"]
                    for base, steered in zip(arm_results["baseline"], arm_results[strength_arm])
                ) / max(len(rows), 1)
                shot_record[f"paired_{safe_name}"] = paired
                if "steer_only" in arm_results:
                    adaptive = _paired_counts(
                        arm_results["steer_only"], arm_results[strength_arm]
                    )
                    adaptive["mcnemar_p_value"] = mcnemar_exact(
                        adaptive["fixed_wins"], adaptive["fixed_losses"]
                    )
                    shot_record[f"{safe_name}_vs_prompt_adaptive"] = adaptive

        # Paired stats: baseline vs router (when both present)
        for candidate_arm in ("static_bank", "router"):
            if "baseline" in arm_results and candidate_arm in arm_results:
                paired = _paired_counts(arm_results["baseline"], arm_results[candidate_arm])
                paired["mcnemar_p_value"] = mcnemar_exact(
                    paired["fixed_wins"], paired["fixed_losses"]
                )
                identical = sum(
                    baseline["text"] == treatment["text"]
                    for baseline, treatment in zip(arm_results["baseline"], arm_results[candidate_arm])
                ) / max(len(rows), 1)
                paired["identical_text_rate"] = identical
                paired["null_intervention_warning"] = identical > 0.95
                shot_record[f"paired_{candidate_arm}"] = paired
            if "steer_only" in arm_results and candidate_arm in arm_results:
                paired = _paired_counts(arm_results["steer_only"], arm_results[candidate_arm])
                paired["mcnemar_p_value"] = mcnemar_exact(
                    paired["fixed_wins"], paired["fixed_losses"]
                )
                shot_record[f"{candidate_arm}_vs_fixed_v2"] = paired
        if "static_bank" in arm_results and "router" in arm_results:
            paired = _paired_counts(arm_results["static_bank"], arm_results["router"])
            paired["mcnemar_p_value"] = mcnemar_exact(
                paired["fixed_wins"], paired["fixed_losses"]
            )
            paired["identical_text_rate"] = sum(
                static["text"] == routed["text"]
                for static, routed in zip(arm_results["static_bank"], arm_results["router"])
            ) / max(len(rows), 1)
            shot_record["router_vs_static_bank"] = paired

        record["shots"][str(n_shot)] = shot_record
    return record


def write_record(record: Dict[str, object], path: str) -> None:
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def write_report(record: Dict[str, object], report_dir: str) -> Path:
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    suite = record.get("suite", "gsm8k")
    lines = [
        f"# ARS Benchmark - {suite.upper()}",
        "",
        "| Shots | Arm | Strict | Marker | Lenient | Format Rate | Guard trips | Non-finite |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for shots, result in record["shots"].items():
        for arm in (
            "baseline", "identity", "fixed", "steer_only", "strength_fixed:1.0",
            "static_bank", "router", "random",
        ):
            if arm not in result:
                continue
            summary = result[arm]
            strict_acc = summary.get("strict_accuracy", summary.get("canonical_accuracy", 0.0))
            marker_acc = summary.get("marker_accuracy", summary.get("lenient_accuracy", 0.0))
            lenient_acc = summary.get("lenient_accuracy", 0.0)
            fmt_rate = summary.get("format_rate", 0.0) * 100.0
            lines.append(
                f"| {shots} | {arm} | {strict_acc:.2f}% | {marker_acc:.2f}% | "
                f"{lenient_acc:.2f}% | {fmt_rate:.1f}% | {summary['guard_trip_rate']:.3f} | {summary['nonfinite_count']} |"
            )
        paired_key = next(
            (key for key in ("paired_steer_only", "paired") if key in result), None
        )
        if paired_key is not None:
            paired = result[paired_key]
            p_val = paired.get("mcnemar_p_value", 1.0)
            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}"
            ident = paired.get("identical_text_rate", 0.0) * 100.0
            delta_pp = paired.get("delta_pp", 0.0)
            ci = paired.get("ci95_pp", [0.0, 0.0])
            lines.extend(
                [
                    "",
                    f"{shots}-shot paired {('steer_only' if paired_key == 'paired_steer_only' else 'fixed')} vs baseline: "
                    f"{paired['fixed_wins']} wins, {paired['fixed_losses']} losses, {paired['ties']} ties. "
                    f"Delta={delta_pp:+.2f} pp, 95% CI=[{ci[0]:+.2f}, {ci[1]:+.2f}], McNemar p={p_str}, identical text={ident:.1f}%.",
                    "",
                ]
            )
        for key, label in (
            ("paired_static_bank", "static_bank vs baseline"),
            ("static_bank_vs_fixed_v2", "static_bank vs fixed V2"),
            ("paired_router", "router vs baseline"),
            ("router_vs_fixed_v2", "router vs fixed V2"),
            ("router_vs_static_bank", "router vs validation-selected static bank"),
        ):
            if key not in result:
                continue
            paired = result[key]
            ci = paired.get("ci95_pp", [0.0, 0.0])
            p_val = paired.get("mcnemar_p_value", 1.0)
            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}"
            lines.append(
                f"{shots}-shot paired {label}: {paired['fixed_wins']} wins, "
                f"{paired['fixed_losses']} losses, {paired['ties']} ties; "
                f"delta={paired.get('delta_pp', 0.0):+.2f} pp, "
                f"95% CI=[{ci[0]:+.2f}, {ci[1]:+.2f}], McNemar p={p_str}."
            )
        if any(key in result for key in (
            "paired_static_bank", "static_bank_vs_fixed_v2", "paired_router",
            "router_vs_fixed_v2", "router_vs_static_bank"
        )):
            lines.append("")
    report_path = destination / f"REPORT_{suite.upper()}.md"
    content = "\n".join(lines) + "\n"
    report_path.write_text(content, encoding="utf-8")
    (destination / "REPORT.md").write_text(content, encoding="utf-8")
    return report_path
