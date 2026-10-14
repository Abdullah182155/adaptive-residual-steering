import gc
import os
import random
import time
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

from src.configs.config import RSCConfig
from src.evaluation.verifier import epoch_end_accuracy
from src.evaluation.multi_dataset import build_benchmark_prompt, evaluate_sample_answer
from src.evaluation.verifier import generate_response
from src.routing.rloo import compute_router_rloo_loss, per_example_answer_nll
from src.training.losses import (
    LRReducer,
    CheckpointManager,
    GradientMonitor,
    compute_l1_gate,
    compute_antisat_gate,
    compute_gate_diversity_loss,
    compute_invariance_loss,
    compute_pg_advantage_lite,
    compute_pg_gate_loss_from_live_alphas,
    log_gpu_memory,
)
from src.training.contrastive import compute_contrastive_gate_loss
from src.experiments.checkpointing import save_ars_checkpoint, load_ars_checkpoint_strict


def _set_batch_context(model, mask: torch.Tensor, labels: torch.Tensor, shield_mask: Optional[torch.Tensor] = None):
    """Install exact few-shot shielding and route only from the live question prompt."""
    prompt = mask.bool() & labels.eq(-100)
    if shield_mask is not None:
        shield = shield_mask.bool() & mask.bool()
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield)
        prompt = prompt & ~shield
    elif hasattr(model, "set_prompt_shield_mask"):
        model.set_prompt_shield_mask(None)
    model.set_router_context_mask(prompt)


def _small_run_waiver(n_validation_examples: int, cfg: RSCConfig) -> bool:
    """Whether diagnostics are informative-only because the holdout is tiny.

    This never changes losses, routing, or checkpoints.  It only prevents a
    smoke test with an intrinsically unreliable holdout from being mislabeled
    as a failed scientific experiment.
    """
    minimum = int(getattr(cfg, "min_reliable_validation_examples", 16))
    return bool(getattr(cfg, "allow_inconclusive_small_runs", True)) and n_validation_examples < minimum


def _optimizer_has_gradients(parameters) -> bool:
    """Whether this accumulation window actually called backward on an optimizer parameter."""
    return any(parameter.grad is not None for parameter in parameters)


def _answer_aware_loss(logits: torch.Tensor, labels: torch.Tensor, cfg: RSCConfig) -> torch.Tensor:
    """Differentiable completion loss that gives the final answer tail real weight."""
    return per_example_answer_nll(
        logits,
        labels,
        answer_tail_tokens=getattr(cfg, "answer_tail_tokens", 0),
        answer_tail_weight=getattr(cfg, "answer_tail_weight", 1.0),
    ).mean()


def save_steer_weights(
    model, path: str, cfg: Optional[RSCConfig] = None, phase: str = "ars", prefixes=None
):
    """Save ARS tensors with an auditable phase/config/hash sidecar."""
    report = save_ars_checkpoint(model, path, phase=phase, cfg=cfg or model.cfg, prefixes=prefixes)
    print(f"✅ Saved {phase} ARS weights to {path} ({report['tensor_count']} tensors; manifest={report['manifest_path']})")


def load_steer_weights(model, path: str, device: Optional[torch.device] = None):
    """Load RSC-specific weights into model."""
    if device is None:
        device = next(model.parameters()).device
    report = load_ars_checkpoint_strict(model, path, device=device)
    phase = (report.get("manifest") or {}).get("phase", "legacy-unmanifested")
    print(f"✅ Loaded strict ARS checkpoint from {path} (phase={phase}, tensors={report['saved_ars_tensors']})")


def evaluate_loss(
    model, data, cfg: RSCConfig, device: torch.device, progress_desc: Optional[str] = None,
    answer_aware: bool = False,
) -> float:
    """Compute standard or final-answer-aware completion loss with visible progress."""
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_weight = 0

    starts = range(0, len(data), cfg.batch_size)
    # Acceptance runs are deliberately visible: deleting completed bars makes
    # the several-minute isolated-layer audit appear to have frozen.
    iterator = tqdm(starts, desc=progress_desc, unit="batch", leave=True) if progress_desc else starts
    for start in iterator:
        batch_idx = list(range(start, min(start + cfg.batch_size, len(data))))
        ids = torch.stack([data[i]["input_ids"] for i in batch_idx]).to(device)
        mask = torch.stack([data[i]["attention_mask"] for i in batch_idx]).to(device)
        lbls = torch.stack([data[i]["labels"] for i in batch_idx]).to(device)
        shield = None
        if "prompt_shield_mask" in data[batch_idx[0]]:
            shield = torch.stack([data[i]["prompt_shield_mask"] for i in batch_idx]).to(device)

        if hasattr(model, "_attn_mask_cache"):
            model._attn_mask_cache = mask
        _set_batch_context(model, mask, lbls, shield)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(input_ids=ids, attention_mask=mask, labels=lbls, use_cache=False)

        n_tokens = (lbls != -100).sum().item()
        if n_tokens > 0:
            if answer_aware:
                batch_nll = _answer_aware_loss(out.logits, lbls, cfg).item()
                batch_weight = ids.shape[0]
            else:
                batch_nll = out.loss.item()
                batch_weight = n_tokens
            total_nll += batch_nll * batch_weight
            total_weight += batch_weight
            if progress_desc:
                iterator.set_postfix(nll=f"{total_nll / total_weight:.4f}")

    model.train(was_training)
    return total_nll / max(1, total_weight)


def _phase1_zero_shot_generation_probe(
    model,
    tokenizer,
    data,
    cfg: RSCConfig,
    device: torch.device,
    n_samples: int,
    layer: Optional[int] = None,
    gate_value: float = 1.0,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure a Phase-1 route on held-out benchmark-format generation.

    Teacher-forced NLL is useful for optimisation but cannot certify an
    autoregressive intervention.  This deliberately reconstructs the exact
    zero-shot ``Question/Answer`` prompt used by GSM8K evaluation from the
    held-out rows and scores free generation with the production parser.
    It is a validation probe, never a test-set selector.
    """
    requested_rows = list(data)[: min(max(0, int(n_samples)), len(data))]
    rows = [row for row in requested_rows if row.get("question") and row.get("final_num") not in (None, "")]
    if len(rows) != len(requested_rows):
        print(
            "  ⚠️ Phase-1 held-out generation probe skipped malformed rows without raw "
            "question/final answer metadata; this run cannot establish capacity from tokenized prompts."
        )

    was_training = model.training
    old_override = model._routing_override
    old_mode = getattr(model, "_router_inference_mode", "argmax")
    correct = 0
    answer_present = 0
    details = []
    try:
        model.eval()
        model.set_router_inference_mode("argmax")
        if layer is None:
            model.set_routing_override("none")
            model.set_gate_freeze(False)
        else:
            model.set_routing_override("only", [int(layer)])
            model.set_gate_freeze(True, float(gate_value))

        iterator = tqdm(rows, desc=progress_desc, unit="sample", leave=True) if progress_desc else rows
        for row in iterator:
            prompt = build_benchmark_prompt("gsm8k", row["question"], n_shot=0)
            response = generate_response(
                model, tokenizer, prompt, device,
                max_new_tokens=int(getattr(cfg, "phase1_generation_max_new_tokens", 256)),
            )
            score = evaluate_sample_answer("gsm8k", response, row["final_num"])
            correct += int(score["correct"])
            answer_present += int(score["has_answer"])
            details.append({
                "question": row["question"],
                "gold": row["final_num"],
                "response": response,
                "extracted": score["extracted"],
                "correct": bool(score["correct"]),
            })
            if progress_desc:
                iterator.set_postfix(acc=f"{correct / max(1, len(details)):.1%}", valid=len(details))
    finally:
        model.set_gate_freeze(True, 1.0)
        model.set_router_inference_mode(old_mode)
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])
        if was_training:
            model.train()

    total = len(details)
    return {
        "accuracy": correct / max(1, total),
        "answer_present": answer_present / max(1, total),
        "n_samples": total,
        "details": details,
    }


def phase1_steernet_acceptance(model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device) -> Dict[str, Any]:
    """Prove independent Phase-1 steering capacity before routing is trained.

    Each candidate is measured as a single-layer, gate=1 intervention.  The
    Every candidate receives a small held-out generation screen, then the
    screen leaders receive the full benchmark-format free-generation probe.
    NLL is auxiliary and never allowed to overrule an autoregressive
    regression.
    """
    train_n = min(int(cfg.phase1_acceptance_examples), len(train_data))
    val_n = min(int(cfg.phase1_acceptance_examples), len(eval_data))
    small_run_waiver = _small_run_waiver(val_n, cfg)
    train_probe = [train_data[i] for i in range(train_n)]
    val_probe = [eval_data[i] for i in range(val_n)]
    old_override = model._routing_override
    old_layers = getattr(cfg, "fixed_steer_eval_layers", None)
    started_at = time.monotonic()
    try:
        print("\n" + "=" * 80)
        print("PHASE 1 ACCEPTANCE AUDIT: Independent SteerNet Capacity")
        print(
            f"         Measuring baseline + {len(model.target_layers)} single-layer candidates "
            f"on {train_n} train / {val_n} validation examples."
        )
        print(
            f"         Then generating {min(int(cfg.phase1_acceptance_accuracy_examples), val_n)} "
            "held-out zero-shot answers for baseline and generation-screen leaders."
        )
        print("         Router and learned gate are disabled during this audit.")
        if small_run_waiver:
            print(
                f"         ⚠️ INCONCLUSIVE DEBUG RUN: validation={val_n} < "
                f"min_reliable_validation_examples={cfg.min_reliable_validation_examples}. "
                "Measurements will be logged, but failure will not stop the curriculum."
            )
        print("=" * 80)
        model.set_routing_override("none")
        model.set_gate_freeze(False)
        print("  [1/5] Baseline answer-weighted teacher-forced NLL...")
        base_train_nll = evaluate_loss(model, train_probe, cfg, device, "Acceptance baseline train NLL", answer_aware=True)
        base_val_nll = evaluate_loss(model, val_probe, cfg, device, "Acceptance baseline val NLL", answer_aware=True)
        print("  [2/5] Baseline held-out generation accuracy...")
        generation_n = min(int(cfg.phase1_acceptance_accuracy_examples), val_n)
        base_generation = _phase1_zero_shot_generation_probe(
            model, tokenizer, val_probe, cfg, device, generation_n,
            progress_desc="Acceptance baseline held-out generation",
        )
        base_val_acc = base_generation["accuracy"]
        print(
            f"        baseline: train_nll={base_train_nll:.4f} val_nll={base_val_nll:.4f} "
            f"heldout_acc={base_val_acc:.1%}"
        )

        candidates = []
        total_candidates = len(model.target_layers)
        print(f"  [3/5] Single-layer NLL sweep across {total_candidates} SteerNets...")
        for candidate_index, layer in enumerate(model.target_layers, start=1):
            model.set_routing_override("only", [layer])
            model.set_gate_freeze(True, 1.0)
            print(f"        candidate {candidate_index}/{total_candidates}: L{layer}...")
            candidate = {
                "layer": int(layer),
                "train_nll": evaluate_loss(model, train_probe, cfg, device, f"Acceptance L{layer} train NLL", answer_aware=True),
                "val_nll": evaluate_loss(model, val_probe, cfg, device, f"Acceptance L{layer} val NLL", answer_aware=True),
            }
            candidates.append(candidate)
            print(f"          L{layer}: train_nll={candidate['train_nll']:.4f} val_nll={candidate['val_nll']:.4f}")
        # A teacher-forced shortlist alone reproduced the false L9 selection
        # in the failed run.  Screen *every* candidate on held-out free
        # generation, then spend the larger 32-example budget only on the
        # screen leaders.  This remains validation-only and is reproducible.
        screen_n = min(
            generation_n,
            max(1, int(getattr(cfg, "phase1_generation_screen_examples", 8))),
        )
        print(f"  [4/5] Held-out generation screen across all {total_candidates} candidates ({screen_n} each)...")
        for row in candidates:
            screen = _phase1_zero_shot_generation_probe(
                model, tokenizer, val_probe, cfg, device, screen_n, layer=row["layer"],
                progress_desc=f"Acceptance L{row['layer']} generation screen",
            )
            row["screen_generation_accuracy"] = screen["accuracy"]
            row["screen_answer_present"] = screen["answer_present"]
            print(
                f"          L{row['layer']}: screen_acc={screen['accuracy']:.1%} "
                f"answer_present={screen['answer_present']:.1%}"
            )
        shortlist_n = max(1, min(int(getattr(cfg, "phase1_generation_candidate_count", 3)), len(candidates)))
        shortlist = sorted(
            candidates,
            key=lambda row: (-row["screen_generation_accuracy"], row["val_nll"]),
        )[:shortlist_n]
        shortlist_labels = [f"L{row['layer']}" for row in shortlist]
        print(f"  [5/5] Full held-out generation shortlist: {shortlist_labels}...")
        for row in shortlist:
            generation = _phase1_zero_shot_generation_probe(
                model, tokenizer, val_probe, cfg, device, generation_n, layer=row["layer"],
                progress_desc=f"Acceptance L{row['layer']} held-out generation",
            )
            row["heldout_generation_accuracy"] = generation["accuracy"]
            row["heldout_answer_present"] = generation["answer_present"]
            print(
                f"          L{row['layer']}: heldout_acc={generation['accuracy']:.1%} "
                f"answer_present={generation['answer_present']:.1%}"
            )
        best = max(
            shortlist,
            key=lambda row: (row["heldout_generation_accuracy"], -row["val_nll"]),
        )
        # The fixed control previously forced alpha=1 even when the learned
        # direction was only useful at a smaller residual intensity.  Calibrate
        # this one scalar on the same held-out validation generation protocol;
        # external benchmark examples remain untouched.
        scale_grid = sorted({
            min(1.0, max(0.0, float(scale)))
            for scale in getattr(cfg, "phase1_gate_scale_grid", (1.0,))
        })
        scale_results = [{
            "gate_scale": 1.0,
            "accuracy": best["heldout_generation_accuracy"],
            "answer_present": best["heldout_answer_present"],
        }]
        print(f"  [calibration] L{best['layer']} residual intensity grid: {scale_grid}")
        for scale in scale_grid:
            if abs(scale - 1.0) < 1e-8:
                continue
            calibrated = _phase1_zero_shot_generation_probe(
                model, tokenizer, val_probe, cfg, device, generation_n, layer=best["layer"], gate_value=scale,
                progress_desc=f"Acceptance L{best['layer']} alpha={scale:.2f}",
            )
            scale_results.append({
                "gate_scale": scale,
                "accuracy": calibrated["accuracy"],
                "answer_present": calibrated["answer_present"],
            })
            print(
                f"          alpha={scale:.2f}: heldout_acc={calibrated['accuracy']:.1%} "
                f"answer_present={calibrated['answer_present']:.1%}"
            )
        best_scale = max(scale_results, key=lambda row: (row["accuracy"], row["gate_scale"]))
        selected_val_acc = best_scale["accuracy"]
        nll_reduction = (base_train_nll - best["train_nll"]) / max(base_train_nll, 1e-8)
        val_nll_reduction = (base_val_nll - best["val_nll"]) / max(base_val_nll, 1e-8)
        acc_gain = selected_val_acc - base_val_acc
        cfg.fixed_steer_eval_layers = [best["layer"]]
        cfg.fixed_steer_eval_gate_scale = float(best_scale["gate_scale"])
        report = {
            "baseline_train_nll": base_train_nll,
            "baseline_val_nll": base_val_nll,
            "baseline_validation_accuracy": base_val_acc,
            "candidates": candidates,
            "selected_layer": best["layer"],
            "selected_validation_accuracy": selected_val_acc,
            "fixed_steer_gate_scale": cfg.fixed_steer_eval_gate_scale,
            "gate_scale_candidates": scale_results,
            "train_relative_nll_reduction": nll_reduction,
            "val_relative_nll_reduction": val_nll_reduction,
            "validation_accuracy_gain": acc_gain,
            "validation_examples": val_n,
            "reliability_waived": small_run_waiver,
            "passed": (
                nll_reduction >= float(cfg.phase1_min_train_nll_reduction)
                and val_nll_reduction >= float(cfg.phase1_min_val_nll_reduction)
                and acc_gain >= float(getattr(
                    cfg,
                    "phase1_min_validation_accuracy_gain",
                    cfg.phase1_min_train_accuracy_gain,
                ))
            ),
        }
        print(
            "  Phase-1 acceptance: "
            f"layer=L{best['layer']} | train/val NLL Δ={nll_reduction * 100:+.1f}%/{val_nll_reduction * 100:+.1f}% "
            f"| alpha={cfg.fixed_steer_eval_gate_scale:.2f} | held-out generation Δ={acc_gain * 100:+.1f}pp "
            f"| {'PASS' if report['passed'] else 'FAIL'}"
        )
        if acc_gain < float(getattr(cfg, "phase1_accuracy_warning_gain", 0.05)):
            print(
                "  ⚠️ Generation improvement is below the reporting target; NLL capacity is accepted, "
                "but the final fixed-SteerNet benchmark must confirm a meaningful accuracy gain."
            )
        print(f"  Acceptance audit finished in {(time.monotonic() - started_at) / 60:.1f} min.")
        if not report["passed"] and small_run_waiver:
            report["status"] = "INCONCLUSIVE"
            print(
                "  ⚠️ Phase-1 acceptance is INCONCLUSIVE, not PASS: the tiny validation holdout cannot "
                "validate SteerNet capacity. Continuing only as a pipeline smoke test."
            )
        elif not report["passed"]:
            report["status"] = "FAIL_CONTINUED"
            warning = (
                "  ⚠️ Phase-1 SteerNet acceptance FAILED. Router/gate results are diagnostic-only because "
                "independent steering capacity was not established. Continuing to preserve checkpoints and evidence."
            )
            print(warning)
            if not getattr(cfg, "continue_on_diagnostic_failure", True):
                raise RuntimeError(warning)
        else:
            report["status"] = "PASS"
        return report
    finally:
        model.set_gate_freeze(True, 1.0)
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])
        # Preserve the chosen fixed ablation layer even if an earlier config
        # had none; it is validation-selected above, not test-selected.
        if not getattr(cfg, "fixed_steer_eval_layers", None):
            cfg.fixed_steer_eval_layers = old_layers


def audit_ars_intervention_health(model, eval_data, cfg: RSCConfig, device: torch.device) -> Dict[str, Any]:
    """Fail closed for research-sized evaluations with routing/gate collapse.

    This is evaluated with the same deterministic argmax route and prompt
    shielding used by generation.  It is deliberately a health gate, not a
    benchmark: test labels never choose a route or tune a threshold.  Tiny
    smoke runs report the same diagnostics as INCONCLUSIVE instead of making
    claims from an unreliable holdout.
    """
    previous_mode = getattr(model, "_router_inference_mode", "argmax")
    previous_override = model._routing_override
    model.reset_routing_diagnostics()
    model.eval()
    model.set_router_inference_mode("argmax")
    model.set_routing_override(None)
    model.set_gate_freeze(False)
    n = min(int(cfg.intervention_audit_examples), len(eval_data))
    small_run_waiver = _small_run_waiver(n, cfg)
    try:
        with torch.no_grad():
            for i in range(n):
                row = eval_data[i]
                ids = row["input_ids"].unsqueeze(0).to(device)
                mask = row["attention_mask"].unsqueeze(0).to(device)
                labels = row["labels"].unsqueeze(0).to(device)
                shield = row.get("prompt_shield_mask")
                shield = shield.unsqueeze(0).to(device) if shield is not None else None
                model._attn_mask_cache = mask
                _set_batch_context(model, mask, labels, shield)
                _ = model(input_ids=ids, attention_mask=mask, use_cache=False)
        report = model.intervention_diagnostics()
        failures = []
        k_dist = report["realized_k_distribution"]
        if cfg.router_require_eval_k0 and k_dist.get(0, 0.0) <= 0.0:
            failures.append("K=0 was never selected")
        if report["realized_k_support"] < int(cfg.router_min_eval_k_support):
            failures.append(f"K support={report['realized_k_support']} < {cfg.router_min_eval_k_support}")
        if report["realized_k_entropy"] < float(cfg.router_min_eval_k_entropy):
            failures.append(f"K entropy={report['realized_k_entropy']:.3f} < {cfg.router_min_eval_k_entropy}")
        if report["active_layer_count"] < int(cfg.router_min_eval_active_layers):
            failures.append(f"active layers={report['active_layer_count']} < {cfg.router_min_eval_active_layers}")
        if report["layer_selection_entropy"] < float(cfg.router_min_eval_layer_entropy):
            failures.append(f"layer entropy={report['layer_selection_entropy']:.3f} < {cfg.router_min_eval_layer_entropy}")
        active_gates = [row for row in report["gate"].values() if row["intervened_tokens"] > 0]
        if not active_gates:
            failures.append("no gate received intervened tokens")
        else:
            mean_seq_std = sum(row["mean_sequence_std"] for row in active_gates) / len(active_gates)
            max_high_sat = max(row["high_saturation_pct"] for row in active_gates)
            report["mean_gate_sequence_std"] = round(mean_seq_std, 4)
            report["max_gate_high_saturation_pct"] = round(max_high_sat, 2)
            if mean_seq_std < float(cfg.gate_min_eval_sequence_std):
                failures.append(f"gate sequence std={mean_seq_std:.4f} < {cfg.gate_min_eval_sequence_std}")
            if max_high_sat > float(cfg.gate_max_eval_high_saturation_pct):
                failures.append(f"gate high saturation={max_high_sat:.1f}% > {cfg.gate_max_eval_high_saturation_pct}%")
        report["passed"] = not failures
        report["failures"] = failures
        report["validation_examples"] = n
        report["reliability_waived"] = small_run_waiver
        report["status"] = "PASS" if not failures else ("INCONCLUSIVE" if small_run_waiver else "FAIL_CONTINUED")
        print(f"  ARS intervention audit: {report}")
        if failures and small_run_waiver:
            print(
                "  ⚠️ ARS health is INCONCLUSIVE, not PASS: this tiny validation holdout cannot establish "
                "K/layer/gate diversity. Continuing only as a pipeline smoke test."
            )
        elif failures:
            warning = (
                "  ⚠️ ARS intervention health FAILED. Final checkpoint is retained for diagnosis only: "
                + "; ".join(failures)
            )
            print(warning)
            if not getattr(cfg, "continue_on_diagnostic_failure", True):
                raise RuntimeError(warning)
        return report
    finally:
        model.set_router_inference_mode(previous_mode)
        if previous_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(previous_override[0], previous_override[1])


def audit_router_policy_health(model, eval_data, cfg: RSCConfig, device: torch.device) -> Dict[str, Any]:
    """Verify deterministic K/layer diversity before expensive gate training.

    Bootstrap sampling can look diverse even if every deployment argmax is the
    same K and triplet.  This preflight runs the deployment policy itself, so
    a collapsed router is rejected before Phase 2B consumes hours of GPU time.
    """
    previous_mode = getattr(model, "_router_inference_mode", "argmax")
    previous_override = model._routing_override
    model.reset_routing_diagnostics()
    model.eval()
    model.set_router_inference_mode("argmax")
    model.set_routing_override(None)
    n = min(int(getattr(cfg, "router_preflight_examples", 128)), len(eval_data))
    waiver = _small_run_waiver(n, cfg)
    try:
        with torch.no_grad():
            for i in range(n):
                row = eval_data[i]
                ids = row["input_ids"].unsqueeze(0).to(device)
                mask = row["attention_mask"].unsqueeze(0).to(device)
                labels = row["labels"].unsqueeze(0).to(device)
                shield = row.get("prompt_shield_mask")
                shield = shield.unsqueeze(0).to(device) if shield is not None else None
                model._attn_mask_cache = mask
                _set_batch_context(model, mask, labels, shield)
                _ = model(input_ids=ids, attention_mask=mask, use_cache=False)
        report = model.intervention_diagnostics()
        failures = []
        k_dist = report["realized_k_distribution"]
        if cfg.router_require_eval_k0 and k_dist.get(0, 0.0) <= 0.0:
            failures.append("K=0 was never selected")
        if report["realized_k_support"] < int(cfg.router_min_eval_k_support):
            failures.append(f"K support={report['realized_k_support']} < {cfg.router_min_eval_k_support}")
        if report["realized_k_entropy"] < float(cfg.router_min_eval_k_entropy):
            failures.append(f"K entropy={report['realized_k_entropy']:.3f} < {cfg.router_min_eval_k_entropy}")
        if report["active_layer_count"] < int(cfg.router_min_eval_active_layers):
            failures.append(f"active layers={report['active_layer_count']} < {cfg.router_min_eval_active_layers}")
        if report["layer_selection_entropy"] < float(cfg.router_min_eval_layer_entropy):
            failures.append(f"layer entropy={report['layer_selection_entropy']:.3f} < {cfg.router_min_eval_layer_entropy}")
        report.update({
            "validation_examples": n,
            "reliability_waived": waiver,
            "passed": not failures,
            "failures": failures,
            "status": "PASS" if not failures else ("INCONCLUSIVE" if waiver else "FAIL_CONTINUED"),
        })
        print(f"  Router deployment preflight: {report}")
        if failures and not waiver:
            warning = (
                "  ⚠️ Router deployment preflight FAILED. Continuing with diagnostic-only Phase 2 results: "
                + "; ".join(failures)
            )
            print(warning)
            if getattr(cfg, "router_fail_fast_on_collapse", False) and not getattr(cfg, "continue_on_diagnostic_failure", True):
                raise RuntimeError(warning)
        return report
    finally:
        model.set_router_inference_mode(previous_mode)
        if previous_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(previous_override[0], previous_override[1])


def _evaluate_and_maybe_checkpoint(
    model, tokenizer, eval_data, cfg: RSCConfig, device: torch.device, save_path: str,
    best_acc: float, best_val: float, patience_ctr: int, train_router: bool, label: str,
):
    """Evaluates val CE loss + accuracy and saves checkpoint if improved."""
    if train_router:
        model.reset_routing_diagnostics()
    val_loss = evaluate_loss(model, eval_data, cfg, device, answer_aware=True)
    acc = epoch_end_accuracy(model, tokenizer, eval_data, device, n_samples=cfg.accuracy_eval_n, seed=cfg.seed)
    # A generation sample is a noisy checkpoint signal.  It may break ties,
    # but it must never replace a checkpoint whose teacher-forced validation
    # loss is materially better (the old rule did exactly that in Phase 2B).
    val_improved = val_loss < best_val - cfg.early_stop_delta
    acc_improved = acc > best_acc + 0.005
    val_tied = abs(val_loss - best_val) <= cfg.early_stop_delta
    improved = val_improved or (getattr(cfg, "use_accuracy_checkpoint", True) and val_tied and acc_improved)
    note = ""
    if train_router:
        diag = model.routing_diagnostics()
        n_candidates = max(1, len(diag))
        dead = sum(1 for s in diag.values() if s["selected_forwards"] / max(1, s["seen_forwards"]) < 0.02)
        k_dist = model.routing_k_distribution()
        k_str = ", ".join(f"K={k}:{v:.2f}" for k, v in k_dist.items() if v > 0.01)
        intervention = model.intervention_diagnostics() if hasattr(model, "intervention_diagnostics") else {}
        note = (
            f"  (active layers: {n_candidates - dead}/{n_candidates} | {k_str}"
            f" | K-H={intervention.get('realized_k_entropy', 0.0):.2f}"
            f" | L-H={intervention.get('layer_selection_entropy', 0.0):.2f})"
        )
    print(f"    [checkpoint check] {label}: val={val_loss:.4f}  acc={acc:.3f}  {'[SAVED]' if improved else ''}{note}")
    if hasattr(model, "intervention_diagnostics"):
        gate_rows = model.intervention_diagnostics().get("gate", {})
        active_gates = [row for row in gate_rows.values() if row.get("intervened_tokens", 0) > 0]
        if active_gates:
            mean_gate_std = sum(row["mean_sequence_std"] for row in active_gates) / len(active_gates)
            max_high_sat = max(row["high_saturation_pct"] for row in active_gates)
            print(
                f"    [gate health] selected_layers={len(active_gates)} "
                f"mean_token_std={mean_gate_std:.4f} max_high_sat={max_high_sat:.1f}%"
            )
    if train_router and hasattr(model, "router_k_cap_gap_summary"):
        print(model.router_k_cap_gap_summary())
    best_state = None
    if improved:
        best_acc = max(best_acc, acc)
        best_val, patience_ctr = val_loss, 0
        best_state = {
            k: v.clone() for k, v in model.state_dict().items()
            if "steer_net" in k or "gate" in k or "router" in k
        }
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        save_ars_checkpoint(model, save_path, phase=label, cfg=cfg)
    else:
        patience_ctr += 1
    return best_acc, best_val, best_state, patience_ctr, improved, val_loss, acc


def train_phase1_steernet(model, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Phase 1: Pure SteerNet training with gates pinned at 1.0 and router in coverage warmup."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_gpu_memory("train_phase1 start", cfg)

    print("\n" + "=" * 80)
    print("PHASE 1: Pure SteerNet Training (Hidden States)")
    print("         Gates FIXED at 1.0 (disabled)")
    print(f"         lr={cfg.phase1_lr}, epochs={cfg.phase1_epochs}")
    print("=" * 80)

    model.set_gate_freeze(True, 1.0)
    model.set_router_warmup(True)
    model.set_router_exploration(False)
    for p in model.router.parameters():
        p.requires_grad = False

    steer_params = []
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = True
            steer_params.append(p)
    print(f"  SteerNet params: {sum(p.numel() for p in steer_params):,}")

    optimizer_p1 = torch.optim.AdamW(
        steer_params,
        lr=cfg.phase1_lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    steps_per_epoch = len(train_data) // (cfg.effective_batch_size)
    total_p1 = max(1, steps_per_epoch * cfg.phase1_epochs)
    warmup_p1 = int(total_p1 * cfg.phase1_warmup_ratio)

    lr_reducer_p1 = LRReducer(
        optimizer_p1,
        total_p1,
        factor=cfg.lr_reduce_factor,
        patience=cfg.lr_reduce_patience,
        min_lr=cfg.lr_reduce_min_lr,
        delta=cfg.early_stop_delta,
        warmup_steps=warmup_p1,
    )
    lr_reducer_p1.base_lrs = [cfg.phase1_lr]
    optimizer_p1.param_groups[0]["lr"] = cfg.phase1_lr
    ckpt_p1 = CheckpointManager(cfg.checkpoint_dir)
    grad_mon_p1 = GradientMonitor(cfg.grad_warn_threshold)

    best_val_p1 = float("inf")
    best_state_p1 = None
    p1_patience_ctr = 0
    logs = {"p1_train": [], "p1_val": []}
    scaler_p1 = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(max(0, cfg.phase1_epochs)):
        model.train()
        # The Phi-2 backbone is frozen.  Its dropout must not make Phase 1
        # optimise a different function from eval()/generate(), where the
        # dropout is disabled.  Phase 2 already enforced this invariant; its
        # absence here was a direct train/deployment distribution mismatch.
        model.base_model.eval()
        # The phase-specific modules use their normal training mode, but the
        # router only supplies deterministic warmup coverage and is frozen.
        model.router.eval()
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_loss, n_optimizer_steps = 0.0, 0
        accum_count_p1 = 0
        optimizer_p1.zero_grad(set_to_none=True)

        pbar = tqdm(
            range(0, len(indices), cfg.batch_size), desc=f"Phase1 {epoch+1}/{cfg.phase1_epochs}"
        )
        for start in pbar:
            batch_idx = indices[start : start + cfg.batch_size]
            ids = torch.stack([train_data[i]["input_ids"] for i in batch_idx]).to(device)
            mask = torch.stack([train_data[i]["attention_mask"] for i in batch_idx]).to(device)
            lbls = torch.stack([train_data[i]["labels"] for i in batch_idx]).to(device)
            shield = None
            if "prompt_shield_mask" in train_data[batch_idx[0]]:
                shield = torch.stack([train_data[i]["prompt_shield_mask"] for i in batch_idx]).to(device)
            model._attn_mask_cache = mask
            _set_batch_context(model, mask, lbls, shield)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(input_ids=ids, attention_mask=mask, labels=lbls, use_cache=False)
                loss = _answer_aware_loss(out.logits, lbls, cfg) / cfg.grad_accum

            if not torch.isfinite(loss):
                continue

            scaler_p1.scale(loss).backward()
            accum_count_p1 += 1

            is_last_p1 = start + cfg.batch_size >= len(indices)
            if accum_count_p1 == cfg.grad_accum or is_last_p1:
                scaler_p1.unscale_(optimizer_p1)
                for p in steer_params:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(steer_params, cfg.clip_grad)
                if cfg.use_gradient_monitor:
                    grad_mon_p1.check(steer_params, n_optimizer_steps)
                scaler_p1.step(optimizer_p1)
                scaler_p1.update()
                lr_reducer_p1.scheduler_step()
                optimizer_p1.zero_grad(set_to_none=True)
                accum_count_p1 = 0
                n_optimizer_steps += 1

            epoch_loss += loss.item() * cfg.grad_accum
            pbar.set_postfix(loss=f"{loss.item() * cfg.grad_accum:.4f}")

        avg_loss = epoch_loss / max(1, n_optimizer_steps * cfg.grad_accum)
        val_loss = evaluate_loss(model, eval_data, cfg, device, answer_aware=True)
        logs["p1_train"].append(avg_loss)
        logs["p1_val"].append(val_loss)

        improved = "✅" if val_loss < best_val_p1 - cfg.early_stop_delta else "  "
        if val_loss < best_val_p1 - cfg.early_stop_delta:
            best_val_p1 = val_loss
            p1_patience_ctr = 0
            best_state_p1 = {
                k: v.clone()
                for k, v in model.state_dict().items()
                if "steer_net" in k or "router" in k
            }
        else:
            p1_patience_ctr += 1

        lr_reducer_p1.step(val_loss)
        lr_now = optimizer_p1.param_groups[0]["lr"]
        print(
            f"  Phase1 Epoch {epoch+1}: train={avg_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}  {improved}"
        )

        if (epoch + 1) % cfg.checkpoint_every_n == 0:
            ckpt_p1.save(model, optimizer_p1, epoch, val_loss, "p1")

        if p1_patience_ctr >= cfg.phase1_patience:
            print(f"  Phase1 early stopping (no improvement for {cfg.phase1_patience} epochs)")
            break

    if best_state_p1:
        model.load_state_dict(best_state_p1, strict=False)
    save_steer_weights(
        model, cfg.lora_save_path, cfg=cfg, phase="phase1-steernet", prefixes=("steer_nets.",)
    )
    grad_mon_p1.summary()
    print(f"\n✅ Phase 1 complete — best_val = {best_val_p1:.4f}")

    model.set_router_warmup(False)
    model.reset_routing_diagnostics()
    log_gpu_memory("after Phase 1", cfg)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_state_p1, best_val_p1, logs


def train_router_bootstrap(model, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Phase 1.5: Router Bootstrap with frozen SteerNet & Gate using RLOO counterfactual loss."""
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = False
    model.set_gate_freeze(True, 1.0)
    model.set_router_warmup(False)
    model.set_routing_override(None)

    router_params = list(model.router.parameters())
    for p in router_params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW(router_params, lr=cfg.router_bootstrap_lr, weight_decay=cfg.weight_decay)
    model.reset_routing_diagnostics()

    pool_size = min(cfg.router_bootstrap_n_examples, len(train_data))
    rng = random.Random(cfg.seed)
    subset_idx = rng.sample(range(len(train_data)), pool_size)
    best_state, best_loss, patience_ctr = None, float("inf"), 0

    print("\n" + "=" * 80)
    print("PHASE 1.5: Router Bootstrap (RLOO Counterfactual Loss)")
    print(f"           epochs={cfg.router_bootstrap_epochs}, lr={cfg.router_bootstrap_lr}")
    print("=" * 80)

    for epoch in range(cfg.router_bootstrap_epochs):
        random.shuffle(subset_idx)
        model.train()
        running, n_steps = 0.0, 0
        counterfactual_k_actions = None
        rloo_batch_size = max(2, int(getattr(cfg, "router_rloo_batch_size", 8)))
        pbar = tqdm(range(0, len(subset_idx), rloo_batch_size), desc=f"RouterBootstrap {epoch + 1}/{cfg.router_bootstrap_epochs}")
        for start in pbar:
            batch_indices = subset_idx[start : start + rloo_batch_size]
            ids = torch.stack([train_data[i]["input_ids"] for i in batch_indices]).to(device)
            mask = torch.stack([train_data[i]["attention_mask"] for i in batch_indices]).to(device)
            lbls = torch.stack([train_data[i]["labels"] for i in batch_indices]).to(device)
            shield = None
            if "prompt_shield_mask" in train_data[batch_indices[0]]:
                shield = torch.stack([train_data[i]["prompt_shield_mask"] for i in batch_indices]).to(device)
            model._attn_mask_cache = mask

            _set_batch_context(model, mask, lbls, shield)
            prompt_mask = mask.bool() & lbls.eq(-100) & ~(shield.bool() if shield is not None else torch.zeros_like(mask, dtype=torch.bool))
            loss, rloo_diag = compute_router_rloo_loss(
                model, ids, mask, lbls, cfg, prompt_mask=prompt_mask, shield_mask=shield
            )
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router_params, cfg.clip_grad)
            optimizer.step()

            running += loss.item()
            n_steps += 1
            postfix_dict = {"loss": f"{loss.item():.4f}"}
            if rloo_diag:
                postfix_dict["mean_Kcf"] = f"{rloo_diag.get('mean_k_sampled', 0.0):.2f}"
                if "win_rate_vs_base" in rloo_diag:
                    postfix_dict["win_base"] = f"{rloo_diag['win_rate_vs_base'] * 100:.0f}%"
                counterfactual_k_actions = rloo_diag.get("counterfactual_k_actions", counterfactual_k_actions)
            pbar.set_postfix(**postfix_dict)

        avg_loss = running / max(1, n_steps)
        k_cf_str = (
            " | K counterfactuals={0," + ",".join(str(k) for k in counterfactual_k_actions) + "}"
            if counterfactual_k_actions else ""
        )
        diag = model.routing_diagnostics()
        active_cnt = sum(1 for s in diag.values() if s["selected_forwards"] > 0)
        coop = model.get_cooperation_analysis() if hasattr(model, "get_cooperation_analysis") else {}
        top_syn_str = ", ".join(f"L{a}+L{b}:{v:+.2f}" for a, b, v in coop.get("top_synergy_pairs", [])[:2])
        win_str = f" | win_base={rloo_diag.get('win_rate_vs_base', 0.0) * 100:.1f}%" if rloo_diag else ""
        k_pref_str = (
            f" | Kpref={rloo_diag.get('k_preference_loss', 0.0):.3f} "
            f"p(K0)={rloo_diag.get('k0_probability', 0.0):.2f} "
            f"Kcost={rloo_diag.get('mean_per_layer_cost', 0.0):.3f}"
            if rloo_diag else ""
        )
        k_argmax_str = (
            " | argmax-K=" + ",".join(
                f"{k}:{v:.2f}" for k, v in rloo_diag.get("argmax_k_distribution", {}).items()
            ) if rloo_diag else ""
        )
        syn_str = f" | top synergy: [{top_syn_str}]" if top_syn_str else ""
        print(f"  RouterBootstrap Epoch {epoch+1}: loss={avg_loss:.4f}{win_str}{k_pref_str}{k_argmax_str} | active layers={active_cnt}/{len(diag)}{syn_str}{k_cf_str}")
        if avg_loss < best_loss - 1e-4:
            best_loss, patience_ctr = avg_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items() if "router" in k}
        else:
            patience_ctr += 1
        if patience_ctr >= cfg.router_bootstrap_patience:
            break

    if best_state:
        model.load_state_dict(best_state, strict=False)
    model.set_routing_override(None)
    save_path = cfg.router_bootstrap_save_path
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    # BUG-C5 fix: was raw torch.save (no manifest/hash/config provenance).
    # Now uses save_ars_checkpoint for consistency with all other phases.
    save_ars_checkpoint(model, save_path, phase='phase1.5-router-bootstrap', cfg=cfg, prefixes=('router.',))
    print(f"✅ Router Bootstrap complete — best_probe_loss = {best_loss:.4f}")
    return best_state, best_loss


def run_staged_phase(
    model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device,
    stage_name: str, epochs: int, param_groups: List[Dict[str, Any]], save_path: str,
    train_steer: bool, train_gate: bool, train_router: bool,
    global_step: int = 0, baseline_ema: Optional[List[float]] = None, patience: int = 3,
    init_best_acc: float = 0.0, init_best_val: float = float("inf"),
) -> Tuple[int, float, float]:
    """Shared Phase 2 epoch loop for SteerNet fine-tune, Gate-only, Joint fine-tune, and Gate re-tune."""
    if baseline_ema is None:
        baseline_ema = [0.0]
    print("\n" + "=" * 80)
    print(f"{stage_name}")
    print(
        f"         epochs={epochs}  train_steer={train_steer}  "
        f"train_gate={train_gate}  train_router={train_router}"
    )
    for g in param_groups:
        print(f"         group size={sum(p.numel() for p in g['params']):,}  lr={g['lr']:.2e}")
    print("=" * 80)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.98), eps=1e-8)
    all_params = [p for g in param_groups for p in g["params"]]
    steps_per_epoch = len(train_data) // cfg.effective_batch_size
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * cfg.phase2_warmup_ratio)
    lr_reducer = LRReducer(
        optimizer, total_steps, factor=cfg.lr_reduce_factor, patience=cfg.lr_reduce_patience,
        min_lr=cfg.lr_reduce_min_lr, warmup_steps=warmup_steps,
    )
    lr_reducer.base_lrs = [g["lr"] for g in param_groups]
    for i, g in enumerate(optimizer.param_groups):
        g["lr"] = lr_reducer.base_lrs[i]
    grad_mon = GradientMonitor(cfg.grad_warn_threshold)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc, best_val, best_state, patience_ctr = init_best_acc, init_best_val, None, 0

    for epoch in range(epochs):
        model.train()
        # The frozen backbone must remain in inference mode.  Leaving it in
        # ``train`` here makes a frozen model's dropout part of the target the
        # small ARS modules are trying to optimise, while deployment uses
        # ``eval``.  Frozen route/steer modules also need their deployment
        # behaviour when training only the gate.
        model.base_model.eval()
        if not train_router:
            model.router.eval()
        if not train_steer:
            for steer_net in model.steer_nets.values():
                steer_net.eval()
        model.set_gate_freeze(not train_gate, 1.0)
        # Gate-only phases need the real task loss.  The gate still observes
        # detached h/delta, so this does not unfreeze or backpropagate into the
        # SteerNet; it only lets task error calibrate alpha.
        model._joint_grad_mode = bool(train_gate)
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_task, n_opt_steps, accum_count = 0.0, 0, 0
        no_gradient_windows = 0

        epoch_gate_l1, epoch_gate_div, epoch_gate_antisat, epoch_gate_contrast, n_gate_aux_steps = 0.0, 0.0, 0.0, 0.0, 0
        epoch_policy_loss, epoch_mean_k, n_router_rloo_steps = 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(range(0, len(indices), cfg.batch_size), desc=f"{stage_name.split(':')[0]} {epoch + 1}/{epochs}")

        for start in pbar:
            batch_idx = indices[start : start + cfg.batch_size]
            ids = torch.stack([train_data[i]["input_ids"] for i in batch_idx]).to(device)
            mask = torch.stack([train_data[i]["attention_mask"] for i in batch_idx]).to(device)
            lbls = torch.stack([train_data[i]["labels"] for i in batch_idx]).to(device)
            shield = None
            if "prompt_shield_mask" in train_data[batch_idx[0]]:
                shield = torch.stack([train_data[i]["prompt_shield_mask"] for i in batch_idx]).to(device)
            model._attn_mask_cache = mask
            _set_batch_context(model, mask, lbls, shield)

            # Compute the detached ON/OFF reward before creating this step's
            # task graph.  Reusing the live gate alphas below avoids the old
            # extra graph on a 4-example PG batch, the direct cause of T4 OOM.
            pg_advantage = None
            if (
                train_gate
                and not train_steer
                and float(getattr(cfg, "lambda_pg", 0.0)) > 0.0
                and accum_count == 0
                and global_step % cfg.pg_every_n_steps == 0
            ):
                pg_advantage = compute_pg_advantage_lite(
                    model, tokenizer, ids, lbls, device, cfg, baseline_ema,
                    attention_masks=mask,
                    router_context_mask=(
                        mask.bool() & lbls.eq(-100) & ~(
                            shield.bool() if shield is not None else torch.zeros_like(mask, dtype=torch.bool)
                        )
                    ),
                    shield_mask=shield,
                )
                _set_batch_context(model, mask, lbls, shield)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(input_ids=ids, attention_mask=mask, labels=lbls, use_cache=False)
                task_loss = _answer_aware_loss(out.logits, lbls, cfg)

            micro_loss = (task_loss / cfg.grad_accum) if (torch.isfinite(task_loss) and task_loss.requires_grad) else None

            if train_gate:
                l1_loss = compute_l1_gate(model)
                asat_loss = compute_antisat_gate(model, threshold=0.85)
                contrast_loss = compute_contrastive_gate_loss(model, ids, lbls, tokenizer, device)
                diversity_loss = compute_gate_diversity_loss(model, floor=cfg.gate_diversity_floor)
                epoch_gate_l1 += l1_loss.item()
                epoch_gate_div += diversity_loss.item()
                epoch_gate_antisat += asat_loss.item()
                epoch_gate_contrast += contrast_loss.item()
                n_gate_aux_steps += 1
                aux_loss = (
                    cfg.lambda_l1_gate * l1_loss
                    + cfg.lambda_gate_antisat * asat_loss
                    + cfg.lambda_gate_usefulness * contrast_loss
                    + cfg.lambda_gate_diversity * diversity_loss
                ) / cfg.grad_accum
                if torch.isfinite(aux_loss) and aux_loss.requires_grad:
                    micro_loss = aux_loss if micro_loss is None else (micro_loss + aux_loss)

            if pg_advantage is not None:
                pg_loss = compute_pg_gate_loss_from_live_alphas(model, pg_advantage) / cfg.grad_accum
                if torch.isfinite(pg_loss) and pg_loss.requires_grad:
                    # PG and task/aux losses share this forward's gate graph;
                    # combine them before one backward rather than retaining
                    # the whole Phi-2 graph for a second backward call.
                    micro_loss = (cfg.lambda_pg * pg_loss) if micro_loss is None else (micro_loss + cfg.lambda_pg * pg_loss)

            if micro_loss is not None:
                scaler.scale(micro_loss).backward()
            del out

            if train_steer and accum_count == 0 and global_step % cfg.invariance_every_n_steps == 0:
                inv_loss = compute_invariance_loss(
                    model, tokenizer,
                    [train_data[i]["question"] for i in batch_idx],
                    [train_data[i]["steps_list"] for i in batch_idx],
                    [train_data[i]["final_num"] for i in batch_idx],
                    cfg, device, seed_offset=global_step,
                )
                if torch.isfinite(inv_loss) and inv_loss.requires_grad:
                    scaler.scale(cfg.lambda_invariance * inv_loss).backward()

            # BUG-08 fix: trigger RLOO only at the last micro-batch of the accumulation window
            # (i.e. just before the optimizer.step()). Previously triggered at accum_count==0
            # (first micro-batch), meaning router got grad_accum × more updates than task loss.
            is_last_accum = (accum_count == cfg.grad_accum - 1) or (start + cfg.batch_size >= len(indices))
            if train_router and is_last_accum and global_step % cfg.router_probe_every_n_steps == 0:
                # A B=1 RLOO probe cannot distinguish input-specific routing
                # from a globally collapsed argmax.  Draw a separate vectorised
                # probe minibatch so the mutual-information diversity term and
                # counterfactual rewards see heterogeneous prompts.
                probe_idx = random.sample(
                    range(len(train_data)),
                    min(max(2, int(getattr(cfg, "router_rloo_batch_size", 8))), len(train_data)),
                )
                probe_ids = torch.stack([train_data[i]["input_ids"] for i in probe_idx]).to(device)
                probe_mask = torch.stack([train_data[i]["attention_mask"] for i in probe_idx]).to(device)
                probe_lbls = torch.stack([train_data[i]["labels"] for i in probe_idx]).to(device)
                probe_shield = None
                if "prompt_shield_mask" in train_data[probe_idx[0]]:
                    probe_shield = torch.stack([train_data[i]["prompt_shield_mask"] for i in probe_idx]).to(device)
                prompt_mask = probe_mask.bool() & probe_lbls.eq(-100) & ~(
                    probe_shield.bool() if probe_shield is not None else torch.zeros_like(probe_mask, dtype=torch.bool)
                )
                router_loss, rloo_diag = compute_router_rloo_loss(
                    model, probe_ids, probe_mask, probe_lbls, cfg,
                    prompt_mask=prompt_mask, shield_mask=probe_shield,
                )
                if torch.isfinite(router_loss) and router_loss.requires_grad:
                    scaler.scale(router_loss).backward()
                if rloo_diag:
                    epoch_policy_loss += rloo_diag["policy_loss"]
                    epoch_mean_k += rloo_diag["mean_k_sampled"]
                    n_router_rloo_steps += 1
            accum_count += 1
            is_last = start + cfg.batch_size >= len(indices)
            if accum_count == cfg.grad_accum or is_last:
                if not _optimizer_has_gradients(all_params):
                    # Legal for K=0: with a frozen backbone and no selected
                    # layer, this window contains no trainable graph.  AMP
                    # cannot unscale before scale/backward, so record the
                    # condition and continue instead of crashing the run.
                    no_gradient_windows += 1
                    if no_gradient_windows == 1:
                        print(
                            f"  ⚠️ {stage_name.split(':')[0]} has a no-gradient window "
                            "(usually K=0/no selected layer); optimizer step skipped."
                        )
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                else:
                    scaler.unscale_(optimizer)
                    for p in all_params:
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            p.grad.zero_()
                    torch.nn.utils.clip_grad_norm_(all_params, cfg.clip_grad)
                    if cfg.use_gradient_monitor:
                        grad_mon.check(all_params, global_step)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_reducer.scheduler_step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_step += 1
                    n_opt_steps += 1

                    if global_step % 50 == 0:
                        gc.collect()
                        log_gpu_memory(f"run_staged_phase step {global_step}", cfg)

                    if (
                        train_router
                        and cfg.router_checkpoint_every_n_steps > 0
                        and global_step % cfg.router_checkpoint_every_n_steps == 0
                    ):
                        best_acc, best_val, sub_state, patience_ctr, _, _, _ = _evaluate_and_maybe_checkpoint(
                            model, tokenizer, eval_data, cfg, device, save_path,
                            best_acc, best_val, patience_ctr, train_router,
                            label=f"step {global_step} (mid-epoch {epoch + 1})",
                        )
                        if sub_state is not None:
                            best_state = sub_state
                        if patience_ctr >= patience:
                            print(f"  {stage_name.split(':')[0]} early stopping (mid-epoch).")
                            break

            epoch_task += task_loss.item()
            postfix_kwargs = {"task": f"{task_loss.item():.4f}"}
            if pg_advantage is not None:
                postfix_kwargs["pg_adv"] = f"{pg_advantage:+.3f}"
            if train_router and n_router_rloo_steps > 0:
                postfix_kwargs["mean_k"] = f"{epoch_mean_k / n_router_rloo_steps:.2f}"
            pbar.set_postfix(**postfix_kwargs)

        if patience_ctr >= patience:
            break
        avg_task = epoch_task / max(1, n_opt_steps * cfg.grad_accum)
        best_acc, best_val, epoch_state, patience_ctr, improved, cur_val_loss, cur_acc = _evaluate_and_maybe_checkpoint(
            model, tokenizer, eval_data, cfg, device, save_path,
            best_acc, best_val, patience_ctr, train_router,
            label=f"epoch {epoch + 1} boundary",
        )
        if epoch_state is not None:
            best_state = epoch_state
        skip_note = f" | skipped_no_grad_windows={no_gradient_windows}" if no_gradient_windows else ""
        print(f"  {stage_name.split(':')[0]} Epoch {epoch + 1}: task={avg_task:.4f}  val={cur_val_loss:.4f}  acc={cur_acc:.3f}  {'✅' if improved else ''}{skip_note}")

    return global_step, best_val, best_acc


def train_streamlined_ars(model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Streamlined 2-Stage ARS Training Pipeline (Priority 5 / EXP-06).
    
    Stage 1: Representation Warmup (SteerNet on target layers, Gate = 1.0)
    Stage 2: Joint Gated Optimization (SteerNet + Token Gate simultaneously with decoupled LRs)
    """
    logs = {}
    print("\n" + "=" * 80)
    print("🚀 RUNNING STREAMLINED 2-STAGE ARS CURRICULUM")
    print("=" * 80)

    # Stage 1: Warmup
    orig_p1_epochs = cfg.phase1_epochs
    cfg.phase1_epochs = getattr(cfg, "streamlined_warmup_epochs", 3)
    best_state_p1, best_val_p1, p1_logs = train_phase1_steernet(model, train_data, eval_data, cfg, device)
    cfg.phase1_epochs = orig_p1_epochs
    logs.update(p1_logs)

    # Stage 2: Joint Gated Optimization
    model.set_gate_freeze(False)
    steer_params = []
    gate_params = []
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = True
            steer_params.append(p)
    for g in model.gates.values():
        for p in g.parameters():
            p.requires_grad = True
            gate_params.append(p)

    train_router = getattr(cfg, "use_router", True) and getattr(cfg, "fixed_layers", None) is None
    for p in model.router.parameters():
        p.requires_grad = train_router

    steer_lr = getattr(cfg, "streamlined_steer_lr", 1e-4)
    gate_lr = getattr(cfg, "streamlined_gate_lr", 1e-3)
    param_groups = [
        {"params": steer_params, "lr": steer_lr},
        {"params": gate_params, "lr": gate_lr},
    ]
    if train_router:
        param_groups.append({
            "params": list(model.router.parameters()),
            "lr": cfg.router_bootstrap_lr * cfg.joint_finetune_router_lr_scale,
        })

    global_step, best_val_s2, best_acc_s2 = run_staged_phase(
        model, tokenizer, train_data, eval_data, cfg, device,
        stage_name="STREAMLINED STAGE 2: Joint Gated Optimization",
        epochs=getattr(cfg, "streamlined_joint_epochs", 3),
        param_groups=param_groups,
        save_path=cfg.phase2_save_path,
        train_steer=True, train_gate=True, train_router=train_router,
        global_step=0, baseline_ema=[0.0], patience=max(1, cfg.phase2_patience),
    )
    logs["streamlined_stage2_val"] = best_val_s2
    logs["streamlined_stage2_acc"] = best_acc_s2

    if os.path.exists(cfg.phase2_save_path):
        load_ars_checkpoint_strict(model, cfg.phase2_save_path, device=device)
        print(f"  ✅ Loaded final streamlined checkpoint from {cfg.phase2_save_path}")

    print("\n" + "=" * 80)
    print("🎉 STREAMLINED ARS TRAINING COMPLETE!")
    print(f"   Stage 1 best validation loss: {best_val_p1:.4f}")
    print(f"   Stage 2 best validation loss: {best_val_s2:.4f} (acc={best_acc_s2:.3f})")
    print(f"   Final saved checkpoint      : {cfg.phase2_save_path}")
    print("=" * 80)
    return logs, best_val_p1, best_val_s2


def train_rsc(model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    """Complete multi-phase staged training pipeline:
    Phase 1: Pure SteerNet (Warmup Rotation, Gates=1.0)
    Phase 1.5: Router Bootstrap (RLOO Counterfactual Loss)
    Phase 2A: SteerNet Fine-Tune (Exhaustive Rotation)
    Phase 2B: Gate-Only Training (Dynamic Routing, Aux & PG Losses)
    Phase 2C: Light Joint Fine-Tuning (SteerNet + Gate + Router)
    Phase 2D: Gate Re-Tune (Against Final Router)
    """
    if getattr(cfg, "curriculum_mode", "full") == "streamlined":
        return train_streamlined_ars(model, tokenizer, train_data, eval_data, cfg, device)

    logs = {}

    # PHASE 1: Pure SteerNet
    best_state_p1, best_val_p1, p1_logs = train_phase1_steernet(model, train_data, eval_data, cfg, device)
    logs.update(p1_logs)
    logs["phase1_acceptance"] = phase1_steernet_acceptance(
        model, tokenizer, train_data, eval_data, cfg, device
    )

    # A router cannot repair a SteerNet that has already failed held-out free
    # generation.  Continuing Phase 2 in that state merely turns an NLL-only
    # artefact into a misleading "ARS" checkpoint, while consuming hours of
    # GPU time.  Preserve Phase 1 for diagnosis, save a strict checkpoint, and
    # return a no-intervention deployment state instead.
    phase1_report = logs["phase1_acceptance"]
    if (
        not phase1_report["passed"]
        and not phase1_report.get("reliability_waived", False)
        and bool(getattr(cfg, "skip_adaptive_phases_when_phase1_fails", True))
    ):
        print("\n" + "=" * 80)
        print("PHASE 1 CAPACITY GATE: ADAPTIVE PHASES SKIPPED")
        print(
            "         Held-out zero-shot generation did not establish an independently useful SteerNet."
        )
        print(
            "         Router/Gate optimisation is skipped; their NLL-only rewards would be scientifically invalid."
        )
        print("         Phase-1 checkpoint and acceptance evidence are retained for diagnosis.")
        print("=" * 80)
        model.set_routing_override("none")
        model.set_gate_freeze(False)
        save_ars_checkpoint(model, cfg.phase2_save_path, phase="phase1-failed-baseline-deployment", cfg=cfg)
        logs["adaptive_phases_status"] = "SKIPPED_UNPROVEN_STEERNET"
        logs["router_bootstrap_loss"] = None
        logs["router_bootstrap_preflight"] = None
        logs["ars_intervention_audit"] = {
            "status": "NOT_RUN_UNPROVEN_STEERNET",
            "passed": False,
            "failures": ["Phase-1 held-out generation capacity gate failed"],
        }
        print("\n🎉 TRAINING FINISHED WITH DIAGNOSTIC-ONLY PHASE-1 ARTIFACTS")
        print(f"   Phase 1 best validation loss : {best_val_p1:.4f}")
        print(f"   Final saved checkpoint        : {cfg.phase2_save_path}")
        return logs, best_val_p1, best_val_p1

    # PHASE 1.5: Router Bootstrap
    _, best_p15 = train_router_bootstrap(model, train_data, eval_data, cfg, device)
    logs["router_bootstrap_loss"] = best_p15
    logs["router_bootstrap_preflight"] = audit_router_policy_health(model, eval_data, cfg, device)
    router_report = logs["router_bootstrap_preflight"]
    if (
        not router_report["passed"]
        and not router_report.get("reliability_waived", False)
        and bool(getattr(cfg, "skip_adaptive_phases_when_router_fails", True))
    ):
        print("\n" + "=" * 80)
        print("ROUTER DEPLOYMENT GATE: ADAPTIVE PHASES SKIPPED")
        print("         Deployment argmax collapsed; Gate/Phase 2 cannot make this a valid ARS policy.")
        print("         Retaining Phase-1 and Router diagnostics; no Full-ARS checkpoint claim is produced.")
        print("=" * 80)
        logs["adaptive_phases_status"] = "SKIPPED_COLLAPSED_ROUTER"
        logs["ars_intervention_audit"] = {
            "status": "NOT_RUN_COLLAPSED_ROUTER",
            "passed": False,
            "failures": list(router_report.get("failures", [])),
        }
        # The Phase-1 artifact is intentionally retained.  Do not overwrite
        # the deployment checkpoint with an invalid gate/router composition.
        return logs, best_val_p1, best_val_p1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # SteerNet parameter collection and freezing
    steer_params_ft = []
    for sn in model.steer_nets.values():
        for p in sn.parameters():
            p.requires_grad = False
            steer_params_ft.append(p)

    # PHASE 2A: SteerNet Fine-tune (Skipped by default under Frozen SteerNet discipline)
    if getattr(cfg, "steer_finetune_epochs", 0) > 0:
        for p in steer_params_ft:
            p.requires_grad = True
        for p in model.router.parameters():
            p.requires_grad = False
        model.set_gate_freeze(True, 1.0)
        model.set_router_warmup(True)

        global_step, best_p2a_val, best_p2a_acc = run_staged_phase(
            model, tokenizer, train_data, eval_data, cfg, device,
            stage_name="PHASE 2A: SteerNet Fine-tune (exhaustive rotation, gate=1.0)",
            epochs=cfg.steer_finetune_epochs,
            param_groups=[{"params": steer_params_ft, "lr": cfg.steer_finetune_lr}],
            save_path=cfg.steer_finetune_save_path,
            train_steer=True, train_gate=False, train_router=False,
            global_step=0, baseline_ema=[0.0], patience=cfg.phase2_patience,
        )
        logs["p2a_val"] = best_p2a_val
        for p in steer_params_ft:
            p.requires_grad = False
    else:
        print("\n" + "=" * 80)
        print("PHASE 2A: SteerNet Fine-tune SKIPPED (Frozen SteerNet Discipline)")
        print("         SteerNet weights locked to Phase 1 optimal checkpoint.")
        print("=" * 80)
        global_step = 0
        best_p2a_val = best_val_p1
        logs["p2a_val"] = best_val_p1

    model.reset_routing_diagnostics()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PHASE 2B: Gate-only Training
    for p in steer_params_ft:
        p.requires_grad = False
    model.set_gate_freeze(False)
    model.set_router_warmup(False)
    # Gate-only learning must see the same deterministic route that will be
    # deployed and validated.  Sampling routes here trained the gate against a
    # different intervention distribution from argmax evaluation.
    model.set_router_exploration(False)
    gate_params_ft = []
    for key in sorted(model.gates.keys(), key=lambda k: int(k)):
        for p in model.gates[key].parameters():
            p.requires_grad = True
            gate_params_ft.append(p)

    global_step, best_p2b_val, best_p2b_acc = run_staged_phase(
        model, tokenizer, train_data, eval_data, cfg, device,
        stage_name="PHASE 2B: Gate-only Training (SteerNet + Router frozen)",
        epochs=cfg.gate_only_epochs,
        param_groups=[{"params": gate_params_ft, "lr": cfg.gate_only_lr}],
        save_path=cfg.gate_only_save_path,
        train_steer=False, train_gate=True, train_router=False,
        global_step=global_step, baseline_ema=[0.0], patience=cfg.phase2_patience,
    )
    logs["p2b_val"] = best_p2b_val

    # run_staged_phase intentionally does not mutate the model back to its
    # best in-memory state.  Phase 2B therefore used to hand its *last* epoch
    # (which can be worse) to Phase 2C, and Phase 2C could overwrite the only
    # final checkpoint even when it was inferior to Phase 2B.  Reload the
    # audited Phase-2B best checkpoint and seed the global Phase-2 checkpoint
    # with it before allowing router updates.
    if not os.path.exists(cfg.gate_only_save_path):
        raise RuntimeError("Phase 2B completed without a checkpoint; cannot establish the Phase 2 baseline.")
    load_ars_checkpoint_strict(model, cfg.gate_only_save_path, device=device)
    save_ars_checkpoint(model, cfg.phase2_save_path, phase="phase2b-global-baseline", cfg=cfg)
    model.reset_routing_diagnostics()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PHASE 2C: Coordinated Joint Fine-tune (SteerNet strictly FROZEN to preserve learned steering direction)
    for p in steer_params_ft:
        p.requires_grad = False
    for p in gate_params_ft:
        p.requires_grad = True
    for p in model.router.parameters():
        p.requires_grad = True
    model.set_router_exploration(True)
    model._joint_grad_mode = True  # Let task gradients flow through alpha to optimize gate directly

    joint_groups = [
        {"params": gate_params_ft, "lr": cfg.gate_only_lr * cfg.joint_finetune_lr_scale},
        {
            "params": list(model.router.parameters()),
            "lr": cfg.router_bootstrap_lr * cfg.joint_finetune_router_lr_scale,
        },
    ]

    global_step, best_val_p2, best_acc_p2 = run_staged_phase(
        model, tokenizer, train_data, eval_data, cfg, device,
        stage_name="PHASE 2C: Coordinated Gate+Router Fine-tune (SteerNet Frozen)",
        epochs=cfg.joint_finetune_epochs,
        param_groups=joint_groups,
        save_path=cfg.phase2_save_path,
        train_steer=False, train_gate=True, train_router=True,
        global_step=global_step, baseline_ema=[0.0], patience=max(1, cfg.phase2_patience),
        init_best_acc=best_p2b_acc, init_best_val=best_p2b_val,
    )
    model._joint_grad_mode = False  # Restore detached mode for Phase 2D
    logs["p2_val"] = [best_val_p2]

    # PHASE 2D: Gate Re-tune.  Reload the globally best Phase-2 state, rather
    # than continuing from the final (possibly non-improving) Phase-2C step.
    load_ars_checkpoint_strict(model, cfg.phase2_save_path, device=device)
    for p in model.router.parameters():
        p.requires_grad = False
    for p in steer_params_ft:
        p.requires_grad = False
    model.set_gate_freeze(False)
    model.set_router_exploration(False)
    for p in gate_params_ft:
        p.requires_grad = True

    global_step, best_p2d_val, best_p2d_acc = run_staged_phase(
        model, tokenizer, train_data, eval_data, cfg, device,
        stage_name="PHASE 2D: Gate Re-tune (against final router)",
        epochs=cfg.gate_retune_epochs,
        param_groups=[{"params": gate_params_ft, "lr": cfg.gate_retune_lr}],
        save_path=cfg.phase2_save_path,
        train_steer=False, train_gate=True, train_router=False,
        global_step=global_step, baseline_ema=[0.0], patience=cfg.phase2_patience,
        init_best_acc=best_acc_p2, init_best_val=best_val_p2,
    )
    logs["p2d_val"] = best_p2d_val
    model.set_router_exploration(False)

    if os.path.exists(cfg.phase2_save_path):
        load_ars_checkpoint_strict(model, cfg.phase2_save_path, device=device)
        print(f"  ✅ Loaded final joint+gate-retune checkpoint from {cfg.phase2_save_path}")
    logs["ars_intervention_audit"] = audit_ars_intervention_health(model, eval_data, cfg, device)

    print("\n" + "=" * 80)
    print("🎉 FULL ARS TRAINING COMPLETE!")
    print(f"   Phase 1 best validation loss           : {best_val_p1:.4f}")
    print(f"   Router bootstrap best probe loss       : {best_p15:.4f}")
    print(f"   Phase 2A (SteerNet ft) best val        : {best_p2a_val:.4f}")
    print(f"   Phase 2B (Gate-only) best val          : {best_p2b_val:.4f}")
    print(f"   Phase 2C (joint ft) best val           : {best_val_p2:.4f}")
    print(f"   Phase 2D (gate re-tune) best val       : {best_p2d_val:.4f}")
    print(f"   Final saved checkpoint                 : {cfg.phase2_save_path}")
    print("=" * 80)

    return logs, best_val_p1, best_val_p2
