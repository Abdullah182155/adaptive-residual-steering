# ARS Deep Code Review — Full Report

**Repository:** `Abdullah182155/adaptive-residual-steering` · **Branch:** `development` · **HEAD reviewed:** `2ef765e`

## 0. Scope, method, and how to read the evidence tags

**The repository does not match the brief.** Handing this to an engineer requires knowing that:

| Brief says | Reality on `development` |
|---|---|
| CodeGen-350M, code-generation pass@1 | **Phi-2 (2.7B)**, GSM8K free-generation accuracy (`src/configs/config.py` L9; `scripts/run_experiment.py` L61–81) |
| root `config.py`, `train_rsc.py` | `src/configs/config.py`; `train_rsc()` is a function in `src/training/trainer.py` L1231; CLI is `scripts/run_experiment.py` |
| `src/models/gate.py`, `src/training/scheduler.py` | `src/gating/gate.py`; the scheduler is `LRReducer` in `src/training/losses.py` L12 |
| `src/data/tokenization`, `src/evaluation/metrics|evaluator` | `src/data/dataset.py`; `src/evaluation/{verifier,multi_dataset,multi_benchmark_runner,benchmark}.py` |
| "Phase 3: gate fine-tuning only" | Phases are 1 → 1.5 → 2A (off) → 2B gate-only → 2C joint gate+router → 2D gate re-tune (`trainer.py` L1231–1472) |

**Reviewed line by line:** `src/steering/steernet.py`, `src/gating/gate.py`, `src/routing/{router,rloo}.py`, `src/models/wrapper.py`, `src/configs/config.py`, `src/training/{trainer,losses,contrastive}.py`, `src/data/dataset.py`, `src/experiments/checkpointing.py`, `src/evaluation/{verifier,benchmark}.py`, `multi_dataset.py` L1–410, `multi_benchmark_runner.py` L1–260, `scripts/run_experiment.py`, the three `experiments/runs/*/EXPERIMENT_REPORT.md`, and the test inventory + key tests in `tests/test_modules.py`.
**Skimmed / not reviewed line by line:** `src/experiments/registry.py`, `src/visualization/*`, `scripts/audit_checkpoint_steering.py`, `scripts/check_architecture_diagrams.py`, `scripts/generate_architecture_diagrams.py`, `scripts/run_steernet_overfit_probe.py`, notebooks, `multi_benchmark_runner.py` L260–442, `multi_dataset.py` L410–527, and tests other than those listed. None of these sit on the training/eval path that determines accuracy.

**What I could and could not run.** The sandbox had no GPU and no PyTorch, and I did not have the failing run's logs. Every finding is therefore tagged:

* **[EXECUTED]** — the logic was reproduced and tested by running Python (LR schedule, answer extractor, paired bootstrap, RLOO baseline simulation).
* **[STATIC]** — read directly from the code; the claim follows from the source with no empirical assumption.
* **[HYPOTHESIS]** — plausible mechanism, **not confirmed**. A cheap diagnostic is supplied; do not apply the fix before the diagnostic confirms.

Fixes that touch torch code were syntax-checked (`ast.parse`) but **not executed**. Run the test additions in §8 before trusting them.

---

## Executive Summary

**ARS is optimised for a quantity that is not the target, validated by gates that do not measure the target, and evaluated by a harness that cannot reliably detect the target.** The three layers compound.

**1. Objective mismatch (root cause).** Every trainable component is fit to teacher-forced answer NLL on *human-written* GSM8K solutions rewritten into a `Step i: … #### N` template (`dataset.py` L58–207). Teacher forcing rewards copying the gold style (calculator annotations, template tokens), not producing a correct chain under the model's own autoregressive distribution. The authors' own config comment says it: a larger LR "produced large teacher-forced NLL gains but severe free-generation regressions" (`config.py` L68–72). Yet Phase-1 checkpoint selection is by validation NLL (`trainer.py` L748–762), and the router and gate are rewarded by the same NLL (`rloo.py` L219–285). The system is, by construction, climbing the proxy that the authors already know anti-correlates with accuracy. (ISSUE-1)

**2. Train regime ≠ deployment regime, and the router cannot repair it.** SteerNets are trained only as **K=1, α=1.0, one random layer at a time** (`trainer.py` L643–647; `router_warmup_max_k=1`, `config.py` L254). They are then frozen (`steer_finetune_epochs=0`). Deployment uses K up to 4, a learned α initialised at 0.5, K-conditioning inputs and a K^-0.5 route budget **none of which were ever seen in training** (ISSUE-3). The gate is then trained under four regularisers that pull α away from 1.0 and penalise steering of prompt tokens that the SteerNets *were* trained to steer (ISSUE-6). The repo's own run reports show validation NLL getting *worse* after gate training (0.64 → 0.78; 0.70 → 0.83), i.e. the gate stage destroys the Phase-1 solution, and the final checkpoint is anchored to that degraded Phase-2B value rather than to Phase 1 or the baseline (`trainer.py` L1426, L1451).

**3. Router and gate learn from in-sample rewards, and diversity is itself an objective.** The router bootstrap and every gate phase train on `train_data`, the very rows the SteerNets were fit on (`trainer.py` L1290, L1374–1452), so "steering wins" is inflated (the repo's reports log a 100 % win-rate vs base). On top, mutual-information diversity terms (λ=0.10 each) and *health audits that fail the run unless K=0 is chosen, ≥3 distinct K values appear, and ≥4 layers are active* (`config.py` L181–187) turn route diversity into an end in itself (ISSUE-7). A router that is rewarded for looking diverse will deploy diverse, partly arbitrary routes.

**4. Nothing ever compares the final composite system to the baseline on held-out generation.** The only generation gate (Phase-1 acceptance) tests a *single layer at K=1*, on n=64 items, selected and confirmed on the same items (winner's curse; ISSUE-11). The final audit checks diversity, not accuracy (`trainer.py` L426–500). A checkpoint that is worse than baseline is saved, loaded and reported as "ARS" with no veto (ISSUE-2).

**5. The measuring instrument is itself confounded.** Benchmark decoding uses `repetition_penalty=1.1` and `no_repeat_ngram_size=6` (`verifier.py` L312–313) — which bans re-emitting any 6-token span already in the prompt, such as the few-shot template ending `dollars.\nFinal answer:`; checkpoint selection uses *different* decoding (none of those flags, 200 tokens) (ISSUE-12). Three divergent answer extractors exist, one with a regex that matches the substring `is` inside `this`/`his` (ISSUE-13). With n=100, a system with **zero true effect** shows an apparent ≥ +10 pp gain ~7 % of the time (and ≥ −10 pp equally often) [EXECUTED, independent-arm model, worst case]; at n=64, ~12 %. The repo's historical "+12 pp / +21 pp" reports cannot be separated from these effects.

**Bottom line.** Fixes 1–4 and 10–13 are necessary to make the experiment *valid*. After them, if the paired 95 % CI on ≥300 held-out problems still straddles 0, the honest result is that this steering mechanism does not beat Phi-2 at this scale — and the code must ship the baseline (ISSUE-2 makes that automatic).

---

## Issue Index

| # | Title | Tag | File | Severity | Evidence |
|---|---|---|---|---|---|
| 1 | Teacher-forced NLL on human text is objective and selector | `[DESIGN-FLAW]` | `dataset.py`, `trainer.py` | **Critical** | STATIC |
| 2 | Final composite never gated against baseline; "baseline-safe" claim not persisted | `[DESIGN-FLAW]` `[BUG-CKPT]` | `trainer.py`, `checkpointing.py`, `multi_benchmark_runner.py` | **Critical** | STATIC |
| 3 | K≥2 routes, K-conditioning, route budget never trained; Phase 2A would not fix it | `[BUG-LOGIC]` `[BUG-TRAIN-EVAL]` | `trainer.py`, `wrapper.py`, `config.py` | **Critical** | STATIC |
| 4 | Router/gate rewards computed on SteerNet's own training rows | `[DESIGN-FLAW]` | `trainer.py` | High | STATIC |
| 5 | SteerNet: no identity at init, constant-norm output, scalar magnitude, hard-coded `/4.0` | `[DESIGN-FLAW]` `[CONFIG-DRIFT]` | `steernet.py`, `wrapper.py` | High | STATIC |
| 6 | Gate stage fights the SteerNet it wraps | `[BUG-TRAIN-EVAL]` `[DESIGN-FLAW]` | `config.py`, `contrastive.py`, `losses.py` | High | STATIC + repo reports |
| 7 | Diversity is rewarded and audited as an end | `[DESIGN-FLAW]` | `config.py`, `rloo.py`, `trainer.py` | High | STATIC |
| 8 | RLOO: mixed-K baseline (variance), selection by training loss | `[DESIGN-FLAW]` | `rloo.py`, `trainer.py` | Medium | EXECUTED |
| 9 | Router features / first-token steering dominated by attention sink | `[DESIGN-FLAW]` `[NUMERICAL]` | `router.py`, `wrapper.py` | Medium | HYPOTHESIS |
| 10 | `LRReducer` warmup is a no-op; plateau restarts the cosine | `[BUG-LOGIC]` `[CONFIG-DRIFT]` | `losses.py` L12–61 | Medium | EXECUTED |
| 11 | Phase-1 acceptance: winner's curse, underpowered, selection = confirmation set | `[DESIGN-FLAW]` | `trainer.py` L214–423 | High | EXECUTED (power) |
| 12 | Decode flags confound results; selection decode ≠ final decode | `[BUG-TRAIN-EVAL]` | `verifier.py` | High | STATIC (mechanism HYPOTHESIS) |
| 13 | Divergent answer extractors; `is` substring regex | `[BUG-SILENT]` | `multi_dataset.py`, `verifier.py` | Medium | EXECUTED |
| 14 | Invariance loss has no gradient and wrong output indexing | `[DEAD-CODE]` `[BUG-GRAD]` | `losses.py`, `trainer.py` | Low | STATIC |
| 15 | Stale checkpoints silently reloaded | `[BUG-CKPT]` | `trainer.py` | Medium | STATIC |
| 16 | Config drift and dead code | `[CONFIG-DRIFT]` `[DEAD-CODE]` | several | Low | EXECUTED (scan) |
| 17 | Gate streaming cache inconsistent; dirac init shifts a token; test passes only at init | `[BUG-TRAIN-EVAL]` | `gate.py`, `test_modules.py` | Low | STATIC |
| 18 | Diagnostic buffers saved into checkpoints | `[BUG-CKPT]` | `router.py` | Low | STATIC |

---

## Detailed Issues

### ISSUE-1: Teacher-forced NLL on human text is the objective and the selector
**Tag:** `[DESIGN-FLAW]`  **Evidence:** STATIC
**Files:** `src/data/dataset.py` L58–207; `src/training/trainer.py` L59–66, L748–762; `src/routing/rloo.py` L219–285; `src/configs/config.py` L68–72

**Problem:**
SteerNets are trained with `_answer_aware_loss` (token NLL of the human GSM8K solution re-templated as `Step i: …\n#### N`, with the `<<a*b=c>>` calculator annotations left in by `split_answer`). Phase-1 early stopping and "best state" use validation NLL:
```python
# trainer.py L752-762
improved = "✅" if val_loss < best_val_p1 - cfg.early_stop_delta else "  "
if val_loss < best_val_p1 - cfg.early_stop_delta:
    best_val_p1 = val_loss
    ...
```
The router reward (`rloo.py` L219–285) and gate loss (`trainer.py` L1003–1005) are the same quantity.

**Root Cause:**
Teacher forcing conditions every token on the *gold* prefix. NLL gains are dominated by predictable template/annotation tokens and by memorising the style of human text; they say nothing about whether the model, conditioned on its *own* previous tokens, reaches the right number. The authors observed exactly this regression (`config.py` L68–72) and responded by lowering the LR instead of changing the target. Selecting the checkpoint with the lowest val NLL then picks the most style-fitted, not the most accurate, adapter. Because steering is bounded and the base model already follows the `Question/Answer: Let's think step by step` prompt, there is little *style* headroom, so the net effect on accuracy is ≈0 or negative, matching the symptom.

**Fix:** train on the frozen model's **own correct chains** (rejection-sampling fine-tuning / STaR). The target is then in-distribution for the backbone, teacher-forced loss and free generation optimise the same distribution, and the steering modules learn to move probability mass toward correct modes the model can already reach. Add `src/data/rft.py` (uses `src/evaluation/answers.py` from ISSUE-13):

```python
"""Rejection-sampled self-distillation (RFT/STaR) targets for ARS."""
from typing import Any, Dict, List, Optional

import torch

from src.evaluation.answers import extract_numeric_answer, has_answer_marker, truncate_at_answer
from src.evaluation.multi_dataset import build_benchmark_prompt


def _gold(value) -> float:
    return float(str(value).replace(",", "").strip())


@torch.no_grad()
def build_rft_items(
    model, tokenizer, rows, device, samples_per_question: int = 4, temperature: float = 0.7,
    top_p: float = 0.95, max_new_tokens: int = 256, max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Sample from the UNSTEERED backbone; keep one verified-correct chain per question."""
    was_training = model.training
    old_override = model._routing_override
    model.eval()
    model.set_routing_override("none")
    kept: List[Dict[str, Any]] = []
    try:
        rows = list(rows)[: (max_rows or len(rows))]
        for row in rows:
            gold = _gold(row["final_num"])
            prompt = build_benchmark_prompt("gsm8k", row["question"], n_shot=0)
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            plen = enc["input_ids"].shape[1]
            out = model.generate(
                **enc, do_sample=True, temperature=temperature, top_p=top_p,
                num_return_sequences=samples_per_question, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            for seq in out:
                text = truncate_at_answer(tokenizer.decode(seq[plen:], skip_special_tokens=True).strip())
                pred = extract_numeric_answer(text)
                if has_answer_marker(text) and pred is not None and abs(pred - gold) <= 1e-2:
                    kept.append({"prompt": prompt, "completion": text,
                                 "question": row["question"], "final_num": row["final_num"]})
                    break
    finally:
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])
        model.train(was_training)
    return kept


def rft_items_to_features(tokenizer, items, max_len: int) -> List[Dict[str, Any]]:
    """Tokenise as prompt ids + completion ids + EOS, exactly as generation would produce them."""
    feats = []
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    for item in items:
        p_ids = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(item["completion"], add_special_tokens=False)["input_ids"] + [eos_id]
        ids = (p_ids + c_ids)[:max_len]
        labels = ([-100] * len(p_ids) + c_ids)[:max_len]
        attn = [1] * len(ids)
        pad = max_len - len(ids)
        ids, labels, attn = ids + [pad_id] * pad, labels + [-100] * pad, attn + [0] * pad
        if sum(1 for t in labels if t != -100) < 4:
            continue
        feats.append({
            "input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels), "prompt_shield_mask": torch.zeros(max_len, dtype=torch.long),
            "question": item["question"], "steps_list": [], "final_num": item["final_num"],
        })
    return feats
```
Config additions (`RSCConfig`): `rft_samples_per_question: int = 4`, `rft_temperature: float = 0.7`, `rft_max_questions: int = 3000`, and set `answer_tail_weight: float = 1.0` (the tail upweighting only existed to paper over teacher-forcing; with self-generated targets it is unnecessary). Wiring is in ISSUE-4's `train_rsc` block. Keep the old gold-CoT path available behind `cfg.train_target = "gold"` only as an ablation.

**Impact on Project Goal:** This is the largest single lever. It removes the "good NLL, bad generation" mismatch at its source and makes Phase-1/router/gate validation losses meaningful. Expected effect size is unknown; the success criterion is the paired test in ISSUE-2, not NLL.

---

### ISSUE-2: Final composite is never compared with baseline; the "baseline-safe" checkpoint is not baseline-safe
**Tag:** `[DESIGN-FLAW]` `[BUG-CKPT]`  **Evidence:** STATIC
**Files:** `trainer.py` L426–500, L574–628, L1257–1287, L1391–1394, L1426, L1451; `checkpointing.py` L121–187; `multi_benchmark_runner.py` L74–78; `scripts/run_experiment.py` L94–97

**Problem:**
1. `audit_ars_intervention_health` checks K-support, entropy, active layers, gate std — never accuracy. `report["passed"]` can be `True` for a model that is far worse than baseline.
2. `init_best_val=best_p2b_val` (L1426) and `init_best_val=best_val_p2` (L1451) anchor later phases to **Phase 2B's** validation NLL, which the repo's reports show is *worse* than Phase 1 (0.7797 vs 0.6392; 0.8348 vs 0.7042). Nothing anchors to Phase 1 or to the unsteered baseline.
3. The Phase-1-failure path claims a baseline deployment:
```python
# trainer.py L1273-1276
model.set_routing_override("none")
model.set_gate_freeze(False)
save_ars_checkpoint(model, cfg.phase2_save_path, phase="phase1-failed-baseline-deployment", cfg=cfg)
```
`_routing_override` is runtime state; it is **not in the saved tensors or manifest**. Any evaluator that reloads this file and runs the `ars` condition calls `set_routing_override(None)` (`multi_benchmark_runner.py` L75) and gets the full learned system. (It happens to equal baseline today only because `k_head` is zero-initialised and argmax of equal logits is K=0 — an accident.)

**Root Cause:** acceptance is defined by *diagnostic shape*, not by outcome, and deployment mode is not part of the artifact. Hence the pipeline can, and per the reported symptom does, emit a worse-than-baseline "ARS".

**Fix:** add a final, baseline-anchored, paired decision whose result is stored in the checkpoint manifest and enforced by the loader and runners. New `src/evaluation/deployment.py`:

```python
"""Baseline-anchored deployment decision (paired, held-out, persisted)."""
from typing import Any, Dict, Sequence

import numpy as np
import torch

from src.evaluation.multi_dataset import build_benchmark_prompt, evaluate_sample_answer
from src.evaluation.verifier import generate_response
from src.experiments.checkpointing import save_ars_checkpoint


def paired_delta_ci(correct_base, correct_ars, n_boot: int = 4000, seed: int = 0, alpha: float = 0.05):
    a = np.asarray(correct_base, dtype=float)
    b = np.asarray(correct_ars, dtype=float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def _flags(model, tokenizer, rows, device, max_new_tokens: int) -> np.ndarray:
    out = []
    for row in rows:
        prompt = build_benchmark_prompt("gsm8k", row["question"], n_shot=0)
        resp = generate_response(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
        out.append(int(evaluate_sample_answer("gsm8k", resp, row["final_num"])["correct"]))
    return np.asarray(out)


def evaluate_paired(model, tokenizer, rows: Sequence[Dict[str, Any]], device, max_new_tokens: int = 256):
    prev_override, prev_mode = model._routing_override, getattr(model, "_router_inference_mode", "argmax")
    model.eval()
    model.set_router_inference_mode("argmax")
    try:
        model.set_routing_override("none")
        model.set_gate_freeze(False)
        base = _flags(model, tokenizer, rows, device, max_new_tokens)
        model.set_routing_override(None)
        model.set_gate_freeze(False)
        ars = _flags(model, tokenizer, rows, device, max_new_tokens)
    finally:
        if prev_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(prev_override[0], prev_override[1])
        model.set_router_inference_mode(prev_mode)
    return base, ars


def finalize_deployment(model, tokenizer, confirm_rows, cfg, device, checkpoint_path: str,
                        min_examples: int = 200, n_boot: int = 4000) -> Dict[str, Any]:
    """ARS is enabled only if the paired 95% CI lower bound on held-out generation is > 0."""
    rows = list(confirm_rows)
    if len(rows) < min_examples:
        raise ValueError(f"deployment decision needs >= {min_examples} held-out rows, got {len(rows)}")
    base, ars = evaluate_paired(model, tokenizer, rows, device)
    delta, lo, hi = paired_delta_ci(base, ars, n_boot=n_boot, seed=cfg.seed)
    mode = "ars" if lo > 0.0 else "baseline"
    meta = {"deployment_mode": mode, "paired_delta": delta, "ci95": [lo, hi], "n": len(rows),
            "baseline_accuracy": float(base.mean()), "ars_accuracy": float(ars.mean())}
    save_ars_checkpoint(model, checkpoint_path, phase="final-deployment", cfg=cfg, extra_metadata=meta)
    model.deployment_disabled = (mode == "baseline")
    print(f"  Deployment decision: {mode.upper()}  Δ={delta*100:+.1f}pp  CI95=[{lo*100:+.1f}, {hi*100:+.1f}]  n={len(rows)}")
    return meta
```
Loader — append to `load_ars_checkpoint_strict` (`checkpointing.py`, immediately before its final `return`), and add `"steer_exclude_first_token"`, `"router_warmup_max_k"` to `CHECKPOINT_CONFIG_FIELDS`:
```python
    extra = (manifest or {}).get("extra") or {}
    if hasattr(model, "deployment_disabled"):
        model.deployment_disabled = (extra.get("deployment_mode", "ars") == "baseline")
```
Wrapper — in `Phi2WithRSC.__init__` (after L65) add `self.deployment_disabled = False`. Runners — make the `ars` condition honour it:
```python
# multi_benchmark_runner.py, replace the `elif cond == "ars":` body (L74-78)
    elif cond == "ars":
        if hasattr(model, "set_routing_override"):
            model.set_routing_override("none" if getattr(model, "deployment_disabled", False) else None)
        if hasattr(model, "set_gate_freeze"):
            model.set_gate_freeze(False)
```
and the same one-line change for `args.eval_stage == "ars"` at `scripts/run_experiment.py` L94–97. Replace the false claim at `trainer.py` L1273–1276 by setting `model.deployment_disabled = True` before `save_ars_checkpoint(..., extra_metadata={"deployment_mode": "baseline"})`. Finally call `finalize_deployment` at the end of `train_rsc` (ISSUE-4 block) and **drop** `audit_ars_intervention_health` as a pass/fail criterion (keep it as telemetry).

**Impact on Project Goal:** Makes "ARS < baseline" impossible to ship silently and makes the real answer observable (the CI). It is a safety net, not a substitute for ISSUE-1/3/4.

---

### ISSUE-3: K≥2 routes, K-conditioning and the route budget are never trained
**Tag:** `[BUG-LOGIC]` `[BUG-TRAIN-EVAL]`  **Evidence:** STATIC
**Files:** `trainer.py` L643–647, L1325–1352; `wrapper.py` L375–386; `config.py` L233, L254

**Problem:**
Phase 1 forces warmup routing with `router_warmup_max_k=1`:
```python
# wrapper.py L375-386
if self._router_warmup:
    warmup_max_k = int(getattr(self.cfg, "router_warmup_max_k", 1))
    width = 1 + (self._warmup_cursor % max(1, min(K_max, warmup_max_k)))   # always 1
    stride = max(1, n_candidates // width)
    ...
```
So `_active_k_realized == 1` always: `k_proj` only ever sees `1/4`, the `K**-0.5` budget (L162–164) is always 1.0, and layers are never co-active. SteerNets are then frozen (`steer_finetune_epochs=0`, `config.py` L233). Even if Phase 2A is enabled, it calls `model.set_router_warmup(True)` (L1331) **without raising `router_warmup_max_k`**, so it also trains only K=1; and the warmup subsets are fixed-stride sets (width 2 → only `{p, p+4}`), not the arbitrary subsets the router samples.

**Root Cause:** the router is asked to choose among *cooperative multi-layer interventions* that no SteerNet has ever been optimised for. At deployment `k_proj(K/4)` is an extrapolated input (`x + w·K/4 + b` with `w` fit at 0.25 only), the per-layer deltas are scaled by 0.5–0.71, and up to four independently-trained perturbations stack. The router can only pick among these; with SteerNets frozen it cannot make any combination work. Counterfactual rewards for K≥2 then reflect *accidental* interference.

**Fix:** train SteerNets under the same route distribution used downstream — uniform random width in `[1, K_max]` and a uniform random subset of that width, per example. Replace the warmup block in `_prepare_routes` (wrapper L375–386):
```python
        if self._router_warmup:
            warmup_max_k = max(1, min(K_max, int(getattr(self.cfg, "router_warmup_max_k", 1))))
            selected_matrix = torch.zeros(h.shape[0], n_candidates, dtype=torch.bool, device=h.device)
            if getattr(self.cfg, "router_warmup_random_subsets", True):
                for b in range(h.shape[0]):
                    width = int(torch.randint(1, warmup_max_k + 1, (1,)).item())
                    chosen = torch.randperm(n_candidates, device=h.device)[:width]
                    selected_matrix[b, chosen] = True
            else:  # legacy deterministic rotation
                width = 1 + (self._warmup_cursor % warmup_max_k)
                stride = max(1, n_candidates // width)
                pos = self._warmup_width_visits[width - 1]
                forced = {self.target_layers[(pos + j * stride) % n_candidates] for j in range(width)}
                self._warmup_width_visits[width - 1] += 1
                row = torch.tensor([idx in forced for idx in self.target_layers], device=h.device, dtype=torch.bool)
                selected_matrix = row.unsqueeze(0).expand(h.shape[0], -1).clone()
            self._warmup_cursor += 1
            self._desired_k = selected_matrix.long().sum(dim=-1).clamp(min=1, max=K_max)
```
Config: `router_warmup_max_k: int = 4`, `router_warmup_random_subsets: bool = True`. In the Phase-2A block (trainer L1330–1331) make the intent explicit:
```python
        model.set_gate_freeze(True, 1.0)
        cfg.router_warmup_max_k = cfg.max_active_layers
        model.set_router_warmup(True)
```
Because Phase-1 SteerNets are now trained inside combinations, the solo "independent capacity" acceptance gate is the wrong test: set `skip_adaptive_phases_when_phase1_fails=False` and `phase1_min_validation_accuracy_gain=0.0`; keep the solo probe as telemetry; the decision is ISSUE-2.

**Impact on Project Goal:** Without this, any K≥2 route is out-of-distribution by construction, and the router's whole premise (cooperation) is untestable.

---

### ISSUE-4: Router and gate rewards are computed on the SteerNet's own training rows
**Tag:** `[DESIGN-FLAW]`  **Evidence:** STATIC (+ repo reports: "Win Rate vs Base = 100.0 %")
**Files:** `trainer.py` L1290, L1374–1382, L1418–1427, L1443–1452

**Problem:** `train_router_bootstrap(model, train_data, …)` and every Phase-2 stage iterate `train_data`, which Phase 1 just fit. For those rows the SteerNets' answer NLL is optimistically low, so every counterfactual arm "beats" the unsteered anchor.

**Root Cause:** reward = in-sample generalisation gap. The K-head and layer policy learn "steering helps, more is better" from rows where that is trivially true; on unseen test rows it is not.

**Fix:** disjoint data for steering and for routing/gating, plus RFT targets (ISSUE-1) and the final decision (ISSUE-2). Replace the body of `train_rsc` from the first line through the Phase-1.5 block (trainer L1243–1292) with:
```python
def split_steer_router(rows, router_fraction: float = 0.2, seed: int = 0):
    n = len(rows)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    cut = max(1, int(n * router_fraction))
    router_idx, steer_idx = sorted(order[:cut]), sorted(order[cut:])
    if hasattr(rows, "select"):
        return rows.select(steer_idx), rows.select(router_idx)
    return [rows[i] for i in steer_idx], [rows[i] for i in router_idx]


def _purge_ckpt(path: str):
    for p in (path, path + ".manifest.json"):
        if os.path.exists(p):
            os.remove(p)


def train_rsc(model, tokenizer, train_data, eval_data, cfg: RSCConfig, device: torch.device):
    if getattr(cfg, "curriculum_mode", "full") == "streamlined":
        return train_streamlined_ars(model, tokenizer, train_data, eval_data, cfg, device)

    from src.data.rft import build_rft_items, rft_items_to_features
    from src.evaluation.deployment import finalize_deployment

    logs = {}
    for path in (cfg.gate_only_save_path, cfg.steer_finetune_save_path, cfg.phase2_save_path):
        _purge_ckpt(path)                                           # ISSUE-15

    steer_rows, router_rows = split_steer_router(train_data, 0.2, cfg.seed)
    if getattr(cfg, "train_target", "rft") == "rft":
        items = build_rft_items(
            model, tokenizer, steer_rows, device,
            samples_per_question=cfg.rft_samples_per_question, temperature=cfg.rft_temperature,
            max_rows=cfg.rft_max_questions,
        )
        steer_data = rft_items_to_features(tokenizer, items, cfg.max_tok_len)
        router_items = build_rft_items(model, tokenizer, router_rows, device,
                                       samples_per_question=cfg.rft_samples_per_question,
                                       temperature=cfg.rft_temperature, max_rows=cfg.rft_max_questions)
        router_data = rft_items_to_features(tokenizer, router_items, cfg.max_tok_len)
        print(f"  RFT targets: steer={len(steer_data)}  router/gate={len(router_data)}")
    else:
        steer_data, router_data = steer_rows, router_rows

    # PHASE 1 — SteerNets on route-matched random subsets (ISSUE-3)
    best_state_p1, best_val_p1, p1_logs = train_phase1_steernet(model, steer_data, eval_data, cfg, device)
    logs.update(p1_logs)
    logs["phase1_acceptance"] = phase1_steernet_acceptance(model, tokenizer, steer_data, eval_data, cfg, device)
    # (the capacity gate block L1252-1287 stays but is non-fatal: cfg.skip_adaptive_phases_when_phase1_fails=False)

    # PHASE 1.5 — router on held-out-from-SteerNet rows
    _, best_p15 = train_router_bootstrap(model, router_data, eval_data, cfg, device)
    logs["router_bootstrap_loss"] = best_p15
    logs["router_bootstrap_preflight"] = audit_router_policy_health(model, eval_data, cfg, device)
    # ... Phase 2A/2B/2C/2D blocks unchanged EXCEPT: every `train_data` argument -> `router_data`
```
In the remaining 2A/2B/2C/2D calls to `run_staged_phase`, pass `router_data` for 2B/2C/2D and `steer_data` for 2A. After the final `audit_ars_intervention_health` call (L1459) add:
```python
    confirm_rows = [eval_data[i] for i in range(len(eval_data) - cfg.deployment_examples, len(eval_data))]
    logs["deployment"] = finalize_deployment(model, tokenizer, confirm_rows, cfg, device,
                                             cfg.phase2_save_path, min_examples=min(200, cfg.deployment_examples))
```
with `deployment_examples: int = 300`. Raise `val_split` to 0.15 (≈1,050 val rows) and, at the top of `train_rsc`, hold the last `cfg.deployment_examples` rows of `eval_data` out of every training-time use: `eval_sel = [eval_data[i] for i in range(len(eval_data) - cfg.deployment_examples)]`, and pass `eval_sel` (not `eval_data`) to all `train_*`, `phase1_steernet_acceptance` and audit calls. Those rows are touched exactly once, by `finalize_deployment`.

**Impact on Project Goal:** removes the optimism that makes the router/gate favour heavy steering.

---

### ISSUE-5: SteerNet has no identity at init, emits a constant-norm vector, and has one scalar magnitude
**Tag:** `[DESIGN-FLAW]` `[CONFIG-DRIFT]`  **Evidence:** STATIC (algebra from the code)
**Files:** `src/steering/steernet.py` L50–70, L58; `src/models/wrapper.py` L102–111; `config.py` L34

**Problem:**
`config.py` defaults to `steer_magnitude_mode="decoupled"`:
```python
# steernet.py L64-70
d_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-7
direction = delta / d_norm
mag = torch.exp(self.log_magnitude).clamp(max=self.max_magnitude)   # one scalar per layer
delta = direction * mag * self.scale.to(x.dtype)
```
With `log_magnitude=0` and `scale = lora_alpha/rank = 32/16 = 2`, **every steered token receives a vector of norm exactly 2.0 at initialisation, whatever the (≈1e-3) pre-normalisation signal** — no identity at init (contrast LoRA's zero `B`). The orthogonal projection and the 4 % relative cap only bind afterwards. `log_magnitude` is a single scalar per layer, so the net cannot choose *how much* to steer a given token; in Phase 1 the gate is pinned at 1.0 (`trainer.py` L643) and in the non-joint path α is detached (`wrapper.py` L195), so nothing else can either. Also `active_k / 4.0` (L58) hard-codes K_max=4 although `max_active_layers` is configurable.

**Root Cause:** the architecture must learn a direction that is harmless for the majority of tokens (formatting, copied numbers) because it cannot be switched off per token, starting from a non-zero perturbation. This is the opposite of a minimal-intervention steering design and inflates the damage visible in free generation.

**Fix:** LoRA-style zero-initialised `up` (exact identity at init), default `direct` mode, a per-token bounded magnitude if `decoupled` is kept, a smooth normaliser (no 1/ε gradient blow-up at zero), and `k_max` from config. Full replacement of `src/steering/steernet.py`:
```python
import math

import torch
import torch.nn as nn


class RSCSteerNet(nn.Module):
    """K-aware low-rank residual steering MLP; identity at initialisation."""

    def __init__(self, hidden_dim: int, rank: int, lora_alpha: float, dropout: float = 0.1,
                 magnitude_mode: str = "direct", max_magnitude: float = 2.0,
                 max_relative_norm: float = 0.04, orthogonal_projection: bool = False, k_max: int = 4):
        super().__init__()
        if magnitude_mode not in {"direct", "decoupled"}:
            raise ValueError("magnitude_mode must be 'direct' or 'decoupled'")
        self.magnitude_mode = magnitude_mode
        self.max_magnitude = float(max_magnitude)
        self.max_relative_norm = float(max_relative_norm)
        self.orthogonal_projection = bool(orthogonal_projection)
        self.k_max = max(1, int(k_max))

        self.norm = nn.LayerNorm(hidden_dim)
        self.down1 = nn.Linear(hidden_dim, rank, bias=False)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.mid = nn.Linear(rank, rank, bias=False)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.up = nn.Linear(rank, hidden_dim, bias=False)

        self.k_proj = nn.Linear(1, rank)
        nn.init.zeros_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        self.register_buffer("scale", torch.tensor(lora_alpha / rank, dtype=torch.float32))

        if magnitude_mode == "decoupled":
            self.mag_head = nn.Linear(rank, 1)           # per-token bounded magnitude
            nn.init.zeros_(self.mag_head.weight)
            nn.init.constant_(self.mag_head.bias, [−6.0→deprecated])  # sigmoid(-6) ~ 0.0025 -> near-identity

        # down/mid keep PyTorch's default init so activations are O(1); `up` = 0 gives delta == 0 at init.
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor, active_k: torch.Tensor = None) -> torch.Tensor:
        h_flt = h.float()
        x = self.drop1(self.act1(self.down1(self.norm(h_flt))))
        if active_k is not None:
            k_norm = (active_k.to(dtype=x.dtype) / float(self.k_max)).view(-1, 1, 1)
            x = x + self.k_proj(k_norm)
        x = x + self.drop2(self.act2(self.mid(x)))
        raw = self.up(x)

        if self.magnitude_mode == "decoupled":
            direction = raw / torch.sqrt(raw.square().sum(dim=-1, keepdim=True) + 1e-4)   # defined and smooth at 0
            mag = self.max_magnitude * torch.sigmoid(self.mag_head(x))                    # (B, T, 1)
            delta = direction * mag * self.scale.to(x.dtype)
        else:
            delta = raw * self.scale.to(x.dtype)

        if self.max_relative_norm > 0:
            h_norm = torch.norm(h_flt, dim=-1, keepdim=True) + 1e-7
            delta_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-7
            delta = delta * torch.clamp(self.max_relative_norm * h_norm / delta_norm, max=1.0)

        if self.orthogonal_projection:
            h_sq = torch.sum(h_flt * h_flt, dim=-1, keepdim=True) + 1e-7
            delta = delta - (torch.sum(delta * h_flt, dim=-1, keepdim=True) / h_sq) * h_flt
        return delta
```
Wrapper (`wrapper.py` L102–111): pass `k_max=self.cfg.max_active_layers` to `RSCSteerNet(...)`. Config: `steer_magnitude_mode: str = "direct"`. Old checkpoints are incompatible (init and, for `decoupled`, new keys): retrain.

**Impact on Project Goal:** a freshly initialised intervention is now a no-op, so early training can only improve on baseline, and per-token magnitude lets the network leave non-reasoning tokens alone.

---

### ISSUE-6: The gate stage fights the SteerNet it wraps
**Tag:** `[BUG-TRAIN-EVAL]` `[DESIGN-FLAW]`  **Evidence:** STATIC + repo reports (historical commits)
**Files:** `config.py` L131, L141–144; `trainer.py` L1009–1026; `losses.py` L128–194; `contrastive.py` L10–117

**Problem:**
* SteerNets are trained at **α=1.0 on every unshielded token, prompt included** (`wrapper.py` L165, `steer_mask = mask`). The gate starts at `gate_init_alpha=0.5` (half strength) and is then trained with four regularisers active simultaneously: L1 (0.001), anti-saturation with per-layer ceilings 0.55–0.85 (0.25), "usefulness" contrastive loss (0.30), diversity (0.5).
* The contrastive loss forces α on **prompt tokens below 0.3** (`contrastive.py` L104–106) although the frozen SteerNets influence the answer *through* steered prompt residuals (they were trained that way).
* Its token classes are wrong: the "math" regex `^[\d\+\-\*\/\=\$\%\.]+$` (L15) includes `.`, `-`, `$` (every sentence end counts as "math"), while `####` and newline tokens are stripped to empty by `lstrip("Ġ")…lstrip("#")` (L22–24) and dropped, so the real answer marker is in neither class.
* The repo's own reports: Phase 2B validation NLL **0.7797 vs Phase-1 0.6392** and **0.8348 vs 0.7042**.

**Root Cause:** the regularisers encode a hand-written prior (alphas should be low, varied, and token-class-ordered) that directly conflicts with the function the frozen SteerNets were fit to. With task gradient only through `_joint_grad_mode`, the prior wins on most tokens, degrading the intervention.

**Fix:** make the task loss primary and regularisers opt-in; start the gate at near-identity relative to Phase 1. Config replacement (`RSCConfig`):
```python
    gate_init_alpha: float = 0.95
    lambda_l1_gate: float = 0.0
    lambda_gate_antisat: float = 0.0
    lambda_gate_usefulness: float = 0.0
    lambda_gate_diversity: float = 0.0
```
If the contrastive term is retained as an ablation, replace `build_contrast_id_cache` (`contrastive.py` L10–47) by decoded-string classification:
```python
def build_contrast_id_cache(tokenizer):
    global _CONTRAST_MATH_IDS, _CONTRAST_FORMAT_IDS, _CONTRAST_DISCOURSE_IDS
    if _CONTRAST_MATH_IDS is not None:
        return
    num_re = re.compile(r"^\$?\d[\d,]*(\.\d+)?%?$|^[+\-*/=]$")
    fmt = {"Question", "Answer", ":", "#", "##", "###", "####"}
    disc_re = re.compile(r"^(therefore|thus|hence|because|since|so|total|each|remaining|left|more|fewer)$", re.I)
    math_ids, fmt_ids, disc_ids = set(), set(), set()
    for tok_id in range(len(tokenizer)):
        raw = tokenizer.decode([tok_id])
        s = raw.strip()
        if "\n" in raw or s in fmt:
            fmt_ids.add(tok_id)
        elif num_re.match(s):
            math_ids.add(tok_id)
        elif disc_re.match(s):
            disc_ids.add(tok_id)
    _CONTRAST_MATH_IDS, _CONTRAST_FORMAT_IDS, _CONTRAST_DISCOURSE_IDS = math_ids, fmt_ids, disc_ids
```
and delete the `prompt_mask` ceiling (L104–106 of `compute_contrastive_gate_loss`).

**Impact on Project Goal:** Phase 2B stops degrading the Phase-1/2A solution; any alpha sparsity must now be earned by task loss.

---

### ISSUE-7: Route diversity is rewarded and audited as an end in itself
**Tag:** `[DESIGN-FLAW]`  **Evidence:** STATIC
**Files:** `config.py` L181–187, L214–216; `rloo.py` L113–132, L337–349; `trainer.py` L459–480, L539–548, L1294–1312

**Problem:** `lambda_router_k_mi = lambda_router_layer_mi = 0.10` reward the router for batch-level K/layer diversity (`rloo.py` L365–376), and the audits mark the run FAILED unless K=0 occurs, ≥3 distinct K occur, K-entropy ≥ 0.25, ≥4 layers are active and layer entropy ≥ 0.45. The git history (`fix(router): train and verify deterministic route diversity`) shows the pipeline was iterated until it satisfied these.

**Root Cause (Goodhart):** if the best policy is "mostly K=1 on one layer, K=0 for hard prompts", a healthy router *fails* the audit and the training pressure pushes it toward arbitrary variety. Layer choice becomes partly noise injected into a model that is frozen and was never trained for most subsets (ISSUE-3).

**Fix:** keep diversity as telemetry, never as loss or pass/fail. Config replacement:
```python
    lambda_router_k_mi: float = 0.0
    lambda_router_layer_mi: float = 0.0
    router_require_eval_k0: bool = False
    router_min_eval_k_support: int = 1
    router_min_eval_k_entropy: float = 0.0
    router_min_eval_active_layers: int = 1
    router_min_eval_layer_entropy: float = 0.0
    gate_min_eval_sequence_std: float = 0.0
    gate_max_eval_high_saturation_pct: float = 100.0
    skip_adaptive_phases_when_router_fails: bool = False
```
(With those values both audit functions pass trivially and keep logging; accuracy acceptance moves to ISSUE-2.) Collapse onto a single route is then a legitimate finding, reported by `routing_k_distribution()`.

**Impact on Project Goal:** removes a force that makes the deployed policy deliberately heterogeneous regardless of utility.

---

### ISSUE-8: RLOO uses a mixed-K baseline; router checkpoint chosen by training loss
**Tag:** `[DESIGN-FLAW]`  **Evidence:** EXECUTED (simulation)
**Files:** `rloo.py` L206–215, L251, L285–294, L361–371; `trainer.py` L884–890

**Problem:**
```python
# rloo.py L291-294
reward_sum = action_rewards.sum(dim=1, keepdim=True) + reward_0.unsqueeze(1)
advantages = action_rewards - (reward_sum - action_rewards) / float(K_max)
policy_loss = -(subset_logprobs_t * advantages).mean()
```
One subset is drawn per K and its baseline is the mean reward of the *other K arms and the unsteered anchor*. Also `k_center_loss` uses `mean_sampled_k = k_actions.float().mean()` — a constant (L361–362), so it is dead code. In `train_router_bootstrap`, the saved router is the epoch with the lowest **training loss** (L884–887), a sum dominated by entropy/MI/preference terms, not by reward.

**Root Cause / measurement.** This baseline is *unbiased* (it is independent of the arm's own subset), so this is a variance problem, not a bias bug; it injects the between-K reward gap into every advantage as pure variance. Simulation (40 000 trials, true layer effect +0.05, reward rising 0.04/K, noise 0.03) [EXECUTED]:

| estimator | mean gradient signal | std | SNR |
|---|---|---|---|
| current (1 draw/K, mixed-K baseline) | 0.00980 | 0.01610 | 0.61 |
| same-K leave-one-out, 2 draws/K | 0.00981 | 0.00893 | 1.10 |
| same-K leave-one-out, 3 draws/K | 0.00982 | 0.00610 | 1.61 |

Same expected gradient, ≈1.8× lower std at 2 draws/K (cost: 2× counterfactual forwards).

**Fix:** same-K leave-one-out with `router_samples_per_k` (default 2) draws, K-head distilled from the per-K mean reward, dead terms removed. Full replacement of `compute_router_rloo_loss` in `src/routing/rloo.py` (syntax-checked; torch behaviour not executed):

```python
def compute_router_rloo_loss(model, ids: torch.Tensor, mask: torch.Tensor,
                              labels: torch.Tensor, cfg, prompt_mask: torch.Tensor = None,
                              shield_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """RLOO subset selection loss for training cooperative router policy with leave-one-out advantage."""
    if prompt_mask is None:
        prompt_mask = mask.bool() & labels.eq(-100)
    # The policy state must come from *this* example.  In bootstrap training
    # there is otherwise no ordinary task forward before this call, so relying
    # on _last_pooled trains current rewards against the previous example.
    # Populate the detached router cache with a fresh, same-example prefill.
    old_override = model._routing_override
    # Counterfactual reward is a comparison between routes.  It must not draw
    # a different dropout mask for each arm, otherwise RLOO learns dropout
    # noise instead of route utility.  Preserve every module's mode because
    # this function is called from a mixed-mode staged curriculum.
    module_modes = {module: module.training for module in model.modules()}
    try:
        # Train exactly the deterministic utility function that deployment
        # uses.  A training-only layer-dropout mask otherwise creates apparent
        # route diversity which disappears under eval/argmax.
        model.router.eval()
        model.set_routing_override(None)
        model.set_router_context_mask(prompt_mask)
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield_mask)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=ids.is_cuda):
                _ = model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])

    u_fresh, k_logits_fresh = model.router.recompute_from_cache()
    synergy = getattr(model.router, "_last_synergy_matrix", None)
    K_max = model.cfg.max_active_layers

    # K is a tiny discrete action space (0..4), so it is both cheaper in
    # variance and more scientifically defensible to score every non-zero K
    # than to hope two random samples cover it.  This is *not* a fixed route:
    # each K still samples a prompt-conditioned Plackett-Luce subset, and the
    # K head is supervised only by these counterfactual rewards.
    # ISSUE-8: draw >= 2 subsets per K so every sample has a SAME-K leave-one-out baseline.
    B = ids.shape[0]
    n_per_k = max(2, int(getattr(cfg, "router_samples_per_k", 2)))
    k_actions = torch.arange(1, K_max + 1, device=ids.device, dtype=torch.long)
    k_actions = k_actions.unsqueeze(0).expand(B, -1)                      # (B, K_max)
    sampled = []                                                          # sampled[k_idx][s] = (mask, logprob)
    for k_idx in range(K_max):
        sampled.append([
            sample_subset_plackett_luce(u_fresh, k_actions[:, k_idx], synergy_matrix=synergy)
            for _ in range(n_per_k)
        ])
    subset_masks = [sampled[k_idx][0][0] for k_idx in range(K_max)]       # first draw per K feeds the pair probe

    old_override = model._routing_override

    def counterfactual_nll(route_mask: torch.Tensor) -> torch.Tensor:
        """Evaluate one fixed route with the same prompt/shield contract."""
        model.set_routing_override("matrix", route_mask.detach())
        model.set_router_context_mask(prompt_mask)
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield_mask)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=ids.is_cuda):
                return per_example_answer_nll(
                    model(input_ids=ids, attention_mask=mask, use_cache=False).logits,
                    labels,
                    answer_tail_tokens=getattr(cfg, "answer_tail_tokens", 0),
                    answer_tail_weight=getattr(cfg, "answer_tail_weight", 1.0),
                )

    def deterministic_counterfactual_nll(route_mask: torch.Tensor) -> torch.Tensor:
        """Score one auxiliary counterfactual without dropout noise."""
        modes = {module: module.training for module in model.modules()}
        prior_override = model._routing_override
        model.eval()
        try:
            return counterfactual_nll(route_mask)
        finally:
            if prior_override is None:
                model.set_routing_override(None)
            else:
                model.set_routing_override(prior_override[0], prior_override[1])
            for module, was_training in modes.items():
                module.train(was_training)

    try:
        model.eval()
        arm_nll = torch.stack(
            [
                torch.stack([counterfactual_nll(mask_s) for (mask_s, _) in sampled[k_idx]], dim=1)
                for k_idx in range(K_max)
            ],
            dim=1,
        )                                                                 # (B, K_max, n_per_k)
        # Arm 0: Ground-truth unsteered baseline anchor
        model.set_routing_override("none")
        model.set_router_context_mask(prompt_mask)
        if hasattr(model, "set_prompt_shield_mask"):
            model.set_prompt_shield_mask(shield_mask)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=ids.is_cuda):
                nll_0 = per_example_answer_nll(
                    model(input_ids=ids, attention_mask=mask, use_cache=False).logits,
                    labels,
                    answer_tail_tokens=getattr(cfg, "answer_tail_tokens", 0),
                    answer_tail_weight=getattr(cfg, "answer_tail_weight", 1.0),
                )
    finally:
        if old_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(old_override[0], old_override[1])
        for module, was_training in module_modes.items():
            module.train(was_training)

    reward_0 = (-nll_0).detach()
    # A fixed K tariff is scale-blind: with answer NLL around one nat, the
    # former 0.025 cost was too small to ever offset a teacher-forced gain and
    # made K=4 the default policy.  Price each extra residual against the
    # unsteered difficulty of the *same* example.  K=0 is therefore a real
    # competing action whenever the marginal likelihood gain is insufficient.
    absolute_cost = float(getattr(cfg, "lambda_cost", 0.005))
    relative_cost = float(getattr(cfg, "router_relative_k_cost", 0.0))
    per_layer_cost = torch.maximum(
        torch.full_like(nll_0, absolute_cost),
        nll_0.detach() * relative_cost,
    )
    rewards = (-arm_nll - per_layer_cost.view(B, 1, 1) * k_actions.float().unsqueeze(-1)).detach()

    # Anchor-grounded leave-one-out advantage for each sampled subset.  K
    # itself is evaluated exhaustively and receives direct action-value
    # supervision below; this policy term learns which *layers* are useful at
    # a given K without an off-policy, forced-K gradient.
    baseline = (rewards.sum(dim=-1, keepdim=True) - rewards) / float(n_per_k - 1)   # same-K leave-one-out
    advantages = rewards - baseline                                                 # (B, K_max, n_per_k)
    subset_logprobs_t = torch.stack(
        [torch.stack([lp for (_, lp) in sampled[k_idx]], dim=1) for k_idx in range(K_max)], dim=1
    )                                                                               # (B, K_max, n_per_k)
    policy_loss = -(subset_logprobs_t * advantages).mean()
    action_rewards = rewards.mean(dim=-1)        # (B, K_max): per-K value estimate used by the K-head distillation

    # Direct pairwise *interaction* signal.  A route's gain over the base is
    # not synergy: two individually useful layers would otherwise make every
    # co-selected pair look positive.  For a sampled pair (i,j), estimate
    # I_ij = R({i,j}) - R({i}) - R({j}) + R(empty), then align only M_ij to it.
    # This preserves the cooperative-router hypothesis while giving destructive
    # pairs a negative target instead of allowing universal +1 saturation.
    reward_deltas = (action_rewards - reward_0.unsqueeze(1)).detach()
    pairwise_interaction_loss = torch.zeros((), device=ids.device)
    mean_pair_interaction = torch.zeros((), device=ids.device)
    if synergy is not None:
        # K=2 is the first non-trivial pair route and therefore measures a
        # clean second-order interaction instead of an arbitrary pair inside
        # a much larger selected set.
        pair_source = subset_masks[1] if K_max >= 2 else subset_masks[0]
        first = pair_source.float().argmax(dim=-1)
        without_first = pair_source.clone()
        without_first.scatter_(1, first.unsqueeze(-1), False)
        second = without_first.float().argmax(dim=-1)
        valid_pair = pair_source.sum(dim=-1).ge(2)
        probe = torch.rand_like(valid_pair.float()) < float(getattr(cfg, "router_pairwise_probe_probability", 0.25))
        valid_pair = valid_pair & probe
        if valid_pair.any():
            singleton_i = torch.zeros_like(pair_source)
            singleton_j = torch.zeros_like(pair_source)
            singleton_i.scatter_(1, first.unsqueeze(-1), True)
            singleton_j.scatter_(1, second.unsqueeze(-1), True)
            pair_mask = singleton_i | singleton_j
            nll_i = deterministic_counterfactual_nll(singleton_i)
            nll_j = deterministic_counterfactual_nll(singleton_j)
            nll_pair = deterministic_counterfactual_nll(pair_mask)
            interaction = (-nll_pair + nll_i + nll_j - nll_0).detach()
            pair_syn = synergy[torch.arange(ids.shape[0], device=ids.device), first, second]
            target_scale = max(float(getattr(cfg, "router_synergy_target_scale", 0.10)), 1e-4)
            interaction_target = torch.tanh(interaction / target_scale)
            pairwise_interaction_loss = F.smooth_l1_loss(
                pair_syn[valid_pair], interaction_target[valid_pair], beta=0.25,
            )
            mean_pair_interaction = interaction[valid_pair].mean()

    h_k, h_subset = policy_entropy_bonus(u_fresh, k_logits_fresh, K_max)
    diversity_temp = float(getattr(cfg, "router_diversity_temperature", 1.0))
    k_mi_loss, layer_mi_loss = mutual_information_diversity_loss(
        k_logits_fresh, u_fresh, temperature=diversity_temp,
    )

    beta_k = getattr(cfg, "beta_entropy_k", 0.15)
    beta_subset = getattr(cfg, "beta_entropy_subset", 0.15)

    # Balance loss removed: old MSE loss forced all layers to ~mean(K)/8 (40% each),
    # preventing specialization. Layer dropout (0.15) already breaks monopoly naturally.
    lambda_synergy = getattr(cfg, "lambda_synergy", 0.10)
    lambda_k_mi = getattr(cfg, "lambda_router_k_mi", 0.0)
    lambda_layer_mi = getattr(cfg, "lambda_router_layer_mi", 0.0)
    k_preference_loss = counterfactual_k_preference_loss(
        k_logits_fresh, k_actions, reward_0, action_rewards,
        temperature=float(getattr(cfg, "router_k_preference_temperature", 0.05)),
    )
    lambda_k_preference = float(getattr(cfg, "lambda_router_k_preference", 0.0))
    synergy_prior_loss = model.router.synergy_prior_loss(synergy) if synergy is not None else torch.zeros((), device=ids.device)
    lambda_synergy_prior = float(getattr(cfg, "lambda_synergy_prior", 0.0))

    # Kept at zero by default for backwards-compatible configuration.  The
    # reported mean is exhaustive K coverage, not a biased random sample.
    mean_sampled_k = k_actions.float().mean()

    loss = (
        policy_loss
        - beta_k * h_k
        - beta_subset * h_subset
        + lambda_synergy * pairwise_interaction_loss
        + lambda_k_mi * k_mi_loss
        + lambda_layer_mi * layer_mi_loss
        + lambda_k_preference * k_preference_loss
        + lambda_synergy_prior * synergy_prior_loss
    )

    win_rate = (action_rewards > reward_0.unsqueeze(1)).float().mean().item()
    diag = {
        "policy_loss": float(policy_loss.detach().item()),
        "synergy_align_loss": float(pairwise_interaction_loss.detach().item()),
        "pairwise_interaction": float(mean_pair_interaction.detach().item()),
        "win_rate_vs_base": float(win_rate),
        "mean_reward_delta": float(reward_deltas.mean().item()),
        "h_k_raw": float(h_k.detach().item()),
        "h_k_weighted": float((beta_k * h_k).detach().item()),
        "h_subset_raw": float(h_subset.detach().item()),
        "h_subset_weighted": float((beta_subset * h_subset).detach().item()),
        "mean_k_sampled": float(mean_sampled_k.detach().item()),
        "k_mi_loss": float(k_mi_loss.detach().item()),
        "layer_mi_loss": float(layer_mi_loss.detach().item()),
        "k_preference_loss": float(k_preference_loss.detach().item()),
        "synergy_prior_loss": float(synergy_prior_loss.detach().item()),
        "k0_probability": float(F.softmax(k_logits_fresh.detach(), dim=-1)[:, 0].mean().item()),
        "mean_per_layer_cost": float(per_layer_cost.mean().item()),
        "argmax_k_distribution": {
            int(k): float((k_logits_fresh.detach().argmax(dim=-1) == k).float().mean().item())
            for k in range(k_logits_fresh.shape[-1])
        },
        "counterfactual_k_actions": list(range(1, K_max + 1)),
    }
    return loss, diag
```
Config: `router_samples_per_k: int = 2`; remove `lambda_k_center` and `router_balance_weight` (ISSUE-16). Replace the checkpoint selection in `train_router_bootstrap` (trainer L884–890) with held-out *deployed-policy* gain, and compute it from `eval_data`, never from the training loss:
```python
def router_heldout_gain(model, rows, cfg, device) -> float:
    """> 0 means the deployed (argmax) route lowers held-out answer NLL vs unsteered."""
    prev = model._routing_override
    prev_mode = getattr(model, "_router_inference_mode", "argmax")
    model.set_router_inference_mode("argmax")
    try:
        model.set_routing_override("none")
        base = evaluate_loss(model, rows, cfg, device, answer_aware=True)
        model.set_routing_override(None)
        routed = evaluate_loss(model, rows, cfg, device, answer_aware=True)
    finally:
        if prev is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(prev[0], prev[1])
        model.set_router_inference_mode(prev_mode)
    return base - routed
```
and in the epoch loop:
```python
        gain = router_heldout_gain(model, [eval_data[i] for i in range(min(128, len(eval_data)))], cfg, device)
        if gain > best_gain + 1e-4:
            best_gain, patience_ctr = gain, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("router.")}
        else:
            patience_ctr += 1
```
(initialise `best_gain = float("-inf")` where `best_loss` was; keep returning `-best_gain` in place of `best_loss` for logs.)

**Impact on Project Goal:** less noisy layer selection, and the saved router is the one that helps held-out rows, not the one with the smallest regulariser-dominated loss. Medium effect; do after 1–7.

---

### ISSUE-9: Router input and steering may be dominated by the attention-sink token
**Tag:** `[DESIGN-FLAW]` `[NUMERICAL]`  **Evidence:** HYPOTHESIS — run the diagnostic first
**Files:** `router.py` L102–125; `wrapper.py` L165

**Problem:** (a) `_base_features` mean-pools the *raw* residual of the first target layer over all valid tokens (L111), including the first token. Decoder LLMs commonly put massive-magnitude activations on the first position/outlier dimensions; a mean over tokens is then nearly the same vector for every prompt, and `LayerNorm` inside `summary_proj` cannot undo outlier-dimension dominance. The three scalar features (`mean_rms`, `rms_std`, `dispersion`, L104–119) are unnormalised and fed straight to `input_proj`. (b) `steer_mask = mask` (wrapper L165) steers the first token, whose K/V every later token attends to. If (a) holds, it explains the repeated "deployment argmax collapsed" / "global K=3 bias" entries in the git history: the router literally cannot tell prompts apart.

**Diagnostic (run before changing anything).** `scripts/diagnose_router_inputs.py` (not executed here — no GPU):
```python
import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.configs.config import RSCConfig
from src.data.dataset import gsm8k_to_cot

cfg = RSCConfig(n_samples=200)
tok = AutoTokenizer.from_pretrained(cfg.model_name)
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float16).cuda().eval()
_, ev = gsm8k_to_cot(tok, cfg)
layer = RSCConfig().rsc_layers_for(len(model.model.layers))[0]
grab = {}
model.model.layers[layer].register_forward_hook(lambda m, a, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))
pooled_all, pooled_nofirst, first_ratio = [], [], []
for i in range(64):
    row = ev[i]; m = row["attention_mask"].bool() & row["labels"].eq(-100)
    ids = row["input_ids"].unsqueeze(0).cuda()
    with torch.no_grad():
        model(input_ids=ids, attention_mask=row["attention_mask"].unsqueeze(0).cuda())
    h = grab["h"][0][m.cuda()]
    pooled_all.append(h.mean(0)); pooled_nofirst.append(h[1:].mean(0))
    first_ratio.append((h[0].norm() / h[1:].norm(dim=-1).median()).item())
def mean_pair_cos(v):
    v = torch.nn.functional.normalize(torch.stack(v), dim=-1)
    c = v @ v.T
    return ((c.sum() - c.diag().sum()) / (c.numel() - len(v))).item()
print("mean pairwise cosine, all tokens   :", mean_pair_cos(pooled_all))
print("mean pairwise cosine, drop 1st tok :", mean_pair_cos(pooled_nofirst))
print("median ||h_0|| / median ||h_t||    :", sorted(first_ratio)[len(first_ratio) // 2])
```
Proceed with the fix **only if** the first cosine is > ~0.95 or the norm ratio is ≫ 10.

**Fix (if confirmed).** Replace `_base_features` in `JointLayerRouter`:
```python
    def _base_features(self, h: torch.Tensor, valid_mask: torch.Tensor):
        h_det = h.detach().float()
        first = valid_mask.float().cumsum(dim=1).eq(1) & valid_mask          # first valid position per row
        valid = valid_mask & ~first
        valid = torch.where(valid.any(dim=1, keepdim=True), valid, valid_mask)  # 1-token prompts keep it
        vf = valid.float()
        denom = vf.sum(dim=1).clamp(min=1)

        log_rms = torch.log1p(h_det.square().mean(dim=-1).sqrt())
        mean_rms = (log_rms * vf).sum(dim=1) / denom
        centered = (log_rms - mean_rms.unsqueeze(1)) * vf
        rms_std = centered.square().sum(dim=1).div(denom).sqrt()
        coverage = vf.mean(dim=1)
        pooled = (h_det * vf.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
        pooled = torch.sign(pooled) * torch.log1p(pooled.abs())              # tame outlier dimensions

        if self.difficulty_aware:
            norm_len = (torch.log1p(denom) / 6.24).clamp(0.0, 1.0)
            dispersion = centered.abs().sum(dim=1) / denom
            self._last_scalars = torch.stack([mean_rms, rms_std, coverage, norm_len, dispersion], dim=-1).detach()
        else:
            self._last_scalars = torch.stack([mean_rms, rms_std, coverage], dim=-1).detach()
        self._last_pooled = pooled.detach()
        return self._last_scalars, self._semantic_from_pooled(pooled)
```
and exclude the first token from steering, in the hook after `steer_mask = mask` (wrapper L165):
```python
                        steer_mask = mask
                        if getattr(wrapper.cfg, "steer_exclude_first_token", True) and h_fp.shape[1] > 1:
                            steer_mask = steer_mask & ~(mask.float().cumsum(dim=1).eq(1) & mask)
```
(single-token cached decode is never position 0, so the `shape[1] > 1` guard is correct for prefill, teacher forcing, and decode alike.) Config: `steer_exclude_first_token: bool = True`. Treat the steering half as an ablation: keep only if the paired test (ISSUE-2) does not get worse.

**Impact on Project Goal:** if (a) is true, no router change can help until prompts become distinguishable; if (b) is true, every route perturbs the sink all later tokens read.

---

### ISSUE-10: `LRReducer` warmup is a no-op and every plateau restarts the cosine
**Tag:** `[BUG-LOGIC]` `[CONFIG-DRIFT]`  **Evidence:** EXECUTED
**File:** `src/training/losses.py` L12–61

**Problem:**
```python
# L55-61
def scheduler_step(self):
    self.current_step += 1
    step = max(0, self.current_step - self.warmup_steps)      # 0 for the whole warmup
    ...
    cosine_factor = 0.5 * (1 + math.cos(math.pi * step / remaining))   # = 1.0 when step == 0
    pg["lr"] = max(self.min_lr, self.base_lrs[i] * cosine_factor)
```
Measured with `warmup_steps=60`, `base=5e-4`: LR at steps 1/10/30/60 = **5.00e-4 each** (full LR from the first step); `phase1_warmup_ratio`/`phase2_warmup_ratio` have no effect. `step()` additionally sets `self.current_step = self.warmup_steps` on plateau (L48), which re-launches the cosine from its peak at a *higher* LR than the one just reached (2.75e-4 → jumps to the reduced peak 2.5e-4 immediately, and in later epochs upward).

**Fix** (tested [EXECUTED]: warmup ramps 8.3e-6 → 5e-4, monotone cosine to 1e-6, plateau never raises LR). Replace `LRReducer`:
```python
class LRReducer:
    def __init__(self, optimizer, total_steps, factor=0.5, patience=3, min_lr=1e-6, delta=1e-4, warmup_steps=0):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.current_step = 0
        self.factor, self.patience, self.min_lr, self.delta = factor, patience, min_lr, delta
        self.best_loss = float("inf")
        self.wait = 0
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.original_base_lrs = list(self.base_lrs)
        self.n_reductions = 0

    def _schedule_factor(self):
        if self.current_step < self.warmup_steps:
            return (self.current_step + 1) / float(self.warmup_steps)
        remaining = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (self.current_step - self.warmup_steps) / remaining)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, val_loss):
        """Plateau: shrink the peak LR but keep the schedule position."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss, self.wait = val_loss, 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.n_reductions += 1
            self.base_lrs = [max(lr * self.factor, self.min_lr) for lr in self.base_lrs]
            self.wait = 0
            print(f"  ⚠️ LR peak reduced (×{self.factor}^{self.n_reductions}): {[f'{lr:.2e}' for lr in self.base_lrs]}")

    def scheduler_step(self):
        f = self._schedule_factor()
        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = max(self.min_lr, self.base_lrs[i] * f)
        self.current_step += 1
```
Also delete the now-redundant `lr_reducer_p1.base_lrs = [cfg.phase1_lr]` / `optimizer_p1.param_groups[0]["lr"] = …` (trainer L677–678): the constructor already reads them.

**Impact on Project Goal:** the first optimizer steps on freshly zero-initialised adapters currently run at full LR; with fp16 backbone and Adam eps 1e-6 that is the likeliest source of early noisy updates.

---

### ISSUE-11: Phase-1 acceptance has a winner's curse, too little power, and confirms on its selection set
**Tag:** `[DESIGN-FLAW]`  **Evidence:** EXECUTED (power); STATIC (selection)
**File:** `trainer.py` L214–423; `config.py` L77–101

**Problem:** The audit screens 8 layers on **8** items (`phase1_generation_screen_examples=8`), takes the top-3, evaluates them on 64, picks the best, then picks the best of 4 α-scales *on the same 64*, then reports `acc_gain = selected_val_acc − base_val_acc` on those same 64 (L353–357) and compares it to a +10 pp threshold. That is max-of-~12 selected on the confirmation set. Power [EXECUTED, independent arms, true effect = 0, base acc 40 %]: P(apparent gain ≥ +10 pp) = **12.4 % at n=64**, 6.9 % at n=100, 0.6 % at n=300 — before any selection effect, which multiplies it.

**Fix:** disjoint select/confirm halves and a paired CI. In `phase1_steernet_acceptance`: after L227 add
```python
    half = val_n // 2
    val_select, val_confirm = val_probe[:half], val_probe[half:]
```
and replace `val_probe` by `val_select` in the four generation calls at L257, L291, L309 and L340 (the NLL sweeps may keep `val_probe`). Replace L353–383 (from `best_scale = …` through the `report = {…}` dict) with:
```python
        best_scale = max(scale_results, key=lambda row: (row["accuracy"], row["gate_scale"]))   # chosen on val_select only
        from src.evaluation.deployment import paired_delta_ci
        base_conf = _phase1_zero_shot_generation_probe(model, tokenizer, val_confirm, cfg, device, len(val_confirm),
                                                       progress_desc="Confirm baseline")
        sel_conf = _phase1_zero_shot_generation_probe(model, tokenizer, val_confirm, cfg, device, len(val_confirm),
                                                      layer=best["layer"], gate_value=best_scale["gate_scale"],
                                                      progress_desc=f"Confirm L{best['layer']}")
        flags_b = [int(d["correct"]) for d in base_conf["details"]]
        flags_s = [int(d["correct"]) for d in sel_conf["details"]]
        n_pair = min(len(flags_b), len(flags_s))
        acc_gain, ci_lo, ci_hi = paired_delta_ci(flags_b[:n_pair], flags_s[:n_pair], seed=cfg.seed)
        selected_val_acc = sel_conf["accuracy"]
        base_val_acc = base_conf["accuracy"]
        nll_reduction = (base_train_nll - best["train_nll"]) / max(base_train_nll, 1e-8)
        val_nll_reduction = (base_val_nll - best["val_nll"]) / max(base_val_nll, 1e-8)
        cfg.fixed_steer_eval_layers = [best["layer"]]
        cfg.fixed_steer_eval_gate_scale = float(best_scale["gate_scale"])
        report = {
            "baseline_train_nll": base_train_nll, "baseline_val_nll": base_val_nll,
            "baseline_validation_accuracy": base_val_acc, "candidates": candidates,
            "selected_layer": best["layer"], "selected_validation_accuracy": selected_val_acc,
            "fixed_steer_gate_scale": cfg.fixed_steer_eval_gate_scale, "gate_scale_candidates": scale_results,
            "train_relative_nll_reduction": nll_reduction, "val_relative_nll_reduction": val_nll_reduction,
            "validation_accuracy_gain": acc_gain, "validation_gain_ci95": [ci_lo, ci_hi],
            "validation_examples": val_n, "confirm_examples": n_pair, "reliability_waived": small_run_waiver,
            "passed": bool(ci_lo > 0.0),          # informational once ISSUE-3's skip flag is False
        }
```
Config: `phase1_acceptance_examples: int = 200`, `phase1_acceptance_accuracy_examples: int = 200`, `phase1_generation_screen_examples: int = 24`.

**Impact on Project Goal:** acceptance stops certifying noise; with ISSUE-2 it no longer controls whether routing is trained.

---

### ISSUE-12: Decode flags confound results; checkpoint-selection decode differs from final decode
**Tag:** `[BUG-TRAIN-EVAL]`  **Evidence:** STATIC (the flags and the mismatch); mechanism = HYPOTHESIS
**File:** `src/evaluation/verifier.py` L191–260, L305–315

**Problem:**
```python
# L305-315 (benchmark, Phase-1 probe, qualitative eval, qualified "ARS" accuracy)
model.generate(**inputs, max_new_tokens=..., stopping_criteria=stopping, do_sample=False,
               repetition_penalty=1.1, no_repeat_ngram_size=6, use_cache=True)
# L246-249 (epoch_end_accuracy → _evaluate_and_maybe_checkpoint checkpoint ranking)
gen = model.generate(input_ids=q_ids, attention_mask=q_attn, max_new_tokens=max_new_tokens,   # 200
                     do_sample=False, pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id)
```
`repetition_penalty` in HF penalises every token already present in `input_ids` *including the prompt*; `no_repeat_ngram_size=6` forbids emitting any 6-token sequence already present in prompt+output. For arithmetic CoT (digits and operators recur) and for few-shot prompts (identical templates) that changes the distribution unequally across arms: e.g. the 8-shot exemplar ends `…8 dollars.\nFinal answer: 8`; the 6-gram `[" dollars", ".", "\n", "Final", " answer", ":"]` is in the prompt, so a test response ending in `dollars.` followed by `Final answer:` is *forbidden*; a steered model trained on `#### N` bypasses it. That would depress the baseline's answer-presence at high shot counts (the repo's baseline reports fall to 74 % at 8-shot) and manufacture an "ARS advantage" unrelated to reasoning. This is a hypothesis; confirm by re-running the baseline without the two flags.

**Root Cause:** decode settings are part of the policy under test, differ between selection (no flags, 200 tokens, no stop criterion) and reporting (flags, 256/512 tokens, answer-stop criterion), and penalise copying of problem quantities that correct chains must copy.

**Fix:** one deterministic greedy decode, shared by every caller. In `verifier.py` replace L305–315 by:
```python
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, stopping_criteria=stopping,
                do_sample=False, use_cache=True,
            )
```
and replace the generation block of `epoch_end_accuracy` (L245–251) by a call through the same path:
```python
            q_attn = torch.ones_like(q_ids)
            stopping = StoppingCriteriaList([AnswerStoppingCriteria(tokenizer, q_ids.shape[1], max_new_tokens)])
            gen = model.generate(
                input_ids=q_ids, attention_mask=q_attn, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True, pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id, stopping_criteria=stopping,
            )
            gen_text = truncate_at_answer(tokenizer.decode(gen[0][q_ids.size(1):], skip_special_tokens=True).strip())
            pred_num = extract_numeric_answer(gen_text)
```
(`truncate_at_answer`, `extract_numeric_answer` come from ISSUE-13; set `epoch_end_accuracy(max_new_tokens=256)` default). Also change `gold_num = extract_final_answer(gold_text)` (L241) to `extract_numeric_answer(gold_text)`.

**Impact on Project Goal:** removes the largest *measurement* confound; every earlier ARS-vs-baseline delta should be re-measured after this change.

---

### ISSUE-13: Three divergent answer extractors; `is` matches inside words
**Tag:** `[BUG-SILENT]`  **Evidence:** EXECUTED
**Files:** `multi_dataset.py` L237–288 (benchmark), `verifier.py` L84–135 (`extract_final_number`, `extract_final_answer`), `losses.py` L244–255 (`_extract_final_number`)

**Problem:**
```python
# multi_dataset.py L277-280
matches = list(re.finditer(r"(?:=|is)\s*\$?(-?[\d,]+\.?\d*)", text))   # 'is' needs no word boundary
```
`"she gave this 5 away, so 7 - 2 = 5"` and `"his 12 books … = 7"` match `is 5` / `is 12` inside `this`/`his`. Phase-1 probes, checkpoint accuracy and the benchmark each call a different function with a different priority list and different post-truncation, so "accuracy" is not one quantity. Responses lacking `####`/`Final answer` (the baseline zero-shot case) are the ones that hit the fragile fallbacks, so the error is *asymmetric between arms*.

**Fix:** one module, word-bounded, tested [EXECUTED] (asserts below passed). Create `src/evaluation/answers.py`; make `verifier.extract_final_answer`, `verifier.extract_final_number`, `multi_dataset.extract_numeric_answer` and `losses._extract_final_number` thin aliases of `extract_numeric_answer` (delete their bodies):
```python
import re

_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_MARKERS = (
    r"####\s*\$?(" + _NUM + ")",
    r"\\boxed\{\s*\$?(" + _NUM + r")\s*\}",
    r"[Ff]inal\s+[Aa]nswer(?:\s+is)?\s*[:\-]?\s*\$?(" + _NUM + ")",
    r"[Tt]he\s+answer\s+is\s*[:\-]?\s*\$?(" + _NUM + ")",
)


def _to_float(s):
    try:
        return float(s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip("."))
    except (ValueError, AttributeError):
        return None


def has_answer_marker(text: str) -> bool:
    return any(re.search(p, text) for p in _MARKERS)


def truncate_at_answer(text: str) -> str:
    """Cut after the earliest explicit answer marker; leave text unchanged if none."""
    ends = [m.end() for p in _MARKERS if (m := re.search(p, text))]
    return text[: min(ends)].strip() if ends else text


def extract_numeric_answer(text: str):
    for pat in _MARKERS:
        m = re.search(pat, text)
        if m:
            return _to_float(m.group(1))
    m = re.search(r"[Tt]herefore[^.\n]*", text)
    if m:
        nums = re.findall(_NUM, m.group(0))
        if nums:
            return _to_float(nums[-1])
    last = None
    for m in re.finditer(r"=\s*\$?(" + _NUM + ")", text):      # '=' only; never the substring 'is'
        last = m.group(1)
    return _to_float(last) if last is not None else None
```
Tests (add to `tests/test_modules.py`):
```python
    def test_extractor_word_boundaries(self):
        from src.evaluation.answers import extract_numeric_answer as ex
        self.assertEqual(ex("Step 1: x\n#### 18"), 18)
        self.assertEqual(ex("#### 1,234."), 1234)
        self.assertEqual(ex("Final answer: $6"), 6)
        self.assertEqual(ex("she gave this 5 away, so 7 - 2 = 5 apples"), 5)
        self.assertEqual(ex("his 12 books and 3 + 4 = 7"), 7)
        self.assertIsNone(ex("no numbers here"))
```

**Impact on Project Goal:** accuracy becomes a single, comparable quantity across training-time checks and benchmarks.

---

### ISSUE-14: Invariance loss has no gradient and wrong output indexing
**Tag:** `[DEAD-CODE]` `[BUG-GRAD]`  **Evidence:** STATIC
**Files:** `losses.py` L196–230; `trainer.py` L1040–1049; `config.py` L270–271

**Problem:** `compute_invariance_loss` runs both forwards under `torch.no_grad()` (L211) so `z1`, `z2` and the returned loss carry no graph; the docstring claim that "gradient flows through the pooled z vectors" (L209–210) is false. The caller guards with `inv_loss.requires_grad` (trainer L1048), so it is **silently skipped** — but only after two extra Phi-2 forwards every `invariance_every_n_steps`, in Phase 2A and the streamlined Stage 2 (the only `train_steer=True` paths). `pooled["cur"] = output[0]` (L202) also takes batch element 0 when a decoder layer returns a tensor (newer `transformers`), not the hidden states.

**Fix:** remove it. Delete `compute_invariance_loss` (`losses.py` L196–230) and its import (`trainer.py` L21), delete the block at `trainer.py` L1040–1049 and the config fields `lambda_invariance`, `invariance_every_n_steps`.

**Impact on Project Goal:** none on accuracy; removes wasted compute and a false belief that paraphrase invariance is being trained.

---

### ISSUE-15: Stale checkpoints are silently reloaded
**Tag:** `[BUG-CKPT]`  **Evidence:** STATIC
**Files:** `trainer.py` L1218–1220, L1391–1394, L1456–1458

**Problem:** `gate_only_save_path` / `phase2_save_path` default to fixed paths under `./checkpoints`. Guards are `os.path.exists(...)`. If a phase saves nothing this run (all-NaN val, early exception caught upstream, or `best_val` never improving on an `init_best_val`) the file from a *previous run* is loaded; `load_ars_checkpoint_strict` validates shapes/keys, not provenance.

**Fix:** `_purge_ckpt` at the start of every run (defined in ISSUE-4 and called at the top of `train_rsc`); for `train_streamlined_ars` add before Stage 2 (after L1177): `_purge_ckpt(cfg.phase2_save_path)`. Also, in `load_ars_checkpoint_strict`, accept an optional `expected_git_commit` and warn when `manifest["git_commit"]` differs.

**Impact on Project Goal:** prevents evaluating a different run's weights under this run's logs.

---

### ISSUE-16: Config drift and dead code
**Tag:** `[CONFIG-DRIFT]` `[DEAD-CODE]`  **Evidence:** EXECUTED (reference scan over `src/`, `scripts/`, `tests/`, notebooks)
**Files:** `config.py`, `rloo.py`, `wrapper.py` L391

* **Defined, never read anywhere:** `catalog_include_pairs`, `catalog_consecutive_triples`, `router_catalog_samples`, `router_max_tok_len`, `router_shot_mix`, `verifier_every_n`, `accuracy_eval_every_n` (accuracy is computed every epoch regardless), `router_dead_layer_veto_frac`. **Read only by tests:** `router_balance_weight` (the loss uses a hard-coded `lambda_balance = 0.0`). Delete them.
* `rloo.py`: `k_center_loss`/`lambda_k_center` operate on a constant (`mean_sampled_k = arange(1,K+1).mean()`), `balance_loss` is a constant zero — removed in the ISSUE-8 replacement.
* `steernet.py` L58 hard-codes `/ 4.0` — fixed in ISSUE-5.
* `wrapper.py` L391 calls `sample_k(k_logits)` **without `k_max`**, while L397 passes it; `sample_k` falls back to `k_max or 4` (rloo L86) and for a 4-class legacy head returns `choice + 1`. With `max_active_layers = 3` the 4-logit head makes this return K=4 > K_max. Fix: `k_real, _ = sample_k(k_logits, k_max=K_max)` at L391, and in `rloo.py`:
```python
def sample_k(k_logits: torch.Tensor, k_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample K in {0..k_max} from a (B, k_max+1) head."""
    if k_logits.shape[-1] != k_max + 1:
        raise ValueError(f"k_logits has {k_logits.shape[-1]} classes, expected {k_max + 1}")
    log_p_grad = F.log_softmax(k_logits, dim=-1)
    with torch.no_grad():
        log_p_det = F.log_softmax(k_logits.detach(), dim=-1)
        u_noise = torch.rand_like(log_p_det).clamp(min=1e-20, max=1 - 1e-20)
        choice = (log_p_det - torch.log(-torch.log(u_noise))).argmax(dim=-1)
    return choice, log_p_grad.gather(1, choice.unsqueeze(-1)).squeeze(-1)
```
(update `tests/test_modules.py::test_rloo_sampling` to a 5-column head and `k_max=4`).
* `lambda_pg=0.0` makes `compute_pg_*` (losses.py L257–543) unreachable by default. If revived: it applies REINFORCE with one scalar advantage to every token's `log α`, treats the continuous α as a Bernoulli probability, and measures the advantage under the argmax route while the gate trains on sampled routes. Delete or leave off.

**Impact on Project Goal:** none directly; eliminates misleading knobs.

---

### ISSUE-17: Gate streaming cache is inconsistent; dirac init shifts a token; the test passes only at init
**Tag:** `[BUG-TRAIN-EVAL]`  **Evidence:** STATIC
**Files:** `gate.py` L31, L144–147; `tests/test_modules.py` L64–75

**Problem:** (1) Prefill caches *pre-conv* features (L135, correct) and the cached decode branch keeps pre-conv (L143, correct), but the no-cache `T == 1` branch caches the **post-conv residual** and repeats it twice (L147). (2) `nn.init.dirac_` sets the centre tap (index 1) of a kernel-3 conv with `padding=2` and `[:T]` truncation, so the output at *t* is the feature of token *t−1*, not *t*; "identity init" is a one-token shift. (3) `test_gate_prefill_matches_streaming_decode` uses a fresh gate: with only the centre tap non-zero the wrong cache happens to give the same numbers. After any training of `causal_conv`, streaming from an empty cache would diverge from prefill.

**Fix:**
```python
        # gate.py __init__: true identity (current-token tap), causal alignment
        with torch.no_grad():
            self.causal_conv.weight.zero_()
            self.causal_conv.weight[:, 0, 2] = 1.0
            self.causal_conv.bias.zero_()
```
```python
            else:
                pre = x                                             # pre-conv features of this token
                x_conv = self.causal_conv(pre.transpose(1, 2))[:, :, :1].transpose(1, 2)
                x = pre + self.drop(self.act(x_conv))
                self._conv_cache = torch.cat([torch.zeros_like(pre), pre], dim=1).detach()   # [pad, token0]
```
Test: perturb the conv before comparing —
```python
    def test_gate_streaming_matches_prefill_with_trained_conv(self):
        gate = RSCUsefulnessGate(hidden_dim=self.hidden_dim, cfg=RSCConfig(gate_dropout=0.0)).eval()
        with torch.no_grad():
            gate.causal_conv.weight.normal_(0, 0.5)
            gate.causal_conv.bias.normal_(0, 0.1)
        h = torch.randn(1, 6, self.hidden_dim)
        delta = torch.randn_like(h)
        gate.reset_conv_cache()
        prefill = gate(h, delta).detach()
        gate.reset_conv_cache()
        streamed = torch.cat([gate(h[:, i:i + 1], delta[:, i:i + 1]).detach() for i in range(6)], dim=1)
        self.assertTrue(torch.allclose(prefill, streamed, atol=1e-6, rtol=1e-5))
```
(Changing the init changes the function the gate starts from; old gate checkpoints remain loadable but behave differently only until trained.)

**Impact on Project Goal:** low in the main path (generation always has a prefill cache); closes a latent train/eval divergence.

---

### ISSUE-18: Diagnostic counters and EMAs are saved into checkpoints
**Tag:** `[BUG-CKPT]`  **Evidence:** STATIC
**File:** `src/routing/router.py` L83–88, L204–215

**Problem:** `logit_ema`, `logit_sq_ema`, `selection_rate_ema`, `k_dist_ema`, `selected_forwards`, `seen_forwards` are persistent buffers under the saved `router.` prefix, so the checkpoint hash changes with every diagnostic update and loaded counters pollute later diagnostics. `observe()` returns early when the router is in `eval()` (L205) and `compute_router_rloo_loss` puts the router in eval, so the EMAs advance only during Phase 2C's sampled forwards; `selection_rate_ema` otherwise stays at its init 0.5.

**Fix:** make them non-persistent so they are neither saved nor loaded:
```python
        self.register_buffer("logit_ema", torch.zeros(n_candidates), persistent=False)
        self.register_buffer("logit_sq_ema", torch.zeros(n_candidates), persistent=False)
        self.register_buffer("selection_rate_ema", torch.full((n_candidates,), 0.5), persistent=False)
        self.register_buffer("k_dist_ema", torch.full((max_active_layers + 1,), 1.0 / (max_active_layers + 1)), persistent=False)
        self.register_buffer("selected_forwards", torch.zeros(n_candidates, dtype=torch.long), persistent=False)
        self.register_buffer("seen_forwards", torch.zeros(n_candidates, dtype=torch.long), persistent=False)
```
Existing checkpoints that contain these keys will then fail the strict loader's "unexpected keys" check; in `load_ars_checkpoint_strict` filter `saved` with `model.state_dict().keys()` for non-persistent names (or re-save once).

**Impact on Project Goal:** none; reproducibility hygiene.

---

## Cross-Cutting Concerns

**A. One proxy, five consumers.** Teacher-forced answer NLL is simultaneously the SteerNet loss, Phase-1 selector, router reward (`rloo.py`), K-head distillation target (`counterfactual_k_preference_loss`) and gate task loss. A flaw in the proxy (ISSUE-1) therefore propagates to every learned component at once; fixing any single consumer does not help. (ISSUES 1, 4, 8, 11)

**B. Regime drift across phases.** Phase 1: K=1, α=1, no budget, random layer. Phase 1.5: K∈{1..4}, α=1. Phase 2B: α learned (init 0.5) under four regularisers, router deterministic. Phase 2C: router sampled, α trained jointly. Deployment: router argmax, α learned. Each hand-off changes the function the frozen SteerNets see, and no stage is validated against the *deployed* policy on generation. (ISSUES 3, 6)

**C. Diagnostics treated as acceptance criteria.** Health audits (K support, entropy, active layers, gate std) are thresholds on *shape*, and the training objective contains matching terms (MI). The system is optimised to pass its own audits. (ISSUES 2, 7)

**D. Statistical underpowering everywhere.** Acceptance n=64, 8-item layer screen, checkpoint-selection accuracy n=100, historical reports n=100. A true-zero system looks like ±10 pp 7–12 % of the time each direction [EXECUTED]. Every historical "+12 pp / +21 pp" in `experiments/runs/*` must be treated as unverified until re-measured with a paired test on ≥300 items under the decode of ISSUE-12. (ISSUES 2, 11, 12)

**E. Evaluation harness asymmetry.** Decode flags (12), extractors (13), differing `max_new_tokens` (200 / 256 / 512) and differing stop criteria between selection and reporting mean "baseline" and "ARS" can differ in *measurement*, not just in model. (ISSUES 12, 13)

**F. Frozen-everything training.** The backbone, then the SteerNets after Phase 1, are frozen. Every later learner (router, gate) can only select or scale what Phase 1 produced. If Phase 1's solution is harmful (ISSUE-1/5), Phase 2 cannot repair it; it can only choose which harm to apply. (ISSUES 1, 3, 5)

---

## Recommended Fix Order

1. **ISSUE-10, 13, 12 (evaluation and schedule correctness).** Smallest diffs, no retraining dependency, and they make every subsequent measurement trustworthy. Re-run the **baseline and the current checkpoint** under the new decode/extractor; the baseline→ARS delta you see now is the first valid number.
2. **ISSUE-2 (paired baseline-anchored decision + persisted deployment mode).** Immediately guarantees "ARS ≥ baseline by construction" and logs the CI. Depends only on 12/13.
3. **ISSUE-15 and 18 (checkpoint hygiene).** Prevents contaminated runs while iterating on the rest.
4. **ISSUE-5 (SteerNet identity init) → ISSUE-3 (route-matched training).** Both change Phase 1 and force retraining; do them together so Phase 1 is trained once.
5. **ISSUE-4 (data split) and ISSUE-1 (RFT targets).** Also retraining; the largest expected effect. Run the **ISSUE-9 diagnostic** here (it needs no training) and apply its fix only if confirmed.
6. **ISSUE-6 and 7 (gate and diversity regularisers off).** Config-only; apply before retraining Phase 2.
7. **ISSUE-8 (per-K RLOO + held-out router selection), ISSUE-11 (acceptance power).** Refinements; they reduce variance and false certification.
8. **ISSUE-14, 16, 17 (dead code, drift, gate cache).** Hygiene.

Stop criterion: after step 5, if `finalize_deployment` reports `baseline`, the CI contains 0, and ≥300 held-out items were used, the supported conclusion is "no demonstrated benefit".

---

## What Correct Behavior Should Look Like

**Phase 1 / 2A.** Train answer-NLL on RFT targets decreases smoothly; validation NLL decreases less than before (targets are the model's own mode, so headroom is small — this is expected, not a failure). LR trace: linear ramp over the first 3–6 % of steps, then cosine; no upward jumps. At step 0 the SteerNet delta is exactly zero in `direct` mode (`steer(h).abs().max() == 0`) and generation with any route is token-identical to baseline.

**Router.** The held-out `router_heldout_gain` (baseline NLL minus routed NLL) is the tracked metric and should be ≥ 0 on the saved epoch. K and layer distributions may legitimately be concentrated (even mostly K=1 or K=0); concentration is no longer a failure. Per-example route *does* vary where the held-out reward varies — verify with the logged argmax-K distribution, not with a threshold.

**Gate.** Mean α stays near its 0.95 init unless task loss earns a reduction; no saturation penalty is acting; validation NLL after Phase 2B is within ~0.01 nat of the Phase-1/2A value (the repo's historical +0.13–0.14 nat degradation should disappear).

**Evaluation.** `baseline` and `ars` use identical greedy decode with the answer-stop criterion; answer-presence is high and similar for both at every shot count. `finalize_deployment` prints a paired Δ with 95 % CI on ≥200–300 held-out rows. Acceptable outcomes are (a) `ARS` with CI lower bound > 0, or (b) `BASELINE` with the CI reported. Anything else (a silent ARS < baseline) cannot occur.

**Do not expect** the old multi-shot "+21 pp" pattern. If it vanishes after ISSUE-12, it was a decode artefact.

---

## 8. Verification Appendix

**Executed in the review sandbox** (pure Python/NumPy; `fixes_pure.py`, `rloo_var.py`):

| Check | Result |
|---|---|
| Current `LRReducer`, warmup_steps=60, base 5e-4 | LR = 5.00e-04 at steps 1, 10, 30, 60 (warmup is a no-op) |
| Fixed `LRReducer` | step0 8.33e-06 → step60 5.00e-04 → last 1.00e-06; plateau never raises LR (assertions passed) |
| Fixed extractor | 6 assertions passed, incl. `"this 5 … = 5" → 5` and `"his 12 … = 7" → 7` |
| Power, true effect 0, base 40 % (independent arms) | P(apparent ≥ +10 pp): 12.4 % (n=64), 6.9 % (n=100), 0.6 % (n=300) |
| RLOO baseline simulation | same expected gradient (0.00980 vs 0.00982); std 0.0161 → 0.0089 (S=2) → 0.0061 (S=3) |
| Config reference scan | 8 fields never referenced; `router_balance_weight` tests-only |
| Patched `compute_router_rloo_loss` | `ast.parse` OK (245 lines); not run |

**Not executed (no torch/GPU):** every torch-code fix (ISSUES 3, 5, 6, 8, 9, 17, 18). **Add these tests before merging:**

1. `test_steernet_identity_at_init`: `RSCSteerNet(64, 16, 32.0, magnitude_mode=m)` → for `m="direct"`, `steer(randn(2,8,64), active_k=tensor([1,3])).abs().max() == 0`; for `"decoupled"`, `.norm(dim=-1).max() < 0.02`.
2. `test_phase1_route_distribution`: with `router_warmup=True`, `router_warmup_max_k=4`, over 500 forwards on a dummy 8-candidate wrapper, assert every K∈{1,2,3,4} occurs and every layer is selected ≥ 15 % of the time.
3. `test_hook_parity_teacher_forced_vs_cached_decode`: fix `set_routing_override("matrix", mask)` and `set_gate_freeze(True, 0.7)`; compare the logits of a full-sequence forward with token-by-token cached decode of the same sequence; require `allclose`. (I did not verify this property; it is the single most informative missing test for train/eval parity.)
4. `test_deployment_mode_roundtrip`: save via `finalize_deployment` with a stub that returns `baseline`, reload with `load_ars_checkpoint_strict`, run `evaluate_model_condition(..., "ars")`, assert output equals `"baseline"` condition output.
5. `test_sample_k_requires_matching_head`: 4-column head with `k_max=4` raises `ValueError`.
6. `test_gate_streaming_matches_prefill_with_trained_conv` (ISSUE-17).
7. `test_extractor_word_boundaries` (ISSUE-13).

**Open questions I could not resolve from the code alone** (each is cheap to answer with the logs of the failing run): (a) the exact baseline/ARS accuracies and answer-presence rates on the current HEAD; (b) whether `microsoft/phi-2` is loaded as native `PhiForCausalLM` (`base_model.model.layers`, `wrapper.py` L23) or via remote code in your environment — the wrapper assumes native; (c) fp16 stability of Phi-2 at the hook boundary (`h.float()` → `h_new.to(h.dtype)`), which the review had no means to test.
