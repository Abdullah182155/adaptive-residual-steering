<USER_REQUEST>
# IMPLEMENTATION_TASKS_SPEC.md

**Adaptive Residual Steering (ARS) on Phi-2 — Implementation Architecture & Task Decomposition**
Author: Lead Implementation Architect (Agent 2) · Source: `MASTER_REASONING_PLAN.md` (2026-10-23) · Spec date: 2026-10-23
Audience: Head Orchestrator (Antigravity) and Code Execution Agents (Agent 3.x)

---

## 0. How to Use This Document

1. **Dispatch order is §2.** Do not reorder. Hard stop at **DECISION 1** (§2.3) until the plan owner has read the audit.
2. **Each task (§3) is self-contained**: target files, prerequisites, signatures/logic, conflict isolation, verification. Signatures here are binding; bodies are specified as logic, not full source (Agent 3.x writes the code).
3. **Regression gate for every task:** `pytest tests -q -x` must be green before commit. Commit message format: `M<id>: <summary>`.
4. **Guardrail inheritance (project instructions §3):** no fatal diagnostics (only the fatal set in M0); retain all existing docstrings/math comments; new `RSCConfig` fields go in an append-only block headed `# ==== M<id> ====` placed immediately before `__post_init__`.
5. **API stability contract** (existing tests depend on these; do not change): `JointLayerRouter.forward(h, valid_mask) -> (u, k_logits)`, `RSCSteerNet(hidden_dim, rank, lora_alpha, dropout)`, `RSCUsefulnessGate(hidden_dim, cfg)`, `sample_k`, `sample_subset_plackett_luce`, `policy_entropy_bonus`, `train_rsc(...) -> (logs, best_val_p1, best_val_p2)`, `Phi2WithRSC.set_gate_freeze / set_routing_override / set_router_context_mask`.
6. **Legacy code is preserved, not deleted.** New behavior is added alongside (new functions / flags). Deprecation = flag default flips + docstring note.
7. **The plan owner decides, code never aborts on diagnostics.** Stage-0 outputs are *advisories* in `run_manifest.json`.

---

## 1. Anchor Verification (Part 0.2 / IV.5 `[verify]` items)

The repository I can read is the **snapshot** (`src/` as attached). The plan and notebook reference code that is **not in the snapshot** (dev branch `development @ 1b7447f`). Findings are split accordingly. Task **A0** (§3) turns the "cannot verify" column into a mechanical script so nothing is left to opinion.

### 1.1 Anchor table

| # | Anchor | Verified from snapshot | Cannot be verified here (A0 must resolve) |
|---|---|---|---|
| 1 | **Gate init** | `gate.py`: `fc2.weight = 0`, `fc2.bias = logit(cfg.gate_init_alpha)`; `config.py`: `gate_init_alpha = 0.80` ⇒ init α is **exactly 0.80 for every token** (logit 1.386). `docs/methodology.md` says bias −6.0 (α≈0.0025) — **stale, wrong for code**. | Preflight log shows L13 `mean_alpha=0.945`, 88.5 % high-saturation. **That is impossible at snapshot init** (zero weights ⇒ constant 0.80). So either (a) dev branch uses a different init, or (b) the gate weights were loaded from a checkpoint (`load_steer_weights` loads any `gate*` keys). A0 must decide which. |
| 2 | **"Gate = 1.0" vs certified 0.30** | `train_phase1_steernet` calls `set_gate_freeze(True, 1.0)` ⇒ hook multiplies `1.0 · delta / √K`, and `delta` already carries SteerNet's internal `lora_alpha/rank = 2.0`. **No constant 0.30 exists anywhere in the snapshot.** | Where 0.30 lives (a `steer_scale`, a relative-norm cap, a profile field). A0 measures the relative norm `‖α·Δ/√K‖/‖h‖` per layer at α∈{0.30, 1.0}, K∈{1,2,3} and matches it to the plan's "≈0.9 % at L13 (cap 2 %)" to identify which α is the certified operating point. |
| 3 | **`audit_router_policy_health` wiring** | Not referenced anywhere in the snapshot. `train_rsc` has no audit and returns `(logs, best_val_p1, best_val_p2)` on a single path. **However** `trainer._evaluate_and_maybe_checkpoint` contains a *soft-fatal*: if `dead_layers > cfg.router_dead_layer_veto_frac · n_candidates`, `improved` is forced `False` (checkpoint not saved). That is a diagnostic suppressing a checkpoint — M0 must neutralize it. | Every call site and every `raise`/early-`return` reachable from `audit_router_policy_health`, and the flags `skip_adaptive_phases_when_router_fails`, `router_fail_fast_on_collapse`, `continue_on_diagnostic_failure`. |
| 4 | **`ENGINE_V40_KAGGLE_PROFILE`** | **Absent** from snapshot `config.py` (only `RSCConfig`). Notebook imports `make_engine_v40_kaggle_config` (dev only). | Full resolved profile. A0 dumps it to `docs/anchor_profile_dump.json` and greps for the flag names above plus `router_demo_features`, `lambda_pg`, `calc_token_weight`, `calculation_weighted`. |

### 1.2 Additional findings from the snapshot (not in the plan's `[verify]` list)

| ID | Finding | Consequence | Owner |
|---|---|---|---|
| F1 | **Dropout active in router counterfactuals.** `train_router_bootstrap` calls `model.train()`; `compute_router_rloo_loss` runs both counterfactual forwards under `no_grad` **without** `eval()`. SteerNet (`lora_dropout=0.05`) and gate (`gate_dropout=0.05`) dropout are live. Confirms plan I.2.3 on the snapshot. | Counterfactual NLL differences contain dropout noise. | M5 (flag path), M6a (`counterfactual_mode`) |
| F2 | **"Fixed SteerNet = K=3 with 5 layers" explained.** In `_prepare_routes`, override mode `"only"` sets `_desired_k = count.clamp(max=K_max)` (→3), but `_active_k_realized = count.clamp(min=1)` (→5). The label comes from `_desired_k`; the hook's `√K` and SteerNet's `k_proj(K/4)` use the realized 5 (`k_norm=1.25`, outside the trained range). | The Fixed arm was out-of-distribution (confirms plan §0.3.4). | M6a |
| F3 | **No empty route exists in the learned policy.** `sample_k` returns ≥1; `_finalize_selection` forces ≥1; eval path uses `argmax+1`. K=0 is only reachable via `override="none"`. | Catalog class 0 requires new plumbing. | M5/M6b |
| F4 | **Invalid override in notebook & CLI.** `kaggle_ars_runner.ipynb` (Stage B cell) and `scripts/run_experiment.py` call `set_routing_override("all")`; the wrapper accepts only `{only, exclude, none, force, matrix}` ⇒ `ValueError`. | Stage-B eval cannot run on snapshot. | M10 (fix callers); M6a keeps the error |
| F5 | **Config/trainer drift.** `trainer.py` reads `cfg` fields absent from snapshot `RSCConfig`: `steer_finetune_epochs/lr/save_path`, `gate_only_epochs/lr/save_path`, `joint_finetune_epochs`, `joint_finetune_lr_scale`, `joint_finetune_router_lr_scale`, `gate_retune_epochs/lr`, `router_dead_layer_veto_frac`, `router_checkpoint_every_n_steps`. | `train_rsc` raises `AttributeError` on snapshot ⇒ dev config is the truth; confirms snapshot is stale. | A0 (inventory), M0 (add defaults) |
| F6 | **Checkpoint load is shape-fragile.** `load_steer_weights` uses `strict=False`, which still raises on *shape* mismatch (e.g., router with a different candidate count). | Audit on the Phase-1 checkpoint must load `steer_nets.*` only. | M1 (`load_steernets_for_audit`) |
| F7 | **Hook already supports per-row routes.** `matrix` override with `(B, n_candidates)` bool works, including all-False rows (`route=0`, fp32 round-trip exact). | The 26-subset table can be built by **batching the subsets as rows** of one forward. | M1 |
| F8 | **Router pooled features are available in any forward.** `_prepare_routes` calls `self.router(h, router_mask)` *before* branching on override, so `router._last_pooled` / `_last_scalars` are populated for every prefill, and they are pre-steering (the router reads `h` before the hook modifies it). | Router features can be **cached in the counterfactual table**, making S2 "table-only, no LLM forwards" possible. | M1 (capture), M5 |
| F9 | **S0 rows drawn from the train split are in-sample for SteerNets** (Phase-1 trained on them). `G_fixed*` on them is optimistic. | Audit must also report a held-out slice and cross-fitted gain. | M1/M2 |

### 1.3 Consolidated operating-point decision (binding for all tasks)

Define **`s_ref = cfg.steer_scale`** (new field, default `1.0` = snapshot Phase-1 behavior). The hook computes
`α_eff = α_gate · s_ref` (learned gate) or `α_eff = s` (constant bypass via `set_steer_scale(s)`, absolute).
A0 determines whether the certified point is `1.0` or `0.30`; if the dev branch's certified scale is `0.30`, A0's follow-up commit sets `steer_scale=0.30` in config and stamps it in the manifest. **Learned-gate identity (α=1) ≡ constant bypass at `s_ref`.** This removes the ambiguity between "Gate=1.0" and "scale 0.30" for conditions 2/3/4/5.

---

## 2. Execution Protocol & Dependency Graph

### 2.1 Task inventory

| ID | Title | New / modified files (summary) | GPU? |
|---|---|---|---|
| **A0** | Anchor verification | `scripts/verify_anchors.py`, `docs/ANCHOR_REPORT.md` | tiny |
| **M0** | Failure-policy refactor + test infra | `src/utils/*`, `config.py`, `trainer.py`, `tests/conftest.py` | no |
| **M6a** | Wrapper plumbing (partial, pre-audit) | `wrapper.py`, `src/models/contracts.py` | no |
| **M1** | Counterfactual table builder | `src/routing/catalog.py`, `src/data/shots.py`, `src/analysis/cf_table.py` | **yes** (45–70 min run) |
| **M2** | Signal audit | `src/analysis/signal_audit.py`, `scripts/run_signal_audit.py` | no (consumes table) |
| **M3** | On-policy sampler | `src/data/onpolicy.py`, `scripts/run_onpolicy_sampling.py` | **yes** (~1 h run) |
| **M4** | S1′ SteerNet trainer | `trainer.py`, `losses.py`, `src/evaluation/paired_generation.py` | **yes** (~40 min run) |
| **M5** | Catalog router + distillation | `router.py`, `src/routing/distill.py`, `rloo.py` | no (<10 min) |
| **M6b** | Wrapper plumbing (rest) | `wrapper.py` | no |
| **M7** | Token gate stage | `gate.py`, `losses.py`, `trainer.py` | **yes** (~30 min) |
| **M8** | S5 deployment calibration | `src/deploy/calibrate.py` | yes (<15 min) |
| **M9** | 5-arm benchmark runner | `src/evaluation/{multi_benchmark_runner,output_cache,stats}.py` | **yes** (2–4 h) |
| **M10** | Reporting + notebook + docs sync | notebook, `scripts/`, `docs/` | no |

### 2.2 Dependency graph

```
A0 ─► M0 ─► M6a ─┬─► M1 ─► M2 ══► DECISION 1 (G1 run: audit on Phase-1 ckpt)
                 │
                 └─► M3 (code only; parallel-safe with M1, disjoint files)
                                                     │
   DECISION 1 ──► [branch A: G_fixed*<0.01] M3-run ─► M4 ─► S1′ run (G3) ─► M1-rebuild (G4/S0b) ─► M2 ══► DECISION 2
              └─► [branch B: otherwise]            M4 (gold + layer-dropout; on-policy if available) ─► (G3,G4) ─► M2 ══► DECISION 2

   DECISION 2 ──► H ≥ 0.02 ........... M5 ─► M6b ─► M7 ─► M8 ─► M9 ─► M10
              ├─► 0.005 ≤ H < 0.02 ... same, optional RLOO polish disabled   (decision: see §5, D2)
              └─► H < 0.005 .......... M5 in `fixed_abstain` mode, then M6b ─► M7 ─► M8 ─► M9 ─► M10
```

### 2.3 Phase 1 Prerequisites and DECISION 1 gate (explicit)

**Phase 1 = {A0, M0, M6a, M1, M2}.** Nothing else may be dispatched to a GPU or merged until DECISION 1 is recorded. Concretely:

1. Dispatch order: `A0 → M0 → M6a → M1 → M2`. (`M3` code may be written in parallel with M1 but not run.)
2. **GPU run G1** (≈45–70 min, one T4):
   `python scripts/run_signal_audit.py --ckpt checkpoints/steer_phase1_adaptive_v11.pt --out_dir checkpoints/audit_s0 --steer_scale <from A0>`
3. Deliverables returned to the plan owner: `checkpoints/audit_s0/signal_audit.json`, `docs/AUDIT_REPORT_S0.md`, and the `advisories` block of `run_manifest.json`.
4. **Nothing in M5, M7, M8, M9 may be started** until the plan owner replies with the branch choice. Router/gate compute is not spent before this.

### 2.4 File ownership matrix (conflict control)

`✎` = edits existing file · `+` = creates · `config.py` is append-only by every task.

| File | A0 | M0 | M6a | M1 | M2 | M3 | M4 | M5 | M6b | M7 | M8 | M9 | M10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `src/configs/config.py` | | ✎ | ✎ | ✎ | ✎ | ✎ | ✎ | ✎ | ✎ | ✎ | ✎ | ✎ | |
| `src/training/trainer.py` | | ✎ | | | | | ✎ | | | ✎ | | | |
| `src/training/losses.py` | | | | | | | ✎ | | | ✎ | | | |
| `src/models/wrapper.py` | | | ✎ | | | | | | ✎ | | | | |
| `src/models/contracts.py` | | | + | | | | | | | | | | |
| `src/routing/router.py` | | | | | | | | ✎ | | | | | |
| `src/routing/rloo.py` | | | | | | | | ✎ | | | | | |
| `src/routing/catalog.py` | | | | + | | | | | | | | | |
| `src/routing/distill.py` | | | | | | | | + | | | | | |
| `src/gating/gate.py` | | | | | | | | | | ✎ | | | |
| `src/data/shots.py` | | | | + | | | | | | | | | |
| `src/data/onpolicy.py` | | | | | | + | | | | | | | |
| `src/analysis/*` | | | | + | + | | | | | | | | |
| `src/evaluation/*` | | | | | | | + | | | | | + | |
| `src/deploy/calibrate.py` | | | | | | | | | | | + | | |
| `src/utils/*`, `tests/conftest.py` | | + | | | | | | | | | | | |
| `scripts/*`, notebook, docs | + | | | | + | + | + | + | | + | + | + | ✎ |

### 2.5 GPU run cards

| Run | After | Command | Output |
|---|---|---|---|
| G1 | M2 | `python scripts/run_signal_audit.py --ckpt checkpoints/steer_phase1_adaptive_v11.pt --out_dir checkpoints/audit_s0` | audit + manifest advisories |
| G2 | M3, DECISION 1 branch A | `python scripts/run_onpolicy_sampling.py --n_questions 1500 --out checkpoints/onpolicy.jsonl` | `onpolicy.jsonl` |
| G3 | M4 | `python scripts/run_s1.py --onpolicy checkpoints/onpolicy.jsonl --out checkpoints/s1_steernet.pt` | S1′ checkpoint |
| G4 | G3 | `python scripts/run_signal_audit.py --ckpt checkpoints/s1_steernet.pt --out_dir checkpoints/audit_s0b` | S0b table + DECISION 2 inputs |
| G5 | M5, M6b, M7, M8 | `python scripts/run_router_distill.py … && python scripts/run_s3_gate.py … && python scripts/run_calibration.py …` | router, gate, `deployment_contract.json` |
| G6 | M9 | `python scripts/run_benchmark.py --contract deployment_contract.json --shots 0 8 --n 1319` | `experiment_record.json` |

### 2.6 Failure/warning code registry (shared vocabulary)

- **FATAL (only these):** `E_NONFINITE_PARAMS`, `E_NONFINITE_LOSS_STREAK`, `E_CKPT_MISSING`, `E_CKPT_CORRUPT`, `E_MANIFEST_CORRUPT`, `E_HOOK_CONTRACT`, `E_BASELINE_IDENTITY`, `E_SHAPE`, `E_ROUTE_PLUMBING`.
- **WARN (advisory, non-blocking):** `W_ROUTER_COLLAPSE`, `W_DEAD_LAYERS`, `W_K_SUPPORT`, `W_K_EXCEEDS_CAP`, `W_INSAMPLE_INFLATION`, `W_ROUTER_REGRET`, `W_GATE_NOT_ADOPTED`, `W_SYNERGY_SIGN`, `W_EXTRACTION_AGREEMENT`, `W_BATCH_INVARIANCE`, `W_ENVELOPE_MISMATCH`, `W_CFG_DEPRECATED_FLAG`.

---

## 3. Task Specifications

---

### TASK A0 — Anchor Verification (read-only audit + report)

**Target files:** `scripts/verify_anchors.py` (new), `docs/ANCHOR_REPORT.md` (generated, committed), `docs/anchor_profile_dump.json` (generated).
**Prerequisites:** none. Runs on dev branch `1b7447f`. Needs one GPU only for item 2 (can run on CPU with a tiny random model for items 1, 3, 4, 5).

**Core changes & logic.** `scripts/verify_anchors.py` produces the report; **it never modifies code and never aborts** (all findings printed; exit code 0). Sections:

1. **Gate init.** Build the gate from the resolved config; run on 64 random hidden states; print `mean α`, `% α>0.85`. Load the Phase-1 checkpoint and print `fc2.bias`, `‖fc2.weight‖` for each `gates.*` key (distinguishes "init" from "loaded checkpoint"). Print the init value implied by `gate_init_alpha`.
2. **Operating point.** On 16 GSM8K prompts, for layers `[13,15,18,20,22]`, K∈{1,2,3}, α∈{0.30, 1.0} print `mean ‖α·Δ/√K‖ / ‖h‖` (per layer). Report which α gives ≈0.9 % at L13 and the observed max (plan: cap 2 %). Recommend `steer_scale` for §1.3. Also `grep -rnE "0\.30|steer_scale|rel_norm|alpha_cap|max_rel" src/` and list hits.
3. **Audit wiring.** `grep -rn "audit_router_policy_health" src/ notebooks/ scripts/`; for each hit print the enclosing function and every `raise`, `return`, `sys.exit`, `assert` reachable from it. Also grep for `skip_adaptive_phases_when`, `router_fail_fast_on_collapse`, `continue_on_diagnostic_failure`, `router_dead_layer_veto_frac`. Emit a table **(file:function, current behavior, M0 replacement)** — M0 consumes this table.
4. **Profile dump.** `dataclasses.asdict(make_engine_v40_kaggle_config(...))` → `docs/anchor_profile_dump.json`; if `ENGINE_V40_KAGGLE_PROFILE` is a dict/constant, dump it too. Print which of: `router_demo_features`, `lambda_pg`, `calc_token_weight`, `calculation_weighted`, `max_active_layers`, `admitted_layers`/candidate list, `gate_init_alpha`, `lora_dropout` it sets. Print the dev `target_layers` actually used vs `[13,15,18,20,22]` (snapshot candidate pool is `rsc_layers_for(32)=[9,11,13,15,18,20,22,24]`).
5. **Dropout in counterfactuals.** grep the dev router-bootstrap/RLOO path for `.eval()`; print whether SteerNet dropout is live (F1).
6. **Fixed-arm K label.** grep dev `multi_benchmark_runner` for the Fixed condition construction; confirm F2 (5 layers, `_desired_k` label 3).
7. **Config drift.** Compare every `cfg.<field>` read in `src/training/*.py` against `RSCConfig` fields; print missing fields (F5).
8. **Eval-split sizes.** Print `len(train_data)`, `len(eval_data)`; locate where the "240 eval rows"/"120 selection rows" are defined (plan references both) and print their indices-hash.

**Conflict isolation:** read-only. No source file is touched. The two generated docs are new files.
**Verification:**
`python scripts/verify_anchors.py --device cpu --tiny` must exit 0 and write `docs/ANCHOR_REPORT.md` containing the section headers `## 1 Gate init` … `## 8 Splits`. `pytest tests -q -x` unchanged.
**Output to plan owner:** the report; if item 2 shows the certified point is 0.30, the Head Agent commits a one-line `steer_scale = 0.30` default before M6a.

---

### TASK M0 — Failure-Policy Refactor, Fatal-Gate Elimination, Test Infrastructure

**Target files:**
`src/utils/__init__.py` (+), `src/utils/failure_policy.py` (+), `src/configs/config.py` (✎), `src/configs/stages.py` (+), `src/training/trainer.py` (✎), dev preflight/audit modules listed by A0 §3 (✎), `tests/conftest.py` (+), `tests/test_failure_policy.py` (+).
**Prerequisites:** A0 (needs its call-site table).

**Core changes & logic**

*Config (block `# ==== M0 ====`):*
```python
failure_policy: str = "warn"                  # "warn" (default) | "strict" (CI only)
manifest_path: str = "./run_manifest.json"
max_consecutive_nonfinite_loss: int = 20
# Added only if absent (F5): steer_finetune_epochs/lr/save_path, gate_only_epochs/lr/save_path,
# joint_finetune_epochs, joint_finetune_lr_scale, joint_finetune_router_lr_scale,
# gate_retune_epochs/lr, router_dead_layer_veto_frac, router_checkpoint_every_n_steps.
steer_scale: float = 1.0                      # s_ref, see §1.3 (A0 may change default)
```
`__post_init__` coercion: when `failure_policy == "warn"`, force `skip_adaptive_phases_when_router_fails=False`, `router_fail_fast_on_collapse=False`, `continue_on_diagnostic_failure=True` (only if those attributes exist) and record `W_CFG_DEPRECATED_FLAG` if the input value differed. `ENGINE_V40_KAGGLE_PROFILE` / `make_engine_v40_kaggle_config` must be edited to stop setting them truthy.

*`src/configs/stages.py`:* `class Stage(str, Enum): S0="S0"; S1="S1"; S0B="S0b"; S2="S2"; S3="S3"; S4="S4"; S5="S5"` plus `LEGACY_TO_CANONICAL = {"phase1":"S1","phase1.5":"S2","2a":"S1","2b":"S3","2c":"S4","2d":"S3"}` (fixes the three-way naming mismatch in docs/notebook/`train_rsc`).

*`src/utils/failure_policy.py` (public API):*
```python
class FatalContractError(RuntimeError):
    code: str
class RunManifest:
    def __init__(self, path: str, policy: str = "warn", run_id: Optional[str] = None): ...
    def set(self, key: str, value: Any) -> None                     # dotted keys allowed
    def warn(self, code: str, message: str, stage: str = "", **data) -> None
    def fatal(self, code: str, message: str, stage: str = "", **data) -> NoReturn   # writes then raises
    def diagnostic(self, code: str, ok: bool, message: str, stage: str = "", **data) -> bool
        # ok -> returns True. not ok -> WARN (policy "warn") or FATAL (policy "strict"); returns False.
    def advisory(self, key: str, value: Any, interpretation: str = "") -> None
    @contextmanager
    def stage(self, name: str): ...        # records start/end/status; exception -> status="failed", then re-raises
    def flush(self) -> None                # atomic: write tmp + os.replace
def get_manifest() -> RunManifest          # module-level singleton; creates default from cfg.manifest_path
def set_manifest(m: RunManifest) -> None
def to_jsonable(x: Any) -> Any             # tensors, numpy scalars/arrays, dataclasses, Paths
```
Manifest schema: `{"schema_version":1,"run_id","failure_policy","config":{},"steer_scale","gate_mode","warnings":[{ts,stage,code,message,data}],"advisories":{key:{value,interpretation}},"stages":{name:{status,t_start,t_end,metrics}}}`. Every `warn/fatal/advisory` flushes immediately (crash-safe).

*`trainer.py` edits (preserve all signatures and `logs` keys):*
1. `_evaluate_and_maybe_checkpoint`: the dead-layer block must **only** `manifest.diagnostic("W_DEAD_LAYERS", …)`; it must **not** set `improved=False` unless `failure_policy=="strict"`. Keep the printed note text.
2. Non-finite handling: keep the existing `continue`-skip of a non-finite micro-batch (this is normal under fp16 `GradScaler`), but count consecutive occurrences; at `max_consecutive_nonfinite_loss` → `manifest.fatal("E_NONFINITE_LOSS_STREAK")`. After every stage call `_assert_finite_params(model)` (steer_nets/gates/router) → `E_NONFINITE_PARAMS`.
3. `train_rsc`: wrap each phase in `with manifest.stage(...)`; **no phase may be skipped** on diagnostic outcomes; at the end, `logs["manifest_path"] = cfg.manifest_path`. Return signature unchanged.
4. Every call site / raise path of `audit_router_policy_health` from A0's table → `manifest.diagnostic(...)`. Entropy, K support, active-layer counts, effective-layer counts, synergy signs, preflight "reliability" become WARN.
5. **Fatal set retained:** non-finite params, missing/corrupt checkpoint or manifest (`E_CKPT_*`, `E_MANIFEST_CORRUPT`), tensor shape / hook-contract violations (`E_SHAPE`, `E_HOOK_CONTRACT`), baseline-identity failure (`E_BASELINE_IDENTITY`, raised by M6a contracts).

*`tests/conftest.py` (shared by all later tasks; CPU, no network):*
```python
class CharTokenizer:            # vocab=128, pad_token_id=0, eos_token_id=1, unk_token_id=2, padding_side="right"
    # __call__(texts|text, return_tensors, padding, truncation, max_length, return_offsets_mapping) -> dict
    # encode, decode, get_vocab; offsets mapping supported (char i -> token i)
@pytest.fixture def tiny_cfg(tmp_path) -> RSCConfig       # lora_rank=4, gate_dim=4, checkpoint paths under tmp_path, batch_size=1, grad_accum=1
@pytest.fixture def tiny_model(tiny_cfg) -> Phi2WithRSC    # PhiForCausalLM(PhiConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                                                           #   num_hidden_layers=12, num_attention_heads=4)), fp32, CPU, eval()
@pytest.fixture def tiny_rows(char_tokenizer) -> list[dict] # 8 rows with input_ids, attention_mask, labels, question, steps_list, final_num
```
(12 layers ⇒ `rsc_layers_for(12) = [3,4,5,6,7,8]`.)

**Conflict isolation:** do **not** delete `skip_adaptive_phases_when_*`, `router_fail_fast_on_collapse`, `continue_on_diagnostic_failure`, `router_dead_layer_veto_frac` fields (profile compatibility). Do not touch loss math. Do not change `train_rsc`'s phase order (M10 supersedes it with a new orchestrator).

**Verification:** `pytest tests/test_failure_policy.py -q`
- `test_manifest_atomic_and_roundtrip`: warn/advisory/stage → file valid JSON after each call.
- `test_warn_policy_never_raises`: `diagnostic(..., ok=False)` returns False, adds WARN; `strict` raises `FatalContractError` with matching `.code`.
- `test_dead_layer_veto_is_warn_only`: call `_evaluate_and_maybe_checkpoint` with a stub model whose `routing_diagnostics()` reports all-dead layers and a stub `epoch_end_accuracy` that improves ⇒ checkpoint **is saved**, manifest has `W_DEAD_LAYERS`.
- `test_train_rsc_completes_under_collapse`: monkeypatch the five stage functions to stubs and make the (dev) audit return "collapsed"; `train_rsc` runs all stages; manifest has ≥1 `W_ROUTER_COLLAPSE`/`W_DEAD_LAYERS`; no exception.
- `test_nonfinite_streak_fatal`: 20 consecutive NaN losses ⇒ `E_NONFINITE_LOSS_STREAK`.

---

### TASK M6a — Wrapper Plumbing (partial: everything the audit needs)

**Target files:** `src/models/wrapper.py` (✎), `src/models/contracts.py` (+), `src/configs/config.py` (✎), `tests/test_wrapper_contract.py` (+).
**Prerequisites:** M0 (manifest, `FatalContractError`, conftest).

**Core changes & logic (all additive; legacy entry points unchanged):**

1. **Config block `# ==== M6a ====`:** `k_scale_mode: str = "sqrt"` (`"sqrt"|"none"`), `steernet_k_conditioning: bool = True`, `gate_task_gradient: bool = False` (consumed in M6b/M7; declared now to avoid later churn).
2. **Operating-point math in the hook** (replaces the block from `delta = s(...)` to `h_new = ...`; keep everything else):
```python
if wrapper._hooks_disabled:                      # top of hook_fn: return output untouched
    return output
active_k = wrapper._active_k_realized
delta = s(h_fp, active_k=active_k if wrapper.cfg.steernet_k_conditioning else None)
if wrapper._steer_scale is not None:             # constant bypass (conditions 2/3, ungated reference)
    alpha_eff = torch.full((h_fp.size(0), h_fp.size(1), 1), wrapper._steer_scale, dtype=h_fp.dtype, device=h_fp.device)
    g._alpha_for_aux, g.last_alpha = None, None  # gate not evaluated
else:                                            # learned / frozen-valued gate (legacy path)
    alpha = g(h_fp, delta, global_context=wrapper.router._last_semantic_shared)
    g._valid_mask = mask
    alpha_eff = alpha.detach() * wrapper.cfg.steer_scale     # M6b flips .detach() on cfg.gate_task_gradient
k_scale = active_k.to(h_fp.dtype).sqrt().view(-1,1,1) if wrapper.cfg.k_scale_mode == "sqrt" else 1.0
step = route * alpha_eff * delta * mask.unsqueeze(-1) / k_scale
h_new = h_fp + step
wrapper._record_envelope(layer_pos, step, h_fp, mask, h_new)   # per-item rel-norm + non-finite (cheap, always on)
```
   With defaults (`steer_scale=1.0`, `k_scale_mode="sqrt"`, `steernet_k_conditioning=True`, no bypass) numerics are **bit-identical** to the snapshot — this is a test.
3. **New public methods:**
```python
def set_steer_scale(self, s: Optional[float]) -> None       # None disables bypass
@contextmanager
def hooks_disabled(self): ...                               # reference "unwrapped" forward; keeps pre-hook running
@contextmanager
def counterfactual_mode(self): ...                          # eval(), no_grad; saves/restores training flag, routing override, steer_scale, gate_freeze, router context mask
def subset_matrix(self, subsets: Sequence[Sequence[int]]) -> torch.BoolTensor   # layer ids -> (B, n_candidates); ValueError if layer not a candidate
def current_route_tuples(self) -> List[Tuple[int, ...]]     # layer ids per batch row, from `_precomputed_routes` (valid at prefill AND cached decode)
routing_step_log: List[dict]                                # when set_routing_trace(True): {"decode": bool, "routes": [...]} appended once per forward at first candidate layer
guard_state: dict                                           # {"rel_max": (B,), "nonfinite": (B,)}; reset on prefill (not decode); max-accumulated across decode steps
```
4. **`_prepare_routes` override branch:** `_desired_k = selected_matrix.sum(-1)` **uncapped, `min=0`** (fixes F2: label equals realized K). For `matrix` mode: validate width `== len(target_layers)` (else `E_SHAPE`); if any row exceeds `cfg.max_active_layers` → `manifest.warn("W_K_EXCEEDS_CAP")`. `set_routing_override("all")` still raises `ValueError` (F4) with a message pointing to `subset_matrix`.
5. **`src/models/contracts.py`:**
```python
def check_baseline_identity(model, tokenizer, prompts, tol: float) -> None   # III-1: override="none" logits vs hooks_disabled() logits; E_BASELINE_IDENTITY
def check_hook_noop(model, tokenizer, prompts) -> None                       # III-2: all-False matrix ⇒ per-layer outputs equal hooks_disabled outputs; E_HOOK_CONTRACT
def check_route_plumbing(model, tokenizer, prompts, forced: Sequence[Sequence[int]]) -> None  # III-3: forced matrix, batch>1, left-padded, generate(max_new_tokens=4); every routing_step_log entry equals forced tuples for prefill AND ≥1 decode step; E_ROUTE_PLUMBING
def run_contract_checks(model, tokenizer, device, n_prompts: int = 16, tol: Optional[float] = None) -> None
    # tol default: 1e-3 if model dtype is fp16 else 1e-5
```
   All raise `FatalContractError`. (III-4 lives in M9.)

**Conflict isolation:** do not edit `_router_mask_for`, `_finalize_selection`, warmup logic, router EMA bookkeeping, or the `routing_k_hist` accounting. Keep `set_gate_freeze` and its semantics (α constant, now × `s_ref`). Do not add catalog/question-span code here (M6b).

**Verification:** `pytest tests/test_wrapper_contract.py -q`
- `test_default_numerics_unchanged`: with defaults, logits equal a recorded reference computed by the *pre-M6a* hook formula re-implemented inline in the test (assert `torch.equal`).
- `test_contracts_pass_tiny`: `run_contract_checks` passes (fp32, tol 1e-5), including batch=2 left-padded `check_route_plumbing`.
- `test_empty_rows_equal_baseline`: `matrix` with an all-False row ⇒ that row's logits `allclose(atol=1e-6)` to `hooks_disabled()` logits.
- `test_steer_scale_equals_gate_freeze`: `set_steer_scale(0.3)` vs `set_gate_freeze(True, 0.3)` (with `steer_scale=1.0`) ⇒ same logits.
- `test_desired_k_matches_realized`: `only` override over 5 layers ⇒ `_desired_k == 5`.
- `test_override_all_raises`.
- `test_counterfactual_mode_disables_dropout`: with `lora_dropout=0.5`, two calls inside `counterfactual_mode()` give identical logits; state restored afterwards.

---

### TASK M1 — Counterfactual Table Builder (S0 / S0b engine)

**Target files:** `src/routing/catalog.py` (+), `src/data/shots.py` (+), `src/analysis/__init__.py` (+), `src/analysis/cf_table.py` (+), `src/configs/config.py` (✎), `tests/test_cf_table.py` (+).
**Prerequisites:** M0, M6a.

**Core changes & logic**

*Config block `# ==== M1 ====`:*
```python
catalog_layers: Tuple[int, ...] = (13, 15, 18, 20, 22)   # must be ⊂ model.target_layers (else ValueError)
catalog_max_k: int = 3
cf_table_dir: str = "./checkpoints/cf_tables"            # under checkpoints/ ⇒ git-ignored
cf_shard_size: int = 25
cf_rows_per_forward: Tuple[Tuple[int,int], ...] = ((0, 13), (2, 8), (4, 7), (8, 5))   # (n_shot, rows per forward)
cf_audit_plan: Tuple[Tuple[int,int], ...] = ((0, 300), (4, 150), (8, 150))            # (n_shot, n_rows)
cf_heldout_n: int = 120                                  # 0-shot rows from eval_data for in-sample-inflation check + L13 repro
```
(Legacy `catalog_include_pairs`, `catalog_consecutive_triples`, `router_catalog_samples` stay, documented as "legacy restricted catalog, unused by exhaustive catalog".)

*`catalog.py`:*
```python
@dataclass(frozen=True)
class Catalog:
    layers: Tuple[int, ...]
    max_k: int
    subsets: Tuple[Tuple[int, ...], ...]          # index 0 == (); then by (len, lexicographic)
    @classmethod
    def build(cls, layers: Sequence[int], max_k: int = 3) -> "Catalog"        # itertools.combinations
    def __len__(self) -> int                                                    # 26 for 5 layers, K<=3
    def index_of(self, subset: Sequence[int]) -> int
    def sizes(self) -> torch.LongTensor                                         # (C,)
    def membership(self) -> torch.BoolTensor                                    # (C, len(layers))
    def to_candidate_matrix(self, class_idx: torch.LongTensor, target_layers: Sequence[int]) -> torch.BoolTensor  # (B, n_candidates)
```
`assert len(Catalog.build([13,15,18,20,22],3)) == 26` (1+5+10+10).

*`shots.py`:*
```python
def from_dataset_row(row: Mapping) -> dict          # strips padding; question_span_mask = labels==-100 & attention_mask (0-shot ⇒ whole prompt)
def build_shot_example(tokenizer, question: str, steps: List[str], final_num: str, n_shot: int,
                       max_len: int = 1536, final_style: str = "auto") -> dict
    # prefix = verifier.build_prompt(question, n_shot) (reuse; matches benchmark prompts)
    # target = steps rendered with STEP_PREFIXES[0]; final line: "#### ans" if n_shot==0 else "Final answer: ans" (matches exemplars)
    # tokenize FULL text with return_offsets_mapping; target tokens = offset_start >= target_char_start;
    # question_span = tokens with start in [last "Question:" char index, target_char_start)
    # returns dict(input_ids, attention_mask, labels, question_span_mask, n_shot) as 1-D tensors, NO padding
def question_span_mask_from_prompts(tokenizer, prompts: List[str], padding_side: str = "left", max_len: int = 2048) -> torch.BoolTensor
    # for generation-time left-padded batches (used by M6b / M9)
```
*`cf_table.py`:*
```python
@dataclass
class CFTable:
    catalog: Catalog
    qid: np.ndarray            # (N,) int   dataset row index in its source split
    split: np.ndarray          # (N,) '<U8' 'train'|'heldout'
    n_shot: np.ndarray         # (N,) int
    steer_scale: float
    ans_nll: np.ndarray        # (N, C) float32  == per_example_answer_nll semantics (mean over target tokens)
    token_nll: np.ndarray      # (N, C, T) float16, padded; per-token NLL for ALL subsets (≈12 MB @600 rows)
    token_mask: np.ndarray     # (N, T) bool
    token_ids: np.ndarray      # (N, T) int32
    pooled: np.ndarray         # (N, d) float16   router pooled state, question-span, pre-steering (F8)
    scalars: np.ndarray        # (N, 3) float32   router (mean_rms, rms_std, coverage)
    meta: dict                 # git commit, ckpt sha256, cfg subset, steer_scale, seed, schema_version
    def rewards(self, c: float = 1e-3) -> np.ndarray              # R = -ans_nll - c*|S|
    def gain(self) -> np.ndarray                                  # ans_nll[:,0:1] - ans_nll
    def select(self, mask: np.ndarray) -> "CFTable"
    def save(self, path: str) -> None; @classmethod load(cls, path: str) -> "CFTable"

def load_steernets_for_audit(model, path: str) -> dict           # F6: keep only "steer_nets." keys (flat or {"steer_state":...});
    # missing admitted layer => FatalContractError(E_CKPT_CORRUPT); non-admitted missing => WARN; returns report incl. sha256
def sample_audit_examples(train_data, eval_data, tokenizer, cfg, seed: int = 0) -> List[dict]
    # NESTED design: pick cfg_audit_plan[0] questions at 0-shot; the first 150 are ALSO rendered at 4 and 8 shot
    # (=> 300+150+150=600 rows, shot-paired questions for the consistency loss). Draw from train_data; plus
    # cfg.cf_heldout_n 0-shot rows from eval_data tagged split='heldout'. Never touch the benchmark test split.
def build_cf_table(model, tokenizer, examples: Sequence[dict], catalog: Catalog, cfg, device,
                   *, steer_scale: float, out_dir: str, seed: int = 0) -> CFTable
```
*`build_cf_table` algorithm:*
1. `model.eval()`; assert no submodule `.training`; `with model.counterfactual_mode():` `model.set_steer_scale(steer_scale)`; `model.set_routing_override("matrix", …)`.
2. For each example: build `rows_per_forward` chunks of the 26 subsets; replicate the example `rows` times (no padding needed; identical length); `matrix = catalog.to_candidate_matrix(class_idx_chunk, model.target_layers)`; one forward; slice logits at target positions **before** upcasting; `F.cross_entropy(..., reduction="none")` → per-token NLL; mean ⇒ `ans_nll`.
3. Set `model.set_router_context_mask(question_span_mask)` before the first chunk; read `router._last_pooled`, `_last_scalars` after it (identical for all rows).
4. Shard to `cf_table_part_{k:04d}.npz` every `cf_shard_size` examples; on restart skip example keys `(split,qid,n_shot)` already present; refuse to mix shards with different `meta` hash (`ValueError`). Final `merge_shards` → `cf_table.npz`.
5. Write `manifest.stage("S0")` metrics (throughput, rows/s).
6. `pair_synergy(table) -> dict` diagnostic only (`NLL[S∪{j}] − NLL[S] − (NLL[{j}] − NLL[∅])`); never a loss.

**Conflict isolation:** do not alter `gsm8k_to_cot` or `PromptTemplateBank`; reuse `verifier.build_prompt`. No training code is touched. Table files must live under `checkpoints/` (gitignored; `*.npz` is not in `.gitignore`).

**Verification:** `pytest tests/test_cf_table.py -q`
- `test_catalog_26`: sizes `[1,5,10,10]`, index 0 is `()`, `to_candidate_matrix` round-trips.
- `test_empty_subset_equals_base` (fp32 tiny): `ans_nll[:,0]` equals NLL from `hooks_disabled()` forward, `atol=1e-5`.
- `test_row_independence`: for 3 random subsets, `ans_nll[:,c]` equals NLL from an independent single-row forward with `set_routing_override("only", layers)`, `atol=1e-5` (proves batching-by-subset is exact).
- `test_eval_mode_enforced`: with `lora_dropout=0.5`, building twice gives identical tables.
- `test_resume_roundtrip`: interrupt after 1 shard, resume, arrays equal an uninterrupted run.
- `test_shot_example_spans`: `labels != -100` ⊂ target span; `question_span_mask` excludes exemplar tokens; target for `n_shot>0` ends with "Final answer: …".
- **Manual acceptance (Kaggle, recorded in manifest):** L13-singleton token-weighted NLL over `eval_data` equals the dev run's Phase-1 val figure (0.8985) within ±0.003; empty-set NLL equals baseline NLL within 1e-3 (fp16).

---

### TASK M2 — Signal Audit Report (Decision 1 engine)

**Target files:** `src/analysis/signal_audit.py` (+), `scripts/run_signal_audit.py` (+), `docs/AUDIT_REPORT_S0.md` (generated), `src/configs/config.py` (✎), `tests/test_signal_audit.py` (+).
**Prerequisites:** M1, M0.

**Core changes & logic**

*Config block `# ==== M2 ====`:* `audit_g_fixed_min: float = 0.01`, `audit_h_low: float = 0.005`, `audit_h_high: float = 0.02`, `audit_g_zero_eps: float = 0.005`, `audit_n_boot: int = 2000`, `audit_inflation_margin: float = 0.01`.

```python
@dataclass
class ShotAudit:
    n: int
    g_fixed_star: float; s_fixed_star: Tuple[int, ...]; g_fixed_ci: Tuple[float, float]
    g_fixed_crossfit: float             # 2-fold cross-fitted (choose on fold A, score on fold B), mean of 20 random splits
    g_oracle: float; g_oracle_ci: Tuple[float, float]
    headroom_H: float; headroom_ci: Tuple[float, float]   # bootstrap re-selects S_fixed* inside each resample
    frac_empty_best: float; frac_best_fixed_hurts: float
    single_layer_gain: Dict[int, float]; best_gain_by_k: Dict[int, float]
    oracle_choice_entropy_bits: float; mean_near_tie_set_size: float   # |E_n| at eps=0.01
    k0_diagnosis: str                   # "rational_abstention" | "feature_failure_possible" | "signal_present"
@dataclass
class AuditResult:
    per_shot: Dict[int, ShotAudit]; overall: ShotAudit
    heldout: Optional[ShotAudit]; flags: List[str]; decision_1: str; decision_2_inputs: Dict[str, float]

def compute_signal_audit(table: CFTable, cfg, *, seed: int = 0) -> AuditResult
def write_audit(result: AuditResult, manifest: RunManifest, out_json: str, out_md: str) -> None
```
*Quantities (plan II.0):* `G_fixed* = max_c mean_n gain[n,c]`; `G_oracle = mean_n max_c gain[n,c]`; `H = G_oracle − G_fixed*`; per shot and overall; CIs by example-level bootstrap (vectorized over the `(N,C)` matrix). `k0_diagnosis`: `G_fixed*<eps ∧ H<eps ⇒ rational_abstention`; `G_fixed*<eps ∧ H≥0.01 ⇒ feature_failure_possible`; else `signal_present`.

*Advisory flags (written with `manifest.advisory`, never raised):*
- `SWITCH_TRAINING_TARGET` if `G_fixed*(0-shot) < audit_g_fixed_min`.
- `ROUTING_HEADROOM_OK` if any shot `H ≥ audit_h_high`; `ROUTING_NO_HEADROOM` if **all** shots `H < audit_h_low`; otherwise `ROUTING_HEADROOM_INTERMEDIATE`.
- `MULTISHOT_NONINFERIORITY_ONLY[shot]` if shot∈{4,8} and `G_fixed* < audit_g_zero_eps`.
- `W_INSAMPLE_INFLATION` (WARN) if `G_fixed*(train 0-shot) − G_fixed*(heldout 0-shot) > audit_inflation_margin` (F9). **Decision-1 thresholds are evaluated on held-out + cross-fit values when they disagree with in-sample by more than the margin**, and the report prints both.
- `decision_1` string ∈ {`"BRANCH_A: SWITCH_TRAINING_TARGET -> M3, M4, rebuild table"`, `"BRANCH_B: PROCEED -> M4 (layer-dropout), rebuild table"`}.

`scripts/run_signal_audit.py` orchestrates: load config/model → `load_steernets_for_audit` → `run_contract_checks` (fatal on failure) → `sample_audit_examples` → `build_cf_table` → `compute_signal_audit` → `write_audit`. Flags: `--ckpt --out_dir --steer_scale --n_boot --resume`.

**Conflict isolation:** pure analysis; no model code edited. Output wording must state clearly that `G_oracle` is an upper bound on routing and that thresholds are advisory.
**Verification:** `pytest tests/test_signal_audit.py -q`
- All-zero gains ⇒ `G_fixed*=0`, flag `SWITCH_TRAINING_TARGET`, `k0_diagnosis=="rational_abstention"`.
- Planted structure: half the rows gain 0.05 only for subset a, half only for subset b ⇒ `G_fixed*=0.025`, `G_oracle=0.05`, `H=0.025` (±1e-6); CI brackets it.
- `G_fixed_crossfit ≤ G_fixed_star` on a random-noise table.
- Manifest gets advisories; function never raises on degenerate tables (N=1, C=1).
- `python scripts/run_signal_audit.py --help` exits 0.

---

### TASK M3 — On-Policy Reasoning Trace Sampler

**Target files:** `src/data/onpolicy.py` (+), `scripts/run_onpolicy_sampling.py` (+), `src/configs/config.py` (✎), `tests/test_onpolicy.py` (+).
**Prerequisites:** M0, M6a (`hooks_disabled`). Code may be written in parallel with M1; **do not run** before DECISION 1.

*Config block `# ==== M3 ====`:* `onpolicy_n_questions=1500`, `onpolicy_samples=(2,4)` (adaptive: 2 first, +2 for questions with 0 correct), `onpolicy_temperature=0.7`, `onpolicy_max_new_tokens=320`, `onpolicy_shots_mix=(0,4)`, `onpolicy_batch_size=16`, `onpolicy_max_tok_len=1280`, `onpolicy_gold_mix=0.4`.

```python
@dataclass
class OnPolicyChain:
    qid: int; sample_idx: int; shot: int; prompt: str; text: str
    pred_strict: Optional[float]; pred_lenient: Optional[float]; gold: float; correct: bool
    n_tokens: int; div_line: Optional[int]        # index of first line that differs from a correct sibling (pairs only)
class BatchAnswerStopping(StoppingCriteria):      # per-row (returns BoolTensor(B)); stops a row on EOS, "####<num>", "Final answer: <num>",
    def __init__(self, tokenizer, prompt_len: int, grace_tokens: int = 12): ...   # or on a new "\nQuestion:" (runaway continuation)
class OnPolicyStore:
    def save(self, path: str); @classmethod load(cls, path: str)                   # JSONL, under checkpoints/ (gitignored)
    def correct_chains(self) -> List[OnPolicyChain]; def pairs(self) -> List[Tuple[OnPolicyChain, OnPolicyChain]]
    def stats(self) -> dict                                                        # n_correct, n_questions_with_pair, pass rate per shot
    def export_spotcheck(self, n: int, path: str, seed: int) -> None              # CSV with human_pred column
    @staticmethod
    def spotcheck_agreement(path: str) -> float
    def build_training_rows(self, tokenizer, cfg, gold_rows: Sequence[dict]) -> Tuple[List[dict], List[dict]]  # (train_rows, val_rows)
def sample_chains(model, tokenizer, questions: Sequence[Mapping], cfg, device, *, seed: int, out_path: str) -> OnPolicyStore
```
*Logic:* sample with `model.hooks_disabled()` and `model.eval()`, left padding (restore tokenizer `padding_side` in `finally`), `do_sample=True, temperature=0.7`, prompts built by `verifier.build_prompt(q, shot)` alternating shots in the mix; truncate generated text at the first `\nQuestion:`; verify with `extract_final_answer` (strict = priorities 1–3 only; lenient = full function + last-number fallback) against gold `final_num`; `correct` uses **strict-or-lenient agreeing with gold** and stores both. `build_training_rows`: row = prompt+chain, `labels=-100` on prompt (exemplars shielded), carries `is_onpolicy`, `shot`, and for pairs `neg_input_ids`, `neg_labels`, `div_token_mask` (tokens of the first divergent line of the incorrect chain, found by line-wise diff or the first line failing `extract_and_verify_equations`). Mix gold rows so gold share = `onpolicy_gold_mix`. Resumable: append to JSONL every batch; on restart skip finished `(qid, sample_idx)`.

**Conflict isolation:** do not modify `AnswerStoppingCriteria` or `generate_response` (single-sequence, used by the legacy benchmark). New stopping criterion lives in `onpolicy.py`; M4/M9 import it.
**Verification:** `pytest tests/test_onpolicy.py -q`
- `BatchAnswerStopping`: rows stop independently on synthetic ids.
- Sampling on `tiny_model` is deterministic under a fixed seed and restores `padding_side`.
- Hooks-off sampling equals `generate` of the unwrapped base model (greedy, tiny fp32).
- `build_training_rows`: labels are `-100` exactly on prompt tokens; `div_token_mask ⊂ neg_labels != -100`.
- **Manual acceptance (Kaggle):** ≥1000 verified-correct chains in the store stats; `spotcheck_agreement ≥ 0.98` else `W_EXTRACTION_AGREEMENT`.

---

### TASK M4 — S1′ SteerNet Subset Trainer (on-policy + layer-dropout)

**Target files:** `src/training/trainer.py` (✎, add `train_s1_steernet`, `run_k_scaling_ablation`), `src/training/losses.py` (✎), `src/evaluation/paired_generation.py` (+), `scripts/run_s1.py` (+), `src/configs/config.py` (✎), `tests/test_s1_trainer.py` (+).
**Prerequisites:** M0, M6a, M3 (data; gold-only fallback allowed), M2/DECISION 1 recorded.

*Config block `# ==== M4 ====`:* `s1_epochs=4`, `s1_lr=5e-3` (same as `phase1_lr`), `s1_onpolicy_fraction=0.6`, `s1_lambda_unl=0.1`, `s1_unl_every_n_steps=4`, `s1_calc_token_weight=1.75`, `s1_val_onpolicy_n=120`, `s1_k_choices=(1,2,3)`, `s1_save_path="./checkpoints/s1_steernet.pt"`, `s1_final_gen_n=240`, `s1_ablation_epochs=1`, `s1_ablation_n_train=1000`.

```python
def sample_training_subset(catalog: Catalog, rng: random.Random, k_choices=(1,2,3)) -> Tuple[int, ...]
    # K ~ Uniform(k_choices), then subset ~ Uniform over C(5,K); never empty
def compute_calc_token_weights(tokenizer, labels: Tensor, ids: Tensor, w: float = 1.75) -> Tensor
    # weight w on tokens whose id is in contrastive._CONTRAST_MATH_IDS (digits/operators/=/$/%/.), 1.0 elsewhere, 0 on labels==-100
    # If the dev branch already ships `calculation_weighted`, import and reuse it instead (A0 item 4).
def weighted_answer_nll(logits, labels, weights) -> Tensor
def unlikelihood_on_divergence(logits_neg, neg_labels, div_token_mask) -> Tensor      # mean[-log(1-p(tok))] over divergent-line tokens; ≥0
def collate_pad(rows: Sequence[dict], pad_id: int) -> dict                             # rows may have different lengths (4-shot rows ≈ 900 tok)
def train_s1_steernet(model, tokenizer, gold_rows, onpolicy_rows, sel_rows, cfg, device) -> Dict[str, Any]
def run_k_scaling_ablation(model, tokenizer, rows, sel_rows, cfg, device) -> Dict[str, Any]
def evaluate_subset_nll(model, rows, catalog, cfg, device) -> np.ndarray              # (len(rows), C) via counterfactual_mode; reused by S0b checks
```
*`train_s1_steernet` logic:*
1. Freeze router and gate (gate bypassed: `model.set_steer_scale(cfg.steer_scale)`); `requires_grad=True` only on `steer_nets.*`. Base stays frozen.
2. Per micro-batch: draw a source (on-policy with prob `s1_onpolicy_fraction`, else gold); draw a subset with `sample_training_subset`; `model.set_routing_override("matrix", model.subset_matrix([subset]))`; forward with grad; loss = `weighted_answer_nll`. Every `s1_unl_every_n_steps`, if the row has a pair, add `s1_lambda_unl · unlikelihood_on_divergence` using a second forward on the incorrect chain **under the same subset**.
3. AdamW/`LRReducer`/GradScaler/clipping exactly as `train_phase1_steernet` (reuse code via small helper, do not copy-paste divergent behavior).
4. **Selection = teacher-forced only:** per epoch compute `evaluate_subset_nll` over all 25 non-empty subsets on `sel_rows` (on-policy val + gold val) and use the **mean NLL gain vs empty set** as score. **No per-epoch generation screens.**
5. After the final epoch: **one** paired batched generation check via `paired_generation.paired_greedy_eval(...)`, n=`s1_final_gen_n`, best-NLL subset vs baseline; results go to the manifest (`s1_final_gen_check`) as an advisory with discordant counts `(b, c)`.
6. Save with `save_steer_weights` plus metadata (steer_scale, git commit, `k_scale_mode`, `steernet_k_conditioning`).
7. `run_k_scaling_ablation`: for each of {`sqrt`,`none`} × {k-cond on, off}: short run (`s1_ablation_epochs`, `s1_ablation_n_train`), score by mean val NLL gain with K uniform; record all four to the manifest; set the winners as **advisory recommendations** (`manifest.advisory("k_scaling_winner", …)`); the Head Agent applies them to config defaults after plan-owner sign-off.

*`paired_generation.py` (consumed again by M7/M8/M9):*
```python
def generate_batch(model, tokenizer, prompts: Sequence[str], device, *, max_new_tokens: int = 512,
                   question_span_mask: Optional[Tensor] = None) -> List[str]
    # greedy; SAME settings as generate_response: repetition_penalty=1.1, no_repeat_ngram_size=6, use_cache=True;
    # left padding; BatchAnswerStopping; post-trim identical to generate_response's regex truncation
def paired_greedy_eval(model, tokenizer, questions, golds, configure_a, configure_b, device, **kw) -> Dict[str, Any]
    # returns {"acc_a","acc_b","b","c","delta","per_item": [...]}
```
**Conflict isolation:** keep `train_phase1_steernet`, `train_router_bootstrap`, `run_staged_phase`, `train_rsc` untouched (legacy). Do not remove the `K/4.0` constant in `RSCSteerNet.forward`. Do not change `generate_response`.
**Verification:** `pytest tests/test_s1_trainer.py -q`
- Subset sampler: over 3000 draws each K∈{1,2,3} within ±5 % of 1/3; no empty subsets.
- Tiny-model overfit: 8 rows, 40 steps ⇒ mean NLL falls ≥20 %; hashes of base-model params, router params, gate params unchanged; `steer_nets` params changed.
- `compute_calc_token_weights`: weight 1.75 exactly on math ids, 0 on `-100`.
- `unlikelihood_on_divergence ≥ 0` and `→ 0` when the divergent token prob → 0.
- `generate_batch` with batch 2 equals two batch-1 calls on `tiny_model` fp32 (greedy).
- **Kaggle acceptance (manifest):** on the on-policy validation set the S1′ checkpoint's L13-singleton NLL < the Phase-1 checkpoint's; no generation screens executed before the final check.

---

### TASK M5 — 26-Subset Catalog Router + Distillation

**Target files:** `src/routing/router.py` (✎), `src/routing/distill.py` (+), `src/routing/rloo.py` (✎), `scripts/run_router_distill.py` (+), `src/configs/config.py` (✎), `tests/test_distill.py` (+).
**Prerequisites:** M1 (`Catalog`, `CFTable` with `pooled/scalars`), M0, DECISION 2 recorded. (No wrapper dependency: S2 is table-only.)

*Config block `# ==== M5 ====`:* `router_policy="catalog"` (`"legacy"|"catalog"`), `router_c_size=1e-3`, `router_eps_tie=0.01`, `router_tau=0.01`, `router_lambda_bal=0.02`, `router_lambda_cons=0.1`, `router_distill_epochs=60`, `router_distill_lr=1e-3`, `router_distill_wd=1e-2`, `router_distill_val_frac=0.2`, `router_catalog_save_path="./checkpoints/router_catalog_v5.pt"`, `router_loss_mode="catalog_distill"` (`"rloo_legacy"` keeps the old path).

*`router.py` additions (existing `forward` untouched):*
```python
# in __init__(…, n_catalog: Optional[int] = None, shot_buckets: Tuple[int,...] = (0,2,4,8)):
self.catalog_head = nn.Linear(d_model, n_catalog)            # zero-init weight+bias  => uniform logits; argmax tie-break => class 0 (abstain) at init
self.register_buffer("bucket_mean", zeros(len(shot_buckets), hidden_dim)); self.register_buffer("bucket_var", ones(...))
self.register_buffer("bucket_count", zeros(len(shot_buckets)))
def catalog_logits_from_features(self, pooled: Tensor, scalars: Tensor, shot_bucket: Optional[LongTensor] = None,
                                 update_stats: bool = False) -> Tensor      # (B, C)
    # pooled -> per-bucket standardization (running mean/var; fall back to pooled stats if bucket unseen) -> _semantic_from_pooled ->
    # _score_from_base tokens -> mean-pool attended_2 -> catalog_head. Also sets self._last_semantic_shared (gate context).
def catalog_logits(self, h: Tensor, valid_mask: Tensor, shot_bucket=None) -> Tensor   # computes features via _base_features then the above
```
No shot-count feature is fed to the network (plan II.3.3: `router_demo_features` removed from this path); the bucket only selects standardization statistics.

*`distill.py`:*
```python
def build_soft_targets(rewards: Tensor, eps: float, tau: float) -> Tuple[Tensor, Tensor]
    # R*_n; E_n = {S: R >= R*_n - eps}; q_n ∝ exp(R/tau) restricted to E_n. Returns (q (N,C), near_opt (N,C) bool)
def kl_q_p(q: Tensor, log_p: Tensor) -> Tensor                # sum q*(log q.clamp_min(1e-12) - log_p), mean over N
def balance_loss(p: Tensor, multi_opt_mask: Tensor) -> Tensor # Switch: C * sum_c f_c * P_c over rows with |E_n|>=2 only; 0 if none
def consistency_loss(p_ref: Tensor, p_other: Tensor) -> Tensor# KL(p_ref.detach() || p_other)  (p_0 is the teacher; see §5 D9)
@dataclass
class DistillReport: train_loss, val_loss, top1_agreement, mean_regret, regret_best_fixed, regret_ratio, route_entropy_bits_by_shot, abstain_rate_by_shot, class_hist
def distill_router(router, table: CFTable, cfg, device, manifest, *, class_mask: Optional[Sequence[int]] = None, seed: int = 0) -> DistillReport
    # SteerNets are not touched (no LLM forwards). Split train/val BY qid (shot-paired copies never straddle).
    # Loss = KL(q||p) + lambda_bal*balance + lambda_cons*consistency (pairs: same qid at two shots, equal bucket weights).
    # class_mask: restrict logits to these classes (others -inf) -> "fixed_abstain" mode = [0, index_of(S_fixed*)] for DECISION 2 (H<0.005).
    # Early stop on val KL; save best router state to cfg.router_catalog_save_path
def router_eval_metrics(router, table, idx, cfg) -> dict      # top-1 vs table argmax; mean regret (R*-R[chosen]); regret vs best fixed; entropy; abstain per shot
def pair_synergy_diagnostics(table) -> dict                   # re-export of cf_table.pair_synergy
```
`rloo.py` edit (flag path only): wrap the two counterfactual forwards in `with model.counterfactual_mode():` (fixes F1), and keep `compute_router_rloo_loss` otherwise intact. MI / synergy-regression / K-preference loss terms in the dev branch (A0 inventory) get flags defaulting to **0 / off**; they are not deleted. An optional RLOO polish (≤1 epoch, lr 1e-4, N≥4 samples with leave-one-out baseline, `eval()` SteerNets) is **out of scope** unless the plan owner requests it after distillation plateaus with `H ≥ 0.02`.

**Conflict isolation:** `JointLayerRouter.forward` arity and behavior unchanged (`tests/test_modules.py`). Legacy K-head and `sample_*` remain. No advisory metric may raise.
**Verification:** `pytest tests/test_distill.py -q`
- `build_soft_targets`: ties split mass equally; a unique clear winner yields a one-hot (±1e-6).
- `balance_loss == 0` when no row has `|E_n|≥2`; positive for a collapsed argmax over tied rows.
- Synthetic table with a planted per-row best subset linearly encoded in `pooled` ⇒ held-out `top1_agreement ≥ 0.9`, `regret_ratio ≤ 0.5`.
- `class_mask=[0, j]` ⇒ router never selects other classes.
- Fresh router argmax equals class 0; existing `tests/test_modules.py` passes unchanged.
- Kaggle acceptance (advisory): `regret_ratio ≤ 0.5` on held-out rows else `W_ROUTER_REGRET`; **no run-time assertion on entropy**.

---

### TASK M6b — Wrapper Plumbing (remaining)

**Target files:** `src/models/wrapper.py` (✎), `src/configs/config.py` (✎), `tests/test_wrapper_catalog.py` (+).
**Prerequisites:** M6a, M5 (`catalog_logits`), M1 (`Catalog`).

*Config block `# ==== M6b ====`:* `abstain_theta=1.01` (`>1` disables), `guards_enabled=True`, `guard_rel_norm_cap=0.02` (calibrated in M8), `guard_k_max=3`.

**Core changes & logic:**
1. **Question-span plumbing:** `set_question_span_mask(mask)` (alias of `set_router_context_mask`, same pending→active semantics) and `set_shot_bucket(b: Union[int, LongTensor])`. Generation callers supply masks built by `shots.question_span_mask_from_prompts` (left-padded).
2. **Catalog policy branch in `_prepare_routes`** (when `cfg.router_policy=="catalog"` and no override and not warmup): `logits = router.catalog_logits(h, router_mask, shot_bucket)`; `p = softmax`; `idx = argmax` (or sample if `_router_inference_mode=="sample"`; **no training-time stochastic sampling in this branch**); abstain rule: `if idx != 0 and p[:,0] >= abstain_theta: idx = 0`; `selected_matrix = catalog.to_candidate_matrix(idx, target_layers)`; `_desired_k = catalog.sizes()[idx]`; update `router.selected_forwards/seen_forwards`; store `_router_probabilities` (p over catalog) for traces. `set_catalog(catalog)`, `set_abstain_threshold(theta)`, `set_router_policy(mode)` are the setters. The legacy branch is untouched.
3. **Task-loss gradient flag:** `alpha_eff = (alpha if cfg.gate_task_gradient else alpha.detach()) * cfg.steer_scale` (the line M6a left as `.detach()`). Gate input `delta` stays detached inside the gate (unchanged).
4. **Guards:** `set_guards(enabled: bool, rel_norm_cap: float, k_max: int)`. With guards enabled the wrapper only **records** `guard_state["tripped"] = (rel_max > cap) | nonfinite | (K > k_max)` per item (exact revert is performed by the runner re-generating tripped items under `override="none"`; see M9). With guards disabled (Raw ARS) the same state is still recorded for the trace but never acted on.
5. **`predict_routes(input_ids, attention_mask, question_span_mask=None, shot_bucket=None) -> List[dict]`:** one `no_grad` prefill under `counterfactual_mode`; returns per row `{route: Tuple[int,...], class_idx, p_empty}`; clears decode caches afterwards so a subsequent `generate` recomputes identically (test).
6. **`route_key(row) -> Tuple[Tuple[int,...], str, Optional[float]]`** = `(route_tuple, gate_mode, steer_scale_or_None)`, canonicalized so any empty route returns `((), "none", None)` (baseline reuse).
7. **Gate-mode accessor:** `set_gate_mode(mode: str, scale: Optional[float])` with `mode ∈ {"learned","const"}` (`const` ⇒ `set_steer_scale(scale)`); property `gate_mode`; value is written to `manifest.set("gate_mode", …)`. The *labels* `bypassed` / `passthrough` / `learned` used in tables are assigned by M8/M9 from the contract.
8. **Batch>1 / left-padding / cached-decode correctness** for the catalog branch (risk register IV.4): decode-cache keys are per batch row (already how `_cached_decode_routes` is stored); add a regression test rather than refactoring.

**Conflict isolation:** do not alter the legacy routing branches, the warmup logic, or `observe`. Do not make guards raise. Keep `set_router_exploration` as a no-op.
**Verification:** `pytest tests/test_wrapper_catalog.py -q`
- Plan tests III-1..III-3 re-run with `router_policy="catalog"` at batch 2, left padded, 4 decode steps (`run_contract_checks`).
- Degenerate router (head biased so class `{L}` always wins) ⇒ greedy outputs and logits identical to `set_routing_override("only",{L})` (precursor of III-4).
- `predict_routes` equals `current_route_tuples()` observed during a subsequent `generate`.
- `abstain_theta` rule: with `p0=0.6, argmax≠0`: `theta=0.5 ⇒ empty`, `theta=1.01 ⇒ argmax`.
- `gate_task_gradient=True` ⇒ `gate.fc2.weight.grad` non-zero after backward; `False` ⇒ `None`.
- Guard: tiny cap ⇒ `guard_state["tripped"]` true for steered rows, false for K=0 rows.
- Question-span plumbing: router `coverage` scalar equals question-token fraction, independent of exemplar length.

---

### TASK M7 — Direct Task-Loss Token Gate, Dominance Hinge (S3)

**Target files:** `src/gating/gate.py` (✎), `src/training/losses.py` (✎), `src/training/contrastive.py` (✎ read-only reuse), `src/training/trainer.py` (✎, add `train_s3_gate`), `scripts/run_s3_gate.py` (+), `src/configs/config.py` (✎), `tests/test_gate_stage.py` (+).
**Prerequisites:** M6b, M5 (trained router), M4 (S1′ SteerNets), M1 (S0b table with `token_nll`).

*Config block `# ==== M7 ====`:* `gate_init_logit=2.0` (α_init=σ(2)=0.8808; takes precedence over `gate_init_alpha`, which is retained), `gate_task_gradient=True` (stage override), `gate_loss_mode="v5"` (`"legacy"` keeps old terms), `lambda_dom=1.0`, `lambda_tok=0.3`, `lambda_l1_gate_v5=0.005`, `gate_tau_u=0.005`, `gate_dom_margin=0.0`, `gate_epochs=3`, `gate_lr=1e-3`, `gate_label_row_frac=0.5`, `gate_v5_save_path="./checkpoints/gate_v5.pt"`, `gate_sel_n=120`.

*`gate.py`:* `init_logit = cfg.gate_init_logit if cfg.gate_init_logit is not None else logit(cfg.gate_init_alpha)`; `fc2.weight` stays zero (gate starts as nearly pass-through and learns deviations). Docstring updated; **the stale −6.0 statement in `docs/methodology.md` is corrected in M10**.

*`losses.py` additions:*
```python
def dominance_hinge(nll_gated: Tensor, nll_ungated: Tensor, margin: float = 0.0) -> Tensor   # per-example relu(g - u.detach() + m), mean
def counterfactual_token_labels(table: CFTable, router_choice: np.ndarray, tau_u: float) -> "TokenLabelStore"
    # u_t = token_nll[n,0,t] - token_nll[n,c_n,t] where c_n = the (frozen) router's argmax class for row n; label 1[u_t > tau_u]; rows with c_n==0 have no label
class TokenLabelStore:  # keyed (split,qid,n_shot) -> (labels BoolTensor[T_ans], weights)
    def get(self, key) -> Optional[Tuple[Tensor, Tensor]]
def gate_bce_tokens(model, labels: Tensor, label_store_rows, ...) -> Tensor     # weighted BCE on gate._alpha_for_aux over selected layers; weights balance pos/neg
def gate_total_loss(nll_gated, nll_ungated, bce_tok, mean_alpha, cfg) -> Tuple[Tensor, dict]
    # NLL_gated + lambda_dom*L_dom + lambda_tok*BCE + lambda_l1*mean(alpha)
def gate_label_auc(model, rows, store, ...) -> float                             # advisory diagnostic only
def paired_gate_vs_ungated(model, rows, cfg, device) -> Dict[str, Any]           # per-example NLL arrays, mean diff, bootstrap CI
```
*`train_s3_gate` logic:* freeze SteerNets and router (`router_policy="catalog"`, argmax, `abstain_theta` as calibrated or 1.01); trainable = gate params only; `cfg.gate_task_gradient=True`. Per micro-batch: (i) gated forward with grad (`gate_mode="learned"`) → `nll_gated`; (ii) ungated reference under `counterfactual_mode()` with the **same route** and `set_steer_scale(cfg.steer_scale)` (full pass-through, α=1·s_ref) → `nll_ungated` (no grad); (iii) hinge; (iv) BCE only for rows present in `TokenLabelStore` (a fraction `gate_label_row_frac` of micro-batches are drawn from labeled table rows, the rest from the S1′ training rows); (v) mean-α L1 using the existing `compute_l1_gate`. **Removed from the v5 objective** (kept in code, selected by `gate_loss_mode="legacy"`): diversity, anti-saturation, contrastive token-class targets (0.85/0.05/0.40), `lambda_pg`. Advisory diagnostics logged: gate-vs-label AUC; mean α on `_CONTRAST_MATH_IDS` tokens vs other tokens. Selection: paired NLL on `gate_sel_n` selection rows after each epoch, early-stop on `paired_gate_vs_ungated` mean diff; record `gate_adopted_nll = (mean_diff < 0)` and, if false, `manifest.warn("W_GATE_NOT_ADOPTED")` and keep the checkpoint (final adoption incl. accuracy is decided in M8).

**Conflict isolation:** `run_staged_phase` and legacy gate losses untouched. The gate still receives `delta.detach()` as an *input* feature. SteerNet gradients stay blocked because SteerNets are frozen. Do not reuse `compute_pg_loss*`.
**Verification:** `pytest tests/test_gate_stage.py -q`
- Init: `sigmoid(fc2.bias) == 0.8808 ± 1e-3`; mean α over random inputs ≈ 0.88.
- Task gradient: with `gate_task_gradient=True`, one backward gives non-zero `fc2.weight.grad` and `fc2.bias.grad`; SteerNet/router params have `grad is None`.
- `dominance_hinge`: zero when gated ≤ ungated, positive otherwise, gradient only through `nll_gated`.
- `counterfactual_token_labels`: synthetic table ⇒ labels equal `1[u>tau]`, rows with class 0 absent.
- Tiny-model smoke: 20 steps, loss finite, only gate params change (hash check).
- Kaggle acceptance: on the 120-row selection split `gated NLL ≤ ungated NLL` **or** `gate_not_adopted` appears in the manifest.

---

### TASK M8 — S5 Deployment Calibration (`deployment_contract.json`)

**Target files:** `src/deploy/__init__.py` (+), `src/deploy/calibrate.py` (+), `scripts/run_calibration.py` (+), `src/configs/config.py` (✎), `tests/test_calibrate.py` (+).
**Prerequisites:** M7, M6b, M4 (`paired_generation`), M1 (`Catalog`).

*Config block `# ==== M8 ====`:* `calib_theta_grid=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01)`, `calib_envelope_quantile=0.999`, `calib_nll_tie_eps=0.003`, `calib_sel_n=120`, `calib_gate_acc_tolerance=1`, `contract_path="./deployment_contract.json"`.

```python
def choose_fixed_subset(model, sel_rows, catalog, cfg, device) -> Tuple[Tuple[int,...], dict]
    # (a) rank all non-empty subsets by paired selection NLL (counterfactual_mode); candidates = those within calib_nll_tie_eps of best;
    # (b) among candidates compare paired greedy accuracy on sel rows (paired_greedy_eval); ties -> smaller K. Evidence dict returned.
def search_theta_abs(model, sel_rows, cfg, device) -> Tuple[float, List[dict]]      # grid over abstain_theta; max paired NLL gain, ties by accuracy, then larger theta (less abstention)
def measure_envelope(model, rows, cfg, device) -> dict                              # per-layer & global quantile of per-item mean ‖step‖/‖h‖ under learned routes+gate (uses guard_state)
def decide_gate_adoption(nll_paired: dict, acc_paired: dict, tol: int = 1) -> Tuple[str, dict]   # "learned" iff (mean NLL diff < 0) and (acc_gated >= acc_ungated - tol*1/n) else "passthrough"
def write_contract(path: str, **fields) -> dict;  def load_contract(path: str) -> dict
def apply_contract(model, contract: dict, which: str)                               # which in {"fixed","router","full_ars","raw_ars"}: sets steer_scale, gate mode, router_policy, theta, guards, K cap, k_scale_mode, steernet_k_conditioning
```
`deployment_contract.json` fields: `schema_version, created, git_commit, steer_scale, k_scale_mode, steernet_k_conditioning, catalog{layers,max_k}, s_fixed_star, fixed_evidence, theta_abs, theta_table, router{policy,inference_mode,ckpt,sha256}, gate{mode,adopted,evidence,ckpt,sha256}, envelope{rel_norm_cap,quantile,p50,p99,max,per_layer}, guards{k_max,finite_check,revert_to_k0}, steernet{ckpt,sha256}, selection_split{n,ids_hash}, manifest_path`. `W_ENVELOPE_MISMATCH` if measured cap differs from 0.02 by more than 2×. `apply_contract` + the contract alone **fully determine conditions 4 and 5** (test below).
**Conflict isolation:** no model/training code changes; the selection split ids must be disjoint from any S1′/S3 training ids and from the benchmark test split (assert via ids-hash, WARN on overlap).
**Verification:** `pytest tests/test_calibrate.py -q`
- Synthetic tie logic: two subsets within `calib_nll_tie_eps`, accuracy decides; equal accuracy ⇒ smaller K.
- Theta search picks the argmax of a planted gain curve; ties ⇒ larger θ.
- `decide_gate_adoption` truth table (NLL better/worse × acc diff −2,−1,0,+1).
- Contract round-trip: `apply_contract(model,"full_ars")` then `"raw_ars"` flips only `guards_enabled`/θ as specified; `load_contract(write_contract(...)) == original`.

---

### TASK M9 — 5-Arm Production Benchmark Runner (output cache, paired statistics)

**Target files:** `src/evaluation/multi_benchmark_runner.py` (✎ dev file `[verify]`; create if absent), `src/evaluation/benchmark.py` (✎), `src/evaluation/multi_dataset.py` (✎ `[verify]`), `src/evaluation/output_cache.py` (+), `src/evaluation/stats.py` (+), `src/evaluation/paired_generation.py` (✎ from M4), `scripts/run_benchmark.py` (+), `tests/test_benchmark_runner.py` (+).
**Prerequisites:** M8 (contract), M6b, M7, M4.

*Config block `# ==== M9 ====`:* `bench_shots=(0,8)`, `bench_secondary_shots=(2,4)`, `bench_secondary_n=500`, `bench_n_boot=10000`, `bench_batch_size=((0,8),(2,6),(4,4),(8,4))`, `bench_max_new_tokens=512`, `bench_cache_dir="./checkpoints/bench_cache"`, `bench_noninf_margin_pp=1.0`.

**Core changes & logic**

1. **Conditions (Part III, binding):**
```python
class Condition(IntEnum): BASELINE=1; FIXED=2; ROUTER=3; FULL_ARS=4; RAW_ARS=5
CONDITION_SPECS = {
 1: dict(route="none",    gate="n/a",     guards=False, theta=None),
 2: dict(route="fixed",   gate="bypassed",guards=False, theta=None),     # S_fixed* from contract, constant scale s
 3: dict(route="router",  gate="bypassed",guards=False, theta=None),     # argmax, abstain = class 0 only
 4: dict(route="router",  gate=contract_gate_mode, guards=True, theta=contract.theta_abs),
 5: dict(route="router",  gate=contract_gate_mode, guards=False, theta=None)}
def configure_for_condition(model, cond: Condition, contract: dict) -> None   # via apply_contract
```
   The legacy hierarchical arbiter (`adaptive → fixed_steernet → baseline`) is removed from the call path (leave a stub that raises `DeprecationWarning`). **Fixed no longer means "all 5 layers".**
2. **Run loop** (`run_benchmark(model, tokenizer, device, contract, *, shots, n, conditions, cache_dir, seed, resume=True) -> dict`):
   - `run_contract_checks` first (fatal).
   - Dataset order fixed by seed; secondary shots use a fixed random 500-question subset; primary = full 1319.
   - For each shot and condition: `predict_routes` over batches (sorted by prompt length; same sorted order and batch size for every condition) → keys via `route_key` → **group by key**, generate only keys absent from the cache, reuse otherwise. K=0 rows reuse the Baseline output.
   - Condition 4: after generation, rows with `guard_state.tripped` are regenerated under `override="none"` and logged per item (`guard_reverted=True`).
   - Condition 5: guards disabled; per-item routing/gate trace always recorded.
   - Prompts: `verifier.build_prompt(q, n_shot)`; decoding identical across conditions (`generate_batch`, same repetition settings as `generate_response`).
3. **`output_cache.py`:** namespace dir = hash of `(model name, steernet sha, router sha, gate sha, decode config, extractor version)`; per-shot JSONL `{key_hash, prompt_hash, route, gate_mode, steer_scale, text, n_tokens, guard_tripped}`; append every batch (resumable); canonicalization: empty route ⇒ `((),"none",None)`; constant-gate keys ignore gate weights; learned-gate keys include gate sha.
4. **Extraction:** both **strict** (`extract_final_answer` priorities 1–3 only) and **lenient** (existing `evaluate_response`) stored per item; tables report both (primary = lenient for continuity, §5 D4).
5. **`stats.py`:**
```python
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict   # delta (pp), ci95, b, c
def mcnemar_exact(b: int, c: int) -> float                      # two-sided exact binomial (scipy.stats.binomtest(min(b,c), b+c, 0.5))
def holm(pvals: Sequence[float], alpha: float = 0.05) -> List[dict]
def stratify_by_steps(correct_a, correct_b, step_counts) -> dict  # buckets easy<=1, medium<=3, hard (same as epoch_end_accuracy)
def noninferiority(delta_ci_lo: float, margin_pp: float = 1.0) -> bool
```
   Primary comparisons: Full ARS − Baseline at 0-shot and 8-shot, Holm over the two. Mechanism chain: Fixed−Baseline, Router−Fixed, Full ARS−Router. Always report discordant counts `(b, c)`, K distribution, layer frequencies, route entropy, abstain rate per shot, and `identical_items_fraction(cond4, cond5)` (if 1.0 the report says so explicitly instead of showing two columns).
6. **III-4 (condition equivalence) test lives here:** Router+SteerNet with a degenerate router (always `{L13}`) reproduces Fixed SteerNet with `S_fixed*={L13}` exactly, **with the cache disabled** (forced independent generation).
7. **Risk handling:** `batch_invariance_check` on 32 prompts (bs=1 vs bs=8 identical-output fraction) ⇒ advisory `W_BATCH_INVARIANCE` if <0.95; this never blocks.
8. Output: `experiment_record.json` (per item × condition: route, K, gate_mode, strict/lenient prediction, correct, guard flags, text hash) and `results_tables.json`. Every table row carries a `gate_mode` column.

**Conflict isolation:** `eval_gsm8k_benchmark` and `generate_response` remain (legacy single-sequence path) and are not edited except for an added `batched=False` default. `AnswerStoppingCriteria` untouched. SVAMP/ARC/MATH500 loaders in `multi_dataset.py` (dev) stay as they are; only the GSM8K adapter is required for this task.
**Verification:** `pytest tests/test_benchmark_runner.py -q`
- `stats`: McNemar exact against hand-computed values (`b=10,c=2 ⇒ p=0.03857`); Holm on `[0.01,0.04]` ⇒ `[0.02,0.04]`; paired bootstrap CI contains the empirical delta.
- Cache: same key ⇒ one generation (call counter), K=0 rows reuse the baseline entry; cache survives restart.
- III-4 equivalence (above).
- Tiny end-to-end: 5 conditions × 12 questions × shots {0,2} completes; conditions 1 and `K=0` rows of 3 are byte-identical.
- Guard revert: a tiny cap forces a revert for condition 4 only; condition 5 untouched; log row present.
- **Kaggle acceptance:** 0-shot, 5 conditions, n=1319 finishes in one session using the cache; III-1..4 pass at start.

---

### TASK M10 — Reporting, Orchestrator Notebook, Docs Sync

**Target files:** `notebooks/kaggle_ars_runner.ipynb` (✎; old copy preserved as `notebooks/legacy_kaggle_ars_runner.ipynb`), `scripts/run_ars_pipeline.py` (+), `scripts/make_report.py` (+), `src/visualization/` and `src/experiments/` (✎ `[verify]` dev modules), `scripts/run_experiment.py` (✎), `docs/training.md`, `docs/methodology.md`, `docs/research-background.md`, `README.md` (✎), `tests/test_report.py` (+), `tests/test_no_stale_calls.py` (+).
**Prerequisites:** M9 (record schema), M8 (contract), M0 (`Stage`).

**Core changes & logic**
1. **`scripts/run_ars_pipeline.py --stage {S0,S1,S0b,S2,S3,S4,S5,bench,all}`:** thin orchestrator over the functions from M1–M9 using canonical `Stage` names; each stage in `manifest.stage(...)`; **no stage is skipped on diagnostics**; DECISION-1/2 outputs are *printed* with the recommended branch and the pipeline waits for `--branch {A,B}` / `--router_mode {full,fixed_abstain}` arguments (explicit flags, no asserts). S4 (joint low-LR) is off unless `--run_s4`.
2. **Notebook (cell outline, replaces old Phase 1.5/2A–2D cells):** (1) env + `nvidia-smi`; (2) clone/update repo; (3) load model, contracts (`run_contract_checks`); (4) **S0** audit → prints audit table + `decision_1` ⟵ *human stops here*; (5) S1′ (with branch A/B switch variable); (6) **S0b** rebuild + DECISION 2 printout; (7) S2 router distill; (8) S3 gate; (9) S5 calibration → contract; (10) benchmark 0/8-shot × 5 conditions; (11) report; (12) save artifacts. Replace every `set_routing_override("all")` (F4). Notebook JSON validity is checked in tests.
3. **`scripts/make_report.py --record experiment_record.json --out_dir report/`:** tables (strict/lenient accuracy, Δ vs baseline + 95 % CI, discordant `(b,c)`, McNemar p, Holm-adjusted p, `gate_mode` column, per-shot route entropy, abstain rate, K distribution, layer frequencies, step-count strata), the audit table from S0/S0b, success-tier classification (T0–T3) computed from pre-registered criteria, LaTeX export (`report/tables/*.tex`) and a `report/REPORT.md`. **One command reproduces all tables from `experiment_record.json`.** A `passthrough` gate row is labeled and never cited as gate evidence.
4. **Docs sync (preserve existing docstrings/math; edit only what is wrong):** `docs/methodology.md` — correct gate init (α_init=σ(2)≈0.88, logit +2; remove "−6.0"), replace RLOO section with catalog distillation, add `question_span_mask`, dominance hinge, and `gate_input` dimension `2d+2+16`; `docs/training.md` — canonical S0–S5 curriculum + failure policy; `docs/research-background.md` — Stage-0 audit paragraph; `README.md` — pipeline overview/commands. Fix `scripts/run_experiment.py` Stage-B branch to use `subset_matrix`/contract instead of `"all"`.

**Conflict isolation:** do not delete the legacy notebook or legacy `--phase` options (`scripts/run_experiment.py` keeps `phase1|bootstrap|all|eval`; `all` prints a deprecation note pointing to `run_ars_pipeline.py`). No model/training logic changes.
**Verification:**
- `pytest tests/test_report.py -q` (synthetic `experiment_record.json` ⇒ tables, LaTeX files, tier classification, identical-items statement).
- `pytest tests/test_no_stale_calls.py -q` (greps repo: no `set_routing_override("all"`, no `−6.0` gate-bias claim in docs).
- `python -c "import nbformat; nbformat.validate(nbformat.read('notebooks/kaggle_ars_runner.ipynb', as_version=4))"`
- `python scripts/make_report.py --help` and `python scripts/run_ars_pipeline.py --help` exit 0.

---

## 4. Definition of Done (applies to every task)

1. New tests added and `pytest tests -q -x` green (including pre-existing `tests/test_modules.py`).
2. Only files in the task's target list were touched (§2.4).
3. Any new diagnostic writes `manifest.warn/diagnostic`, never raises (except §2.6 FATAL set).
4. No docstring or math comment removed (project guardrail 6).
5. All new artifacts are written under `checkpoints/` or `report/` (gitignored or explicitly committed).
6. The task's manifest stage block records: git commit, config hash, `steer_scale`, `gate_mode` (where applicable).

---

## 5. Open Decisions for the Plan Owner (architectural calls I made where the plan was silent)

| ID | Question | My default (implemented in the spec) |
|---|---|---|
| D1 | "teacher-forced **answer-weighted** NLL" is ambiguous. | Mean per-target-token NLL (`per_example_answer_nll`), plus full per-token NLL stored so any weighting (e.g., calc ×1.75) can be applied post hoc. |
| D2 | Plan gap for `0.005 ≤ H < 0.02`. | Proceed with M5 (distillation is <10 min), disable the optional RLOO polish, flag `ROUTING_HEADROOM_INTERMEDIATE`. |
| D3 | Decision 1 "otherwise → proceed" is undefined. | Branch B runs M4 with layer-dropout subsets on gold chains (plus on-policy if M3 data exist); table is always rebuilt (S0b) afterwards. |
| D4 | Primary extractor (strict vs lenient). | Lenient (= existing `evaluate_response`) for continuity with earlier numbers; strict reported in every table as a sensitivity analysis. |
| D5 | Shot-stratified S0 sampling (300/150/150). | **Nested** design: 300 questions at 0-shot; 150 of them re-rendered at 4- and 8-shot (600 rows total), enabling the shot-consistency loss. |
| D6 | S0 rows from the train split are in-sample for SteerNets (F9). | Add 120 held-out 0-shot rows; Decision-1 thresholds use held-out/cross-fitted values when they diverge from in-sample by >0.01 nats. |
| D7 | Plan lists "non-finite loss" as fatal; the existing loop skips such micro-batches (normal under fp16 `GradScaler`). | Non-finite *parameters* ⇒ fatal; non-finite *loss* ⇒ fatal only after 20 consecutive occurrences. |
| D8 | Meaning of θ_abs. | `chosen = argmax over all 26 classes`; if `chosen ≠ ∅` and `p(∅) ≥ θ_abs` ⇒ abstain; `θ_abs > 1` disables (condition 3 = disabled). |
| D9 | Direction/detach of the shot-consistency KL. | `KL(p_0.detach() ‖ p_k)` — the 0-shot routing is the teacher. |
| D10 | Per-item guard "revert to K=0". | Wrapper records trips; the **runner** regenerates tripped items with `override="none"` (exact K=0 output; an in-hook revert could not undo earlier layers' steering). |
| D11 | Router-feature caching for table-only S2. | Store question-span pooled states + scalars in the table (F8), so S2 needs no LLM forwards. |
| D12 | `s_ref` / certified scale. | `α_eff = α_gate · cfg.steer_scale`; default 1.0 pending A0 (§1.3). |

---

## 6. Appendix — Config Delta Summary (append-only blocks)

| Block | New `RSCConfig` fields |
|---|---|
| M0 | `failure_policy`, `manifest_path`, `max_consecutive_nonfinite_loss`, `steer_scale`, (+ F5 missing legacy fields) |
| M6a | `k_scale_mode`, `steernet_k_conditioning`, `gate_task_gradient` |
| M1 | `catalog_layers`, `catalog_max_k`, `cf_table_dir`, `cf_shard_size`, `cf_rows_per_forward`, `cf_audit_plan`, `cf_heldout_n` |
| M2 | `audit_g_fixed_min`, `audit_h_low`, `audit_h_high`, `audit_g_zero_eps`, `audit_n_boot`, `audit_inflation_margin` |
| M3 | `onpolicy_*` (9 fields) |
| M4 | `s1_*` (10 fields) |
| M5 | `router_policy`, `router_c_size`, `router_eps_tie`, `router_tau`, `router_lambda_bal`, `router_lambda_cons`, `router_distill_*`, `router_catalog_save_path`, `router_loss_mode` |
| M6b | `abstain_theta`, `guards_enabled`, `guard_rel_norm_cap`, `guard_k_max` |
| M7 | `gate_init_logit`, `gate_loss_mode`, `lambda_dom`, `lambda_tok`, `lambda_l1_gate_v5`, `gate_tau_u`, `gate_dom_margin`, `gate_epochs`, `gate_lr`, `gate_label_row_frac`, `gate_v5_save_path`, `gate_sel_n` |
| M8 | `calib_*`, `contract_path` |
| M9 | `bench_*` |

*End of specification.*


ده  كان الفايل اللي خرجه الايجنت اتنين
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-23T16:27:13+03:00.
</ADDITIONAL_METADATA>