<USER_REQUEST>
# MASTER_REASONING_PLAN.md

**Adaptive Residual Steering (ARS) on Phi-2: Root-Cause Analysis, Redesign, and Recovery Plan**
Run analyzed: `exp_20261023_081635_ars_phi2_kaggle` · Plan date: 2026-10-23

---

## 0. Executive Summary, Evidence Base, and Corrections to the Brief

### 0.1 Bottom line

The failure is not mainly the kill-switch, the K=0 collapse, or the untrained gate. Those are real, but they sit downstream of a more basic problem: **we have not yet shown that any steering subset improves greedy-generation accuracy.** Evidence from the run:

| Observation | Value |
|---|---|
| Phase 1 teacher-forced NLL improvement (L13) | train/val −15.1% / −14.7% |
| Phase 1 held-out generation confirmation (L13, α=0.30) | **Δ −6.2pp** (calibration +8.3pp, combined +0.0pp) |
| Counterfactual utility spread in router bootstrap | K0 +0.450, K1 +0.467, K2 +0.463 (a **~0.015-nat** spread) |
| Fixed SteerNet vs Baseline, 2/4/8-shot | **+0.0pp** at every shot |
| Router+SteerNet vs Baseline, 0-shot | +1pp on n=100, with a noise floor of about ±10pp |
| Effective steering magnitude at L13 | ~0.9% of the hidden-state norm (cap 2%) |

Steering lowers the NLL of gold-style chains of thought. In the 0-shot qualitative traces it mostly changes format (it adopts the "Step N:" grammar). It does not yet change which problems are solved. The router and gate cannot create signal that SteerNet does not provide, so the recovery plan inserts a cheap **signal audit** (Stage 0) with explicit go/no-go criteria before any more router or gate engineering. It then fixes the training target (on-policy correct traces), the router (supervised distillation from an exhaustive counterfactual table), the gate (task-loss gradient, identity init, dominance constraint), and the evaluation (full test set, paired inference, route-identical output reuse).

### 0.2 Evidence base and caveats

- Code analyzed: the attached `src/` snapshot (wrapper, steernet, gate, router, rloo, trainer, losses, contrastive, dataset, verifier) plus the full Kaggle notebook log.
- **The notebook imports modules I could not read:** `make_engine_v40_kaggle_config`, `multi_benchmark_runner`, `multi_dataset`, `audit_router_policy_health`, `experiments`, `visualization`, and the v4 router with `get_layer_synergy_matrix`. The attached `config.py`/`router.py` predate them. Everything below that touches the v4 code is inferred from log output and flagged **[verify]**. The implementation agent must confirm those points against branch `development` @ `1b7447f`.

### 0.3 Corrections to the brief

1. **The gate was not near zero at init.** The brief cites bias −6.0 (α≈0.0025), which is the methodology doc. The code I can see uses `gate_init_alpha=0.80`, and the preflight log shows L13 gate `mean_alpha=0.945` with 88.5% high-saturation. The untrained gate was near-saturated, not silent. **[verify]** which init the dev branch uses.
2. **"Full ARS lost because of the untrained gate" is plausible but unproven.** The 0-shot gap is 3 questions, CI [−7, 0] (touches zero). Treat it as a hypothesis the new gate evaluation must test.
3. **Single-layer dominance is partly legitimate.** L13 had the best individual val NLL (0.8985 vs 0.91–0.93 for the others). A router that picks L13 when layers are near-interchangeable is rational. Diversity must be earned from data, not forced.
4. **"Fixed SteerNet" ran at an untrained operating point.** It activated 5 layers at once (labelled K=3) with a constant gate, while SteerNets were trained only at K∈{1,2}. That is a strawman baseline and a potential source of harm (§I.6).
5. **Phase naming is inconsistent** across `docs/training.md`, the notebook headings, and `train_rsc`. The plan below defines a single canonical naming (§II.5).

---

# Part I — Root Cause Deconstruction

## I.1 Meta-cause: train/eval objective and operating-point mismatch

Four mismatches stack:

1. **Objective:** SteerNet is trained on human GSM8K chains (teacher-forced NLL). Phi-2's greedy decode distribution differs from those chains. Lowering NLL on gold text mostly teaches style/format. It does not teach the model to make fewer arithmetic or logic errors on its own samples.
2. **Selection noise:** Phase 1 checkpoint and layer selection used 12-sample generation screens (binomial SE ≈ 14pp) and 24/32-sample calibration/confirmation. Selection on this is noise-fitting, which explains why the "winner" (L13, +8.3pp calibration) flipped to −6.2pp on disjoint confirmation.
3. **Operating point:** SteerNet certified at α=0.30 and K≤2. The benchmark's "Gate=1.0" conditions and the 5-layer Fixed condition do not match this **[verify the semantics of "gate pinned at 1.0" vs certified scale 0.30]**.
4. **Context:** Training is mostly 0-shot with 30% shielded few-shot. Evaluation at 2/4/8 shots places 300–950 tokens of exemplars before the question, shifting every pooled feature the router sees.

## I.2 Why routing collapses to L13

1. **No per-prompt signal to learn from.** Counterfactual reward spreads are ~0.015 nats across K=0/1/2, and layer-to-layer differences are smaller. RLOO with N=2 estimates `A(S_a)=R(S_a)−R(S_b)` from one pair per example, so the gradient is dominated by noise.
2. **Router inputs are low-information.** The router sees mean-pooled hidden state plus RMS/coverage scalars from one early layer. These characterize prompt style and length far more than "which layer's correction helps this problem."
3. **Possible variance injected by the bootstrap itself.** The old `train_router_bootstrap` path runs counterfactual forwards with `model.train()` under `no_grad`. If SteerNet dropout (0.05) is active in those forwards, NLL differences between subsets include dropout noise comparable to the 0.015 signal. **[verify in v4]**
4. **Joint NLL worsened across bootstrap epochs** (0.831 → 0.867 → 0.930) while "SteerNet compatibility correction" was updating SteerNets at lr 5e-5. That is evidence of SteerNet drift while the router trains. Coupling the two defeats attribution.
5. **L13 carries 96.3% of effective energy** (L18: 3.7%). Even when K=2 is selected, the second layer's effective norm is ~0.003 relative. Layers 15/20/22 contributed nothing in deployment.

## I.3 Why K→0 on 2/4/8 shots

- **Plausible mechanism (OOD features):** `router_demo_features=True` plus long prefixes puts pooled-state statistics out of the training distribution, and `k_head` falls back to its bias, where the tariff and p(K0) prior (already 0.23–0.32 on 0-shot) win.
- **Plausible mechanism (rational abstention):** Fixed SteerNet at 2/4/8 shots is identical to baseline in accuracy. If steering does nothing in that regime, K=0 is the cost-optimal action under any positive tariff. The route entropy of 0.0 bits is then the router correctly reporting that it has no gain to offer.
- **These hypotheses are separable** with the counterfactual table in §II.0. If per-example NLL gain at 4/8 shots is ≈0 for every subset, the collapse is rational and the multi-shot claim should be non-inferiority. If gain exists but the router abstains, it is a feature/calibration failure.
- **Arbiter masking:** Full ARS and Raw ARS are identical in every cell. The "deployment arbiter" currently does not differentiate the conditions, so the benchmark cannot tell them apart.

## I.4 Why the gate cannot currently show efficacy

1. **The gate receives no task-loss gradient.** In `wrapper.py` the hook computes `route_st * alpha.detach() * delta ...`, so the gate learns only from auxiliary losses. Those auxiliaries are hand-written token-class targets (math tokens → 0.85; prompt/format → 0.05; discourse/other → 0.40), L1, anti-saturation, diversity and budget terms. None measures whether steering at that token helps. `lambda_pg=0` disables the only outcome-based term.
2. **No dominance guarantee.** Nothing constrains gated NLL ≤ ungated NLL.
3. **K double-compensation.** The hook divides by √K, and SteerNet is also K-conditioned via `k_proj`. The two interact and are never ablated.
4. **The gate was never trained in this run** (Phase 2 skipped), so the Full ARS 0-shot number measures a randomly initialized module.

## I.5 Loss conflict analysis (router and gate)

| Term | Intended effect | Magnitude vs signal | Conflict |
|---|---|---|---|
| Counterfactual NLL reward | Pick helpful subsets | Spread ~0.015 nats | Weakest term by 1–2 orders of magnitude |
| Tariff (`lambda_cost`, relative K cost 0.03, ramp) | Prefer small K | Comparable to the reward spread | Directly pushes K→0 when gain is small; unmodelled at out-of-distribution shot counts |
| K entropy (0.10) / subset entropy (0.05) | Prevent collapse | Entropy up to ln(26)≈3.3 nats | Dominates the reward; keeps the policy diffuse in training but argmax at deployment is still a constant |
| Layer MI + K MI (0.15 each) | Prompt-dependent routes | Needs prompt-dependent signal that does not exist | Pushes diversity without data support |
| K-preference distillation (0.25) | Match exhaustive K=0..4 rewards | Targets the same flat reward | Redundant with RLOO; two estimators of one noisy quantity |
| Synergy regression (0.25) + prior (0.01) | Pair interactions | Pair estimates ±0.1–0.3 on 63 steps | Over-fits noise (top pairs flip sign between epochs: L15+L20 +0.29 → −0.12) |
| Gate: L1, budget, antisat, diversity, contrastive | Sparsity / selectivity | Hand-tuned heuristics | Mutually antagonistic (sparsity vs. anti-saturation vs. "math tokens at 0.85"), none tied to usefulness |
| Phase-1B / compat SteerNet updates during router training | Cooperative residuals | Moves the very quantity the router measures | Non-stationary reward |

Conclusion: there are about nine loss terms against one weak signal. The redesign removes most of them (§II.2, §II.4).

## I.6 Pipeline fragility

- `skip_adaptive_phases_when_router_fails=True` plus an entropy threshold (0.25) on a policy that is correctly mostly-L13 turned a diagnostic into a pipeline abort. Phase 2 never ran.
- A **fatal** check was applied to a metric (layer entropy) that is not a correctness property.
- Compute was spent on low-information procedures: ~85 min of 12-sample generation screens, a 43-min acceptance audit with n=24–32 slices, and ~42 min of router bootstrap, while all final claims rest on n=100.

## I.7 Statistical power

For a paired comparison with discordant-pair rate *d* (say 12%), SE(Δ) ≈ √(d/n):

| n | SE(Δ) | Minimum detectable Δ (~80% power, α=0.05) |
|---|---|---|
| 100 | ≈3.5pp | ≈9.7pp |
| 500 | ≈1.5pp | ≈4.3pp |
| 1319 (full GSM8K test) | ≈0.95pp | ≈2.7pp |

At n=100, a +4 to +8pp true effect is mostly undetectable. All headline claims must use the full test set.

---

# Part II — Architectural and Algorithmic Solutions

## II.0 Stage 0: Counterfactual Signal Audit (new; cheap; decides everything)

**Purpose:** build once a *counterfactual table* that every later stage reuses, and decide go/no-go on routing and gating *before* training them.

- **Catalog:** the empty set plus all subsets of the 5 admitted layers `[13, 15, 18, 20, 22]` with K≤3 → 1 + 5 + 10 + 10 = **26 subsets**.
- **Data:** 600 held-out-from-training examples drawn from the training split (never from the benchmark or the 240 eval rows), stratified by shot count (0: 300, 4: 150, 8: 150).
- **Per (example, subset):** teacher-forced answer-weighted NLL, plus per-token NLL for the empty set, each single layer, and the best fixed subset.
- **Cost:** ~26×600 forwards ≈ 45–70 min on one T4 at the observed ~6 forwards/s.
- **SteerNet runs in `eval()` mode** (no dropout) for all counterfactuals.

**Computed quantities**

1. `G_fixed*` = best single fixed subset's mean NLL gain over baseline.
2. `G_oracle` = mean gain if each example picks its best subset (an upper bound for routing).
3. **Routing headroom** `H = G_oracle − G_fixed*`, with a bootstrap CI.
4. Per-shot repeat of 1–3 (separates "rational K=0" from "feature failure").

**Go/no-go (advisory outputs, written to the run manifest; the *plan owner* reads them, the code does not abort):**

| Result | Interpretation | Action |
|---|---|---|
| `G_fixed*` < 0.01 nats at 0-shot | Steering is not helping even on NLL | Switch training target (II.1) before anything else |
| `H` < 0.005 nats | Routing has no headroom | Ship router as "fixed-subset selection with abstain"; stop investing in diversity |
| `H` ≥ 0.02 nats at some shot | Routing can help | Proceed with II.2 |
| 4/8-shot `G_fixed*` ≈ 0 | K=0 is rational there | Multi-shot claim = non-inferiority (II.3) |

## II.1 Training target: on-policy correct traces (fixes the NLL→accuracy gap)

**Phase S1′ (SteerNet training) uses model-generated, verified-correct chains instead of (or mixed with) gold GSM8K chains.**

1. For ~1500 training questions, sample 2–4 chains from frozen Phi-2 (T=0.7, batched generation, mixed 0/4-shot prompts, `max_new_tokens` ≤ 320). Verify with the existing numeric extractor.
2. Keep correct chains. For questions with both correct and incorrect samples, retain the pair.
3. **Loss (token-level):** (a) NLL on correct chains (on-policy, low-perplexity targets, so steering nudges rather than rewrites style); (b) optional unlikelihood or contrastive margin on the first divergent step of an incorrect sibling chain, weighted small (λ≈0.1).
4. Keep a 30–50% mix of gold chains for coverage of questions the base model never solves.
5. Keep `calculation_weighted` token weights (calc_token_weight 1.75), but compute the weights on the on-policy chains.
6. **Train with layer dropout over subsets** (sample K∈{1,2,3} subsets per step from the 5-layer pool, K sampled uniformly) instead of "K=1 rotation, then 1B co-adaptation." This trains the SteerNets cooperatively at every operating point the router may select, and removes the need for Phase 1B.
7. **Checkpoint selection:** teacher-forced NLL on 120 selection rows plus a *single* batched generation check (n=240, all 5 layers' best subset vs baseline, paired) at the end. Remove per-epoch 12-sample generation screens.

Budget: sampling ≈ 1 h with batched HF generation (aggregate ~250–400 tok/s) **[verify]**; training ≈ the existing ~40 min.

## II.2 Router redesign: catalog classifier distilled from the counterfactual table

**Replace** Plackett-Luce + K-head + RLOO N=2 + MI + synergy regression + K-preference with a **catalog policy** over the 26 subsets.

**Policy.** `p(S | x) = softmax(f(x))` over the catalog (empty set included as class 0). Keep the existing attention-over-candidates encoder, but the head outputs 26 logits (or per-layer utilities plus a subset scorer; the implementer may keep `JointLayerRouter` and add `catalog_head`).

**Targets (ε-tie soft labels).** For example *n* with rewards `R[n,S] = −NLL[n,S] − c·|S|` (c = 0.001, a tie-breaker only, no ramp, no relative tariff):

```
Let R*_n = max_S R[n,S]
Near-optimal set  E_n = { S : R[n,S] ≥ R*_n − ε }       (ε = 0.01 nats, ≈ measurement noise)
q_n(S) ∝ exp(R[n,S]/τ) restricted to E_n                 (τ = 0.01)
Loss   L_router = KL(q_n ‖ p(·|x_n))  +  λ_bal · L_balance
```

Why this fixes collapse honestly:

- **Near-ties split mass.** When layers are interchangeable the target is spread over them, so the router does not receive a spurious "always L13" label.
- **When one subset is clearly best, the router picks it,** and that is the correct behaviour.
- **L_balance** (λ_bal = 0.02, load-balancing à la Switch Transformer) is applied *only over examples whose E_n contains ≥2 subsets*, so it never fights real signal.

**Other rules**

- **SteerNets frozen during router training.** This removes the non-stationarity seen in the 1.5 epochs (joint NLL drifting 0.83→0.93).
- **No MI, synergy regression, K-preference, K-entropy or subset-entropy terms.** Pair synergy remains a *diagnostic* computed from the table (`NLL[S∪{j}] − NLL[S] − ...`), not a loss.
- **Optional RLOO polish** (≤1 epoch, lr 1e-4) only if Stage 0 shows `H ≥ 0.02` and distillation accuracy plateaus; it must use `eval()` SteerNets and N≥4 samples per example with a leave-one-out baseline.
- **Router inputs:** pooled state from the question span only (II.3) plus per-layer depth embedding; no shot-count feature.
- **Calibrated abstain:** class 0 (empty set) is chosen by the learned policy. In deployment, abstain is *additionally* gated by a threshold θ_abs chosen on the selection split (II.5).
- **Router-evaluation metrics (advisory, never fatal):** top-1 agreement with table argmax; mean regret (R* − R[chosen]); regret vs best fixed subset; route entropy; per-shot abstain rate.

## II.3 Multi-shot conditioning

1. **Question-only router context.** Pool over the question span only (tokens of the final `Question: … Answer:` segment), using the existing prompt-shield mask extended with a `question_span_mask`. Exemplars are excluded from pooling, RMS, and coverage statistics.
2. **Shot-invariant normalization.** Standardize the pooled vector with running statistics computed per shot-count bucket during training (LayerNorm on the pooled feature plus per-bucket mean/var buffers; at test time use the bucket matching the prompt's shot count, falling back to pooled stats).
3. **Shot-mix training for the router.** The distillation set contains 0/2/4/8-shot formatted prompts (from the Stage-0 table, resampled with equal bucket weights). `router_demo_features` is **removed** (there is no longer a feature to saturate).
4. **Consistency loss.** For the same question rendered at two shot counts, add `λ_cons · KL(p_0 ‖ p_k)` (λ_cons = 0.1) so routing depends on the problem, not the context length.
5. **SteerNet shielding stays:** no steering on exemplar tokens.
6. **Multi-shot objective reframed after Stage 0.** If 4/8-shot headroom is ≈0, the claim at those shots is **non-inferiority** (Full ARS − Baseline ≥ −1.0pp, paired CI lower bound), with K=0 a legitimate outcome.
7. **Collapse definition (advisory warning):** route entropy = 0 *and* Stage-0 headroom at that shot > 0.01 nats.

## II.4 Token gate: calibration, synergy, and a dominance guarantee

**Gate trained after the router, with SteerNets and router frozen.**

1. **Task-loss gradient into the gate.** In the hook use `alpha` (not `alpha.detach()`) in `h + route · alpha · delta · mask / k_scale`, while keeping `delta` detached *as the gate's input feature* (as now). This gives the gate a direct "does this token's steering lower NLL" gradient. SteerNet gradients remain blocked because SteerNets are frozen in this stage.
2. **Identity initialization.** `fc2.bias` initialised so `α_init ≈ 0.88` (logit +2) with zero weights, so the gate starts as nearly pass-through and learns *deviations*. This also removes the ambiguity between the doc's −6.0 and the code's 0.80.
3. **Counterfactual token labels** (from Stage 0): `u_t = NLL_base,t − NLL_steered,t` per target token. The gate's auxiliary supervision is a weighted BCE toward `1[u_t > τ_u]` (τ_u = 0.005). It replaces the hand-specified 0.85/0.05/0.40 token-class targets, the contrastive loss and the anti-saturation loss. Keep optional hand heuristics as *advisory diagnostics* only (does the gate fire more on calculation tokens?).
4. **Dominance hinge.** `L_dom = relu(NLL_gated − NLL_ungated.detach() + m)` with m=0, computed on the same batch with the same route. This makes "gated ≥ ungated" a trained constraint.
5. **Total gate loss:** `NLL_gated + λ_dom·L_dom + λ_tok·BCE_tok + λ_L1·mean(α)` with λ_dom = 1.0, λ_tok = 0.3, λ_L1 = 0.005. Remove `diversity` (floor 0.05) and `antisat`; keep a quadratic *budget* only as an optional regularizer, off by default.
6. **K-scaling cleanup.** Run a one-time ablation of {√K division on/off} × {K-conditioning on/off}. Keep the best on the selection split and remove the other mechanism.
7. **Adoption rule (the honest form of "Full ARS ≥ SteerNet").** The gate is adopted only if, on the 120-row selection split with the router frozen: (a) paired NLL is lower than ungated, and (b) generation accuracy is not worse by more than one question (paired). Otherwise the gate checkpoint is recorded as `gate_not_adopted` and the `ars` condition is evaluated with `gate_mode=passthrough` and is *labelled as such in every table*. A passthrough result must never be presented as evidence of gate efficacy.
8. **Generalization caveat:** the guarantee holds on the selection split by construction. On the test set it is a hypothesis, tested by the paired comparison in Part III.

## II.5 Curriculum simplification and failure policy

**Canonical stages (replace Phase 1, 1B, 1.5, 2A–2D):**

| Stage | Name | Trains | Frozen | Replaces |
|---|---|---|---|---|
| S0 | Counterfactual Signal Audit | nothing | everything | Phase-1 acceptance audit, generation screens |
| S1 | SteerNet (on-policy, layer-dropout subsets) | SteerNets | router, gate (pass-through α=s) | Phase 1, 1B, 2A |
| S0b | Rebuild table with the S1 SteerNets (if S1 changed them materially) | nothing | all | (new) |
| S2 | Router distillation (+ optional polish) | router | SteerNets, gate | Phase 1.5 |
| S3 | Gate (task loss + counterfactual labels + dominance) | gate | SteerNets, router | Phase 2B, 2D |
| S4 | *Optional* joint low-LR pass (all three, lr ×0.05, ≤1 epoch, early stop on paired selection NLL) | all | none | Phase 2C |
| S5 | Deployment calibration (abstain threshold, envelope check, gate adoption) | scalars only | all | the "arbiter" |

S0 runs once before S1 to size the opportunity on the existing Phase-1 checkpoint. S0b reruns it after S1 (the SteerNets changed, so the table must change). S4 is skipped unless S2/S3 leave measurable NLL on the table.

**Failure policy (the core reliability change).**

- **Fatal (abort):** non-finite loss or parameters, missing or corrupt checkpoint or manifest, shape or hook-contract violation, baseline-identity test failure (Part III).
- **Everything else is advisory:** entropy thresholds, K support, active-layer count, synergy signs, effective-layer count, preflight "reliability" checks. These write `WARN` entries into `run_manifest.json` (and into the final report), and the pipeline *continues*.
- Delete the behaviours of `skip_adaptive_phases_when_*`, `router_fail_fast_on_collapse` and `continue_on_diagnostic_failure=False` as fatal gates. Replace with `failure_policy: "warn"` (default) and a `strict` mode available only for CI.
- **Always finish training, then decide deployment** in S5. A failed diagnostic changes which condition is *recommended* for deployment, never whether later stages run.
- **Compute reallocation:** remove ~2 h of low-power generation screening and rerun the same budget as the S0 table, S1′ data generation, and full-test-set evaluation.

---

# Part III — Explicit Definition of the Five Experimental Conditions

**Global invariants (all conditions):** same prompts, same greedy decoding (`do_sample=False`, same `max_new_tokens`, same repetition settings), same extractor (report strict and lenient), same per-shot seed, same item order. Shot sets {0, 2, 4, 8}; **primary endpoints are 0-shot and 8-shot on all 1319 GSM8K test questions**; 2/4-shot are secondary and may use a fixed random 500-question subset. Outputs are cached by key `(prompt_hash, route_tuple, gate_mode, steer_scale)`; when two conditions produce the same key, generation runs once and the output is reused, which guarantees identical results and saves ~40–60% of compute (K=0 routes reuse the Baseline output).

| # | Condition | Routing | Gate | Steer scale | Abstain / fallback | What it answers |
|---|---|---|---|---|---|---|
| 1 | **Baseline** | `override="none"` (K=0 for all items) | n/a | n/a | n/a | Frozen Phi-2 reference |
| 2 | **Fixed SteerNet** | One static subset `S_fixed*`, chosen on the selection split from the 26-subset catalog by (a) paired NLL, then (b) accuracy; ties → smaller K | Bypassed: constant scale `s` | `s` = the certified training scale (0.30 unless S1′ changes it), declared once in the manifest | None | Best fixed intervention (the bar the router must beat) |
| 3 | **Router+SteerNet** | Learned catalog router, argmax, abstain = class 0 only (no θ_abs) | Bypassed: constant scale `s` | same `s` | None | Value of per-prompt routing |
| 4 | **Full ARS** | Learned router with calibrated abstain θ_abs (S5) | Learned token gate (`gate_mode=learned`, else labelled `passthrough`) | learned α | Production guards only: relative-norm cap = training envelope, K≤3, NaN/Inf guard, per-item guard that reverts that item to the K=0 output if any numeric guard trips (logged) | The shippable system |
| 5 | **Raw ARS** | Learned router, argmax, no θ_abs | Learned token gate, no cap beyond the training envelope | learned α | **None.** No guards, no reverts; full routing and gate trace recorded | Scientific lens: what the learned policy does unmodified |

**Critical definitional changes vs. the current run**

- **Fixed SteerNet is no longer "all 5 layers."** It is the best-on-selection static subset (expected: L13 or a pair). This also removes the K=5 out-of-distribution evaluation.
- **No hierarchical arbiter** (`adaptive → fixed_steernet → baseline`). Abstention is a router output; the only deployment-time swaps are the numeric guards in condition 4, and they are logged per item. Full ARS and Raw ARS must therefore be able to differ; if their outputs are identical on every item, the report says so explicitly rather than showing two columns.
- **`gate_mode` is a column in every table.** A row with `passthrough` cannot be cited as gate evidence.
- **Steer-scale parity:** conditions 2 and 3 use the same constant scale `s` that the SteerNets were trained/certified at. **[verify]** that "Gate=1.0" in the current code does not mean 3.3× the certified magnitude.

**Required automated tests (run every time, fatal)**

1. **Baseline identity:** for 16 prompts, logits under `override="none"` equal logits from the unwrapped base model (max abs diff ≤ 1e-3 in fp16).
2. **Hook no-op:** with all selected = False, hidden states are unchanged.
3. **Route plumbing:** for a forced `matrix` route, the recorded route tuple equals the forced route for prefill *and* cached decode steps.
4. **Condition equivalence:** Router+SteerNet with a degenerate router (always {L13}) reproduces Fixed SteerNet with S_fixed* = {L13} exactly.

**Inference plan (pre-registered)**

- **Primary:** Full ARS − Baseline at 0-shot and at 8-shot, n=1319, paired bootstrap (10k resamples) and exact McNemar; Holm correction across the two primary comparisons.
- **Mechanism chain (secondary):** Fixed − Baseline; Router − Fixed; Full ARS − Router. Success pattern: Full ARS ≥ Router ≥ Fixed > Baseline, each step with its own CI.
- **Report** discordant-pair counts (b, c), not only accuracy; stratify by problem step count; report routes (K distribution, layer frequencies) per shot.
- **Seeds:** at least one more training seed for S1–S3 (two total) if the primary effect is positive but CI-marginal; otherwise report as single-seed.

**Success tiers** (so "success" is decided before the data arrives)

| Tier | Criterion |
|---|---|
| T3 (target) | Full ARS − Baseline ≥ +4pp at 0-shot (CI excludes 0) and ≥ −1pp (non-inferior) at 8-shot; chain Fixed ≤ Router ≤ Full ARS holds at point estimates |
| T2 | Positive, CI-excluding-0 gain (≥ +2pp) at 0-shot only, with non-inferiority elsewhere |
| T1 | Non-inferior everywhere; routing/gate mechanisms documented with Stage-0 evidence of why gains are small |
| T0 | Anything else → report as a negative result with the signal audit |

My honest prior: a ~1.4M-parameter residual adapter at ≤2% relative norm gaining +4–8pp *and* beating a strong 8-shot baseline is a stretch. T2 is the realistic target and T3 requires S1′ (on-policy data) to move the Stage-0 gains materially. The plan is designed so that this is learned in about three GPU hours (S0 + S1′ data) rather than after another end-to-end run.

---

# Part IV — Execution Roadmap and Handoff Specification (for Agent 2)

## IV.1 Module breakdown

| ID | Module | Files (likely) | Deliverable | Acceptance test |
|---|---|---|---|---|
| **M0** | Failure-policy refactor | `config.py`, `trainer.py`, preflight/audit code **[verify names]** | `failure_policy="warn"`; fatal set limited to numeric/contract errors; all diagnostics write to `run_manifest.json` | Simulated router-collapse run completes all stages; manifest contains WARN entries |
| **M1** | Counterfactual table builder | new `src/analysis/cf_table.py` | 26-subset catalog enumerator; per-(example, subset) answer NLL + per-token NLL (empty, singles, best fixed); eval-mode SteerNets; shot-stratified sampler; resumable cache (npz) | Table reproduces the Phase-1 NLLs (L13 val 0.8985 ± 0.003); empty-set NLL equals the baseline NLL |
| **M2** | Signal audit report | `src/analysis/signal_audit.py` | `G_fixed*`, `G_oracle`, headroom `H` with bootstrap CIs, per-shot; go/no-go table in manifest and notebook | Runs on the existing Phase-1 checkpoint before any retraining |
| **M3** | On-policy data generator | new `src/data/onpolicy.py` | Batched sampler (left padding, hooks off), verifier filter, paired correct/incorrect store, mix with gold chains | ≥1000 verified-correct chains; extraction agreement ≥98% with verifier on a 100-chain manual-spot-check set |
| **M4** | SteerNet S1′ trainer | `trainer.py`, `losses.py`, `steernet.py` | Layer-dropout subset sampling K∈{1,2,3}; on-policy + gold loss mix; teacher-forced selection; single final generation check; K-scaling ablation switch | Beats Phase-1 L13 val NLL on the on-policy validation set; no per-epoch generation screens |
| **M5** | Router catalog head + distillation | `router.py`, new `src/routing/distill.py` | 26-class policy; ε-tie soft targets; balance loss; frozen SteerNets; question-only pooling; shot-mix + consistency loss; removal of RLOO/MI/synergy/K-pref losses (kept behind flags) | Mean regret ≤ 50% of best-fixed regret on held-out table rows; top-1 agreement reported; no run-time assertion on entropy |
| **M6** | Wrapper plumbing | `wrapper.py` | `question_span_mask` plumbing; non-detached `alpha` in hook (flagged); route/gate trace per item; per-item guards (cond. 4); constant-scale bypass mode `s`; route-tuple cache keys | Tests III-1..4 pass; trace shows per-item routes at all shots |
| **M7** | Gate stage | `gate.py`, `losses.py`, `contrastive.py` | Identity init; counterfactual token-label BCE; dominance hinge; adoption rule; `gate_mode` stamped in manifest | On the selection split, gated NLL ≤ ungated or `gate_not_adopted` recorded |
| **M8** | Deployment calibration (S5) | new `src/deploy/calibrate.py` | θ_abs search on selection split; envelope check; gate adoption; writes `deployment_contract.json` | Contract file fully determines conditions 4 and 5 |
| **M9** | Benchmark runner | `multi_benchmark_runner.py`, `benchmark.py`, `multi_dataset.py` **[verify]** | Five conditions per Part III; full 1319 test set; batched generation; output cache by route key; strict+lenient extraction; paired bootstrap and McNemar; discordant counts; Holm | 0-shot 5-condition run on n=1319 completes within one Kaggle session using the cache; conditions 2/3 equivalence test passes |
| **M10** | Reporting and notebook | `kaggle_ars_runner.ipynb`, `visualization`, `experiments` | Notebook trimmed to S0→S5 + benchmark; tables with `gate_mode` column; per-shot route entropy and abstain rates; LaTeX export with CIs | One command reproduces tables from `experiment_record.json` |

## IV.2 Order of execution and decision points

```
M0 ──► M6(partial: bypass mode, traces) ──► M1 ──► M2   ← DECISION 1 (signal audit on existing Phase-1 ckpt)
                                                  │
         ┌────────────────────────────────────────┤
         │ G_fixed* < 0.01 nats  → M3 → M4 (switch objective), then M1 again
         │ otherwise             → proceed
         ▼
        M4 ──► M1 (rebuild) ──► M2 ←── DECISION 2 (headroom H per shot)
                                   │
                    H ≥ 0.02 ──► M5 ──► M7 ──► M8 ──► M9 ──► M10
                    H < 0.005 ──► router = fixed-subset selector + abstain; M7 still runs; M9 as above
```

M0 and M6-partial are prerequisites and should land first (they unblock everything and fix correctness issues). M1+M2 are the single highest-value-per-GPU-hour items. Do **not** start M5/M7 until DECISION 1 is recorded.

## IV.3 Compute budget (Kaggle T4, indicative)

| Item | Time |
|---|---|
| S0 table (600 ex × 26 subsets) | 45–70 min |
| M3 on-policy sampling (~1500 questions × 2–4 samples) | ~1 h **[verify throughput]** |
| S1′ SteerNet training (4 epochs, as now) | ~40 min |
| S0b rebuild | 45–70 min |
| S2 router distillation (table-only, no LLM forwards) | <10 min |
| S3 gate (3 epochs) | ~30 min |
| S5 calibration | <15 min |
| Benchmark, 0-shot and 8-shot × 1319, with output cache | 2–4 h (batched) |

Roughly 6–9 h across two sessions versus ~8 h previously for a run that produced no usable evidence.

## IV.4 Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Steering genuinely cannot add accuracy at this capacity | Medium-high | Stage-0 audit decides early; T1 deliverable is still a rigorous negative/partial result |
| Batched generation incompatible with per-item routing/hook state | Medium | Left-padding path in `_valid_mask_for`, cached-decode routes keyed per batch row; run test III-3 under batch size >1 before M9 |
| On-policy data too small (Phi-2 solves ~55–60%) | Low-medium | Add gold chains; sample more per question; use all siblings |
| Table-derived token labels noisy | Medium | τ_u margin, label smoothing, report gate-vs-label AUC as diagnostic; dominance hinge remains the primary safety |
| Distribution shift between table (teacher-forced) and decode (autoregressive) | Medium | Final paired generation check in S1′/S5; report both NLL and accuracy chains |
| Hidden dev-branch behaviour differs from my reading | Medium | All **[verify]** items listed as the first task of the implementation agent |

## IV.5 First-day checklist for Agent 2

1. Confirm the **[verify]** items: gate init value; "Gate=1.0" vs certified scale 0.30; whether router-bootstrap counterfactual forwards ran with SteerNet dropout active; why Fixed SteerNet is labelled K=3 with five layers; how `audit_router_policy_health` is wired into `train_rsc` return paths.
2. Land M0 (failure policy) and the test suite (Part III, tests 1–4).
3. Implement M1/M2 against the *existing* Phase-1 checkpoint (`steer_phase1_adaptive_v11.pt`) and report `G_fixed*`, `G_oracle`, `H` per shot **before** touching the router or gate.
4. Return the audit table to the plan owner; the remaining modules proceed according to DECISION 1 and 2.

---

*End of plan.*

ده نتيجه اول ايجنت بتاع التنفيذ اي رايك
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-23T15:00:25+03:00.
</ADDITIONAL_METADATA>