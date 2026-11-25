# Training pipeline

The final study separates fresh fixed-adapter training from route-bank and router training. This keeps the expensive collection and fixed-adapter work in its own saved handoff. Notebook 01b can continue from the handoff without repeating Stage A.

## Stage A: fresh fixed adapter and verified rollouts

Run `notebooks/01a_train_fixed_and_bank.ipynb` with a CUDA GPU and internet access. Do not attach an old adapter, norm file, rollout shard, bank, or router. The notebook creates a unique run directory and pulls the configured `development` branch. It records the exact source commit in the handoff manifest.

### A1. Runtime and diagnostics

1. Select `RUNTIME_PLATFORM="auto"` or set Kaggle/Colab explicitly. Set `GPU_MODE="auto"`, `"single"`, or `"dual"`.
2. Check CUDA device visibility. Two GPUs allow the two rollout shards to run concurrently; one GPU collects them sequentially.
3. Run the repository unit and contract suite. The notebook records status and does not hide a failure.
4. Calibrate fresh p99 norms for the fixed route using 128 GSM8K training prompts in the full profile. No uploaded calibration is read.
5. Run batch-generation equivalence and a sampled rollout-quality probe as diagnostics. The probe is nonblocking; its result is recorded separately from training validity.

### A2. Rollout collection

The full profile schedules 5,000 unique GSM8K training questions across two shards. For each question:

1. Sample two initial completions at temperature `0.7` and `top_p=0.95`.
2. If neither initial attempt is a strict-positive answer, request two targeted retries at temperature `0.9` using diversified prompt templates.
3. Keep the requested 0/2/4-shot prompt mixture with weights `[0.60, 0.25, 0.15]`; the largest share is zero-shot.
4. Deduplicate correct completions by the reasoning key and retain at most two distinct strict-positive training rows per source question.
5. Preserve each attempt, retry status, source question ID, shot count, prompt-template ID, and shard statistics for audit.

The output validator checks both shard files, per-question attempt counts, duplicate question IDs, retry conditions, and stats files before training. A low yield is reported in the collection record. Missing or invalid shards prevent the handoff from becoming ready.

### A3. Fixed-route adapter

Train architecture `ars.fixed_route_low_rank_residual` version `1.0.0` on route `[12, 15, 18, 21]`, with rank 16 and `rho_max=0.10`. The backbone is frozen. The adapter trains on verified positive completions using token-mean causal NLL; prompt and padding tokens are excluded from the target loss. The full profile runs two epochs with micro-batch size 1 and gradient accumulation 32. Prompt templates are diversified and training shot weights are `[0.60, 0.25, 0.15]`.

Steering-intensity calibration and route-bank training are explicitly deferred to Stage B. Stage A calibrates p99 norms for the 14 candidate bank layers from the newly collected rollout prompts.

### A4. Stage A handoff

The handoff is ready only if its required phases and files pass validation. The ZIP contains the fixed adapter, fixed-route and candidate-layer norm calibrations, both rollout shards and their statistics, collection plan and summary, phase status, required code snapshots, and a root manifest with SHA-256 hashes and the pinned commit. The output is `ars_fixed_adapter_bundle.zip`.

## Stage B: alpha calibration, independent route bank, and router

Run `notebooks/01b_train_router_and_package.ipynb`. Attach either the Stage A ZIP or one extracted directory whose root contains `manifest.json`, `checkpoints/`, `calibration/`, `training/`, and `code/`. The notebook verifies manifest format, architecture ID/version, member paths, checksums, and the pinned code snapshot before training. It requires CUDA and both complete rollout shards. It does not recollect data or retrain the fixed adapter.

### Active full-run settings

The maintained Stage B notebook settings are `SMOKE_TEST=False` and `TRAINING_PROFILE="extended_quality"`:

| Variable | Active value | Meaning |
|---|---:|---|
| `BANK_EPOCHS` | 2 | Training passes per route expert |
| `ROUTE_EXAMPLES_PER_EXPERT` | 768 | Example budget per route expert |
| `SCREEN_VALIDATION_ROWS` | 24 | Held-out rows used to screen each expert |
| `ALPHA_CALIB_N` | 400 | Questions used for fixed-adapter gain calibration |
| `MAX_PAIR_ROUTES` | 8 | Maximum retained pair routes |
| `MAX_TRIPLE_ROUTES` | 8 | Maximum retained triple routes |
| `MAX_QUAD_ROUTES` | 8 | Maximum retained quad routes |
| `ROUTER_EPOCHS` | 3 | Maximum router training epochs |
| `MAX_ROUTER_ROWS` | 2,048 | Maximum router training rows |
| `STATIC_SCREEN_QUESTIONS` | 128 | Held-out questions for static-action selection |
| `TASK_VALIDATION_ROWS` | 84 | Held-out task rows for router checkpoint selection |
| `ROUTE_COST` | 0.01 | Cost added per active layer in action scoring |
| `ROUTER_ALPHAS` | `[1.0, 1.25, 1.5, 1.75, 2.0]` | Supported learned router gains |

The Stage A manifest supplies `CANDIDATE_LAYERS`, `BANK_MAX_QUESTIONS`, and the per-question row limit. The router question cap is `min(2,000, BANK_MAX_QUESTIONS)`. The candidate spacing settings are `MIN_LAYER_GAP=3`, `CONTROL_GAP=2`, and `SPACING_CONTROL_LIMIT=8`. Static-action screening uses gains `[1.0, 1.5, 2.0]`; the router's action catalog uses the five-value grid above. The bank's preference weight defaults to `0.25` in the training CLI.

**Run-specific note:** the final handoff's alpha-calibration JSON records `n=300` and selects alpha 1.0; its Stage B manifest records that alpha calibration was reused during the `router_only_resume`. The notebook profile listed 400 calibration questions. For reproduction of the shipped artifact, use the saved calibration file and manifest (`n=300`); for a new run, the current notebook setting is 400. These are different records and must not be conflated.

`standard` and `smoke` remain exploratory profiles. The active profile is printed before work begins and is saved into the training manifest. `SMOKE_TEST=True` overrides the profile values with the small smoke settings in the notebook.

### B1. Gain calibration and bank training

Notebook 01b calibrates the fixed adapter over the five-value gain grid and trains the independent route bank. When two GPUs are available, gain calibration runs on one GPU while bank training runs on the other. On one GPU the tasks run sequentially. Router training waits until the bank checkpoint passes its save and compatibility checks.

The bank curriculum proceeds from singleton routes to spaced pairs, triples, and quads. It screens candidates on held-out questions, keeps within the route caps, and saves each selected route as a separate expert. Preference pairs use a correct and a well-formatted wrong completion from the same prompt when available. The loss is positive-completion token-mean NLL plus a weighted softplus pairwise margin. Experts remain independent after initialization and fine-tuning; the fixed adapter is not copied into the bank.

### B2. Static-bank action

The pipeline scores route/gain actions on a held-out static-selection split, accounting for `ROUTE_COST × k`. It uses preference evidence where available and completion NLL as a fallback. A separate task-accuracy screen selects one fixed bank action. The resulting action is saved as `session3_static_bank_action.json` and is reused by the static-bank evaluation arm.

### B3. Router training

The router is trained from route-action teacher scores for up to `MAX_ROUTER_ROWS` and `MAX_ROUTER_QUESTIONS`. Its input is pooled prompt hidden states plus normalized shot count and question-token count. It learns scores for the finite route/gain catalog, including `k=0`. Its loss combines smooth-L1 regression to normalized teacher scores and a KL term for teacher action preference. Route cost is included when teacher action scores are built.

The router uses up to 84 held-out task-validation rows for checkpoint selection. The notebook records route, active-layer-count, and gain counts from teacher choices and the learned policy. These counts describe observed behavior; they do not prove that a route is optimal or that diversity improves accuracy.

### B4. Stage B completion gate

Notebook 01b writes `ars_training_bundle.zip` only after it records a complete bank and router, static action, alpha calibration, required norms, and matching artifact hashes. The root manifest records training profile, source Stage A commit and content identity, architecture contracts, route catalog, artifact hashes, and phase statuses. Evaluation notebooks require `training_ready=true` and the full required artifact set.

## Execution boundaries

- Smoke settings establish that code paths run; they do not estimate final quality.
- A failed phase does not become successful because later cells produce a partial ZIP.
- Preserve a completed Stage A bundle before beginning Stage B.
- Preserve Stage B output before running benchmark notebooks.
- The test set is reserved for final paired evaluation. It is not used to tune bank routes, static actions, or router checkpoints.
