# Final-study notebook guide

This directory is the maintained end-to-end run sequence. Training is split into two GPU sessions so the Stage A output can be saved before the independent route-bank and router work begins. Every notebook lists required inputs and writes a self-contained handoff or result bundle.

## Run order and handoffs

| Notebook | Objective | Required input | Output |
|---|---|---|---|
| `01a_train_fixed_and_bank.ipynb` | Fresh fixed-route adapter, verified rollout collection, fixed/candidate norm calibration | No previous model, norm, or rollout files; internet and CUDA | `ars_fixed_adapter_bundle.zip` |
| `01b_train_router_and_package.ipynb` | Alpha sweep, independent route-bank experts, static-bank selection, router training | Stage A ZIP or extracted tree with root `manifest.json` | `ars_training_bundle.zip` |
| `01c_resume_router_and_package.ipynb` | Continue an interrupted 01b run by reusing the adapter, bank, calibration, and rollout artifacts | Saved output of the matching interrupted 01b run | Updated `ars_training_bundle.zip` |
| `02_gsm8k_in_distribution.ipynb` | Four-arm in-distribution benchmark, one shot condition per session | Stage B bundle; latest GSM8K result ZIP is optional on later runs | `ars_gsm8k_n{N}_shots_{...}_results.zip` |
| `03_math_ood_evaluation.ipynb` | GSM-Plus, SVAMP, or ASDiv math transfer | Complete Stage B training bundle | `ars_math_ood_results.zip` |
| `04_multiple_choice_transfer.ipynb` | AQuA-RAT or ARC-Challenge transfer | Complete Stage B training bundle | `ars_mcq_results.zip` |
| `05_router_ablations.ipynb` | Read-only route usage and paired-result diagnostics | One or more completed result bundles | JSON and CSV diagnostics |
| `06_paper_figures.ipynb` | Presentation and paper figures from saved results | One or more completed result bundles | `ars_paper_assets.zip` |

Keep the latest result ZIP when moving between slices. Notebooks 2-4 merge attached prior results before adding new records. Notebooks 5-6 load saved outputs only; they do not load Phi-2 or run model inference.

Notebook 01c reuses the saved bank and does not repeat route-bank training or calibration. The current teacher-scoring loop writes its targets after scoring the selected questions, so an interrupted scoring pass starts again from that stage with a smaller question cap; it cannot resume at the exact interrupted forward.

## Notebook 01a: fresh Stage A

Attach no prior checkpoints, calibrations, or rollouts. The full profile uses fixed route `[12, 15, 18, 21]`, rank 16, `rho_max=0.10`, p99 norm calibration, and two training epochs. It collects 5,000 unique GSM8K questions with two initial attempts and up to two targeted retries for questions with zero strict-positive initial attempts. It keeps at most two distinct strict-positive training rows per question and uses 0/2/4-shot weights `[0.60, 0.25, 0.15]`.

The notebook verifies both rollout shards and required statistics, trains the fixed adapter on the new collection, calibrates candidate-layer norms, and writes a SHA-256-protected handoff. It does not train the route bank or router. The handoff carries `manifest.json`, the fixed adapter, both norm files, rollout shards/statistics, collection metadata, phase status, and the pinned source snapshot.

## Notebook 01b: active Stage B profile

Attach the exact output of 01a as `ars_fixed_adapter_bundle.zip` or attach its extracted directory with root `manifest.json`. Do not attach multiple candidate handoffs. Stage B validates the architecture identity, file paths, hashes, and code snapshot before using the data.

The current settings are `SMOKE_TEST=False` and `TRAINING_PROFILE="extended_quality"`:

| Setting | Value |
|---|---:|
| Bank epochs | 2 |
| Examples per route expert | 768 |
| Held-out route-screen rows | 24 |
| Alpha-calibration questions | 400 |
| Pair / triple / quad route caps | 8 / 8 / 8 |
| Router epochs / maximum rows | 3 / 2,048 |
| Static-action questions / task-validation rows | 128 / 84 |
| Route cost | 0.01 |
| Router gain grid | `[1.0, 1.25, 1.5, 1.75, 2.0]` |
| Minimum layer spacing / control spacing / controls cap | 3 / 2 / 8 |

The Stage A manifest supplies the candidate layers, bank question cap, and per-question row cap. Router questions are limited to `min(2,000, bank questions)`. Two visible GPUs run alpha calibration and bank training concurrently; one GPU runs them sequentially. Router training starts only after the bank passes its checkpoint validation. `training_ready` must be `true` before attaching the resulting bundle to an evaluation notebook.

## Notebook 02: GSM8K

Attach one Stage B bundle or its extracted tree. Run one shot condition per Kaggle session. In the configuration cell, set `SHOT_COUNT` to `0`, `2`, or `4` and keep `BENCH_N=1319` for the full official GSM8K test split. Every arm shares the same seeded test IDs and prompt schedule within that condition. The four arms are baseline, fixed adapter, static bank, and router.

The default 0-shot run is expected to take about 7–8.5 hours on a Tesla T4, based on the prior measured 700-question four-arm run. A full 2-shot run is estimated at about 5–6 hours. A full 4-shot run is less certain and is budgeted at about 5–7 hours. Run each shot condition in a separate session. Download the cumulative result ZIP after every run; its filename records the data count and complete shot conditions, and it includes JSON/JSONL records, logs, timing and quality metrics, paired comparisons, router-use tables, code snapshots, and a bundle manifest.

For the first run, attach only the Stage B `ars_training_bundle.zip`. For each later shot condition, attach the same training bundle and the latest cumulative GSM8K result ZIP to retain earlier conditions in the final table and archive. Stage A's fixed-adapter ZIP is not needed.

## Notebook 03: numeric math transfer

Choose exactly one of `gsm_plus`, `svamp`, or `asdiv` per session. Each dataset uses up to 500 paired questions at 0-shot and 2-shot with seed 42. The report records the actual split, question IDs, and paired count. The selected ASDiv release uses its validation pool as an external evaluation pool, not an official test split. MATH-500 is excluded until a symbolic-equivalence grader has been verified.

## Notebook 04: multiple-choice transfer

Choose `aqua_rat` or `arc` per session. The notebook requests up to 500 test questions at 0-shot and 2-shot with seed 42. It uses demonstrations only from training data. The saved record reports the actual eligible and paired question count.

## Notebooks 05 and 06: analysis after evaluation

Notebook 05 summarizes observed route count, entropy, most-used-route share, `k`/layer/gain use, paired accuracy, and arm comparisons. Diversity is interpreted alongside paired performance. Notebook 06 creates figures and tables only from saved records; missing values stay missing and are listed in its manifest.

The analysis tables include paired accuracy and confidence intervals, wins/losses/ties, McNemar results, latency and throughput with timing coverage, token counts, stop reasons, answer-format/quality checks, and route-use summaries where the source record supports them.

## Runtime and output handling

- 01a and 01b need CUDA. Notebook 01a supports Kaggle or Colab runtime detection; Stage B accepts the matching extracted handoff or ZIP.
- 02-04 need CUDA for model inference and internet or cached model/data artifacts.
- 05-06 are CPU-only and require only completed result bundles.
- Copy each completed ZIP to durable storage before starting the next long session.
- A partial run is not a completed result. Check each notebook's final phase-status object and manifest.

Current benchmark values and conclusions are intentionally left pending until the full evaluation records are available.
