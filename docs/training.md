# Training & Staged Execution Pipeline

Run end-to-end with `python scripts/run_ars_pipeline.py --stage all`, or one stage at a time.

| Stage | Purpose | Output / decision |
|---|---|---|
| S0  | Counterfactual table over the subset catalog + signal audit | Decision 1 (is there routable signal?) |
| S1  | SteerNet subset trainer (on-policy traces + layer-dropout), gate fixed open | SteerNet checkpoint |
| S0b | On-policy reasoning trace sampler | Trace set used by S1/S2 |
| S2  | Catalog router distillation (26-subset catalog, question-span pooling) | Router checkpoint |
| S3  | Direct task-loss token gate with dominance hinge | Gate checkpoint |
| S5  | Deployment calibration on the selection split ($n=120$) | `deployment_contract.json` |
| bench | 5-arm production benchmark (GSM8K test, 0/8-shot) | `experiment_record.json` |

Report: `python scripts/make_report.py --record experiment_record.json` -> `report/REPORT.md`, `report/tables/*.tex`.

## Failure policy

- A stage may return `{"halt": true}`; the orchestrator stops (exit code 2) and persists all prior decisions in `checkpoints/pipeline_state.json`.
- Runtime guards (finite check, relative-norm envelope, `k_max`) flag an item; a flagged item reverts to the unsteered ($K=0$) output.
- The deployment contract is immutable: checkpoint SHA-256s are verified before the benchmark loads any weights.
- Gate adoption is conservative: the learned gate is used only if it is paired-NLL-better and not accuracy-worse than passthrough.

## Evaluation arms

Baseline (`override="none"`), Fixed-control ($s^*_{\text{fixed}}$, const gate), Router-only ($\theta^*$, const gate), Full ARS ($\theta^*$, adopted gate, guards), Raw ARS ($\theta=1.01$, learned gate, no guards).  Generations are cached by `(prompt, route_key)`; every empty route shares the single baseline generation.
