# Adaptive Residual Steering (ARS)

ARS trains small residual-stream steering modules for mathematical reasoning while keeping the `microsoft/phi-2` backbone frozen. The repository contains three research sessions, shared data and evaluation code, and Kaggle notebooks for execution.

## Current execution paths

| Session | Purpose | Entry point | Main artifacts |
|---|---|---|---|
| Session 1 | Collect verified GSM8K rollouts, train fixed-route SteerNet, and compare steering with baseline | `scripts/run_rft_pipeline.py` and `scripts/run_ars_pipeline.py` | `steernet.pt`, rollout shards, paired benchmark reports |
| Session 2 | Train or evaluate the V2 fixed route `[12, 15, 18, 21]` using rank 16 and calibrated p99 norms | `scripts/run_rft_pipeline.py` with `--steernet_arch v2` | V2 checkpoint and norm calibration |
| Session 3 | Train a variable-route steering bank and prompt-conditioned router; compare baseline, fixed V2, validation-selected static bank, and router | `scripts/run_session3_router.py` | steering-bank checkpoint, router checkpoint, paired GSM8K/OOD reports and diagnostics |

Session 3 can reuse attached rollout shards and existing V2 artifacts. Its Kaggle notebook also supports collecting rollouts or rebuilding missing V2 artifacts when configured to do so. Notebook configuration cells define the exact paths and execution scale for a run.

## Model and invariants

- Phi-2 parameters remain frozen; training updates steering and routing modules only.
- Phase 1 uses token-mean NLL on verified positive RFT completions.
- The Phase 1 learning rate is locked at `1.5e-4`.
- `steer_only` is the isolated fixed-route steering arm: static layers, no gate provider, no router selection, and unit intervention gain.
- Session 2 V2 uses route `[12, 15, 18, 21]`, rank 16, `rho_max=0.10`, p99 norm calibration, and no magnitude head or convolution.
- Session 3 trains bank adapters on positive RFT completions plus matched, well-formatted wrong-answer preferences. Bank and router selection use held-out preference data; static-action selection and router checkpoint selection also use greedy answer accuracy. The test benchmark remains separate.
- Session 3 requires all four arms before the notebook reports a complete GSM8K comparison. It records paired metrics and route usage; route diversity is an empirical outcome and must be interpreted with accuracy.

## Repository layout

```text
src/
├── configs/       Configuration and locked session presets
├── data/          GSM8K loading, prompt construction, and shielding masks
├── deploy/        Deployment calibration contracts
├── evaluation/    Generation, answer scoring, paired benchmarks, OOD evaluation
├── gating/        Optional gate modules retained for supported legacy experiments
├── models/        Frozen Phi-2 wrapper and steering hooks
├── routing/       Fixed and adaptive route-selection models
├── steering/      SteerNet V5/V2 implementations and norm loading
├── training/      RFT, fixed-route, on-policy, checkpoint, metric, bank, router, batch, and optimizer utilities
└── utils/         Logging and CLI runtime setup

scripts/
├── calibrate_norms.py        Collect layer norm calibration statistics
├── run_ars_pipeline.py       Existing fixed-route, gate, router, and benchmark entry point
├── run_rft_pipeline.py       Session 1/2 equivalence, Gate 0, rollout, and RFT stages
└── run_session3_router.py   Session 3 steering-bank, router, and benchmark stages

notebooks/
├── kaggle_ars_runner.ipynb             Session 1/2 execution
├── kaggle_ars_session3_router.ipynb    Session 3 execution and diagnostics
├── kaggle_ars_equivalence_debug.ipynb  Isolated generation-equivalence experiments
└── ood_evaluation.ipynb                Out-of-distribution evaluation

tests/                    Unit and contract tests for supported paths
docs/                     Architecture, training, and change documentation
```

## Setup

Install the packages listed in `requirements.txt` in an environment with PyTorch configured for the target hardware. The benchmark and training scripts download Phi-2 and GSM8K through Hugging Face, so the runtime needs access to those artifacts or a populated local cache.

Run the repository test suite from the project root:

```powershell
python -m pytest tests/ -x -q
```

For Kaggle, open the relevant notebook and run its runtime setup and configuration cells before the session cells. The notebooks print the selected artifact paths, scale settings, phase status, and output directory. Keep generated checkpoints and reports in the run output directory; they are not source files.

## Entry-point stages

`scripts/run_rft_pipeline.py` exposes `equivalence`, `gate0`, `collect`, `train`, and `all` stages. Use `--n_questions` and stage-specific options to control probe and collection sizes. `--skip_collection` with `--rollout_dir` reuses existing shards for training; V2 runs also require a compatible p99 norm-calibration file.

`scripts/run_session3_router.py` exposes `train_bank`, `train_router`, and `benchmark`. The bank and router reuse the rollout corpus, splitting questions into separate training, bank-screen, router-validation, and static-action-selection partitions. The benchmark requires baseline, fixed V2, static bank, and router arms when `--require_all_arms` is set. Numeric OOD suites can be requested separately and are reported per suite, even if an OOD source is unavailable.

The existing `scripts/run_ars_pipeline.py` remains the compatibility entry point for fixed-route Phase 1, optional contrastive/STaR experiments, legacy router/gate stages, deployment calibration, GSM8K benchmarking, and OOD runs. Its public imports are also used by the other scripts and tests, so retain them when reorganizing the implementation.

## Research results

Results are run-specific. Use the paired JSON report generated for the exact run, test subset, seed, and prompt-shot condition when comparing arms. Do not treat a 50-, 400-, or 800-question subset as a full GSM8K test-set result, and do not compare unpaired accuracy values when paired per-question records are available.

## License

MIT
