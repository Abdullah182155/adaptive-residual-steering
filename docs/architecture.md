# ARS Architecture

This document describes the implementation currently in `src/` and the three execution sessions. Historical audit reports under the repository root describe earlier experiments and are retained as records; their settings are not current defaults.

## System boundary

ARS wraps `microsoft/phi-2` and adds trainable modules to selected residual-stream layers. The backbone is frozen. Training updates steering adapters and, in adaptive runs, a route-selection model. Inference can use a fixed route or a prompt-conditioned route.

```text
prompt tokens
    │
    ▼
frozen Phi-2 ──► selected residual hooks ──► next-token generation
                    │
                    ├── fixed SteerNet route
                    └── adaptive steering bank + route-value router
```

`src/models/wrapper.py` owns hook registration, prompt shielding, route masks, and intervention gains. `src/models/loading.py` builds the frozen Phi-2 backbone. The legacy gate remains available to the older pipeline stages, but the Session 1/2/3 Kaggle paths set it off.

## Session architectures

| Session | Steering modules | Active route policy | Purpose |
|---|---|---|---|
| 1 | SteerNet V5, rank 32, causal convolution, `rho_max=0.08`, identity magnitude head | Fixed route `[17, 20, 23, 26]` | Train from verified positive RFT completions and compare isolated steering with baseline |
| 2 | SteerNet V2, rank 16, no convolution, no magnitude head, `rho_max=0.10` | Fixed route `[12, 15, 18, 21]` | Test the redesigned adapter on the same RFT objective |
| 3 bank | One V2-format adapter per candidate layer, using p99 calibrated layer norms | Variable singleton, pair, triple, or quad subset; `k=0` is the no-intervention action | Learn adapters for multiple routes from Session 1 style rollouts |
| 3 router | RouteValueRouter over a finite catalog of bank routes and gains | Prompt features select one catalog action per question | Predict route scores learned from teacher-forced route evaluation |

The Session 3 router receives prompt-derived features. Positive completions are used to score candidate actions when building training targets; those completions are not router input features at inference. Route variety is measured in benchmark diagnostics and is not guaranteed by the architecture alone.

## SteerNet modules

Both adapter variants project the hidden state to a low-rank representation, apply GELU, and project back to the residual width. V5 may add causal depthwise convolution. V2 omits convolution and the magnitude head. V2 initializes its up projection to zero, so the adapter begins with a zero residual delta.

The steering correction is relative to the residual-stream norm and is bounded by the active route's configured amplitude and the calibrated per-layer reference. Session 2/3 V2 uses p99 norm statistics. Session 1 V5 uses its configured norm statistic. The forward implementation and exact clamp semantics are defined in `src/steering/steernet.py`; this document intentionally avoids duplicating the formula in a way that can drift from the code.

`src/steering/steernet.py` also owns loading calibration files. V2 requires the selected statistic to exist for each active layer; calibration is not silently substituted across statistics.

## Routing and gating

The repository contains two routing families:

- `src/routing/router.py` and `src/training/router_trainer.py`: the older JointLayerRouter and RLOO training path exposed by `scripts/run_ars_pipeline.py`.
- `src/routing/adaptive_subset_router.py` and `src/training/adaptive_router_training.py`: the Session 3 finite-catalog route-value router, trained from measured route scores.

These are separate experiment paths and should not be treated as interchangeable checkpoints. The Session 3 benchmark uses baseline, fixed V2 steering, and the adaptive router when compatible bank and router artifacts are present. It records route usage so a collapsed policy can be detected.

`src/gating/` and `src/training/gate_trainer.py` implement optional legacy gate experiments. Session 1's `steer_only` arm explicitly removes gate providers and route selectors and applies the fixed adapter route at unit gain.

## Generation and evaluation

`src/evaluation/generation.py` provides shared left-padded generation, sequence-level stopping, precise prompt-width slicing, and the batch equivalence harness. `src/evaluation/benchmark.py` evaluates arms on the same ordered test rows and records paired per-question outcomes, accuracy intervals, and disagreement statistics. A subset report is valid for its stated subset and seed only; it does not stand in for a full GSM8K test result.

Answer extraction and chain-level diagnostics live in `src/evaluation/answers.py` and `src/evaluation/chain.py`. OOD dataset loading and scoring are separate from the GSM8K benchmark.

## Package map

```text
src/configs/       Locked configuration and validated overrides
src/data/          GSM8K loading, prompt construction, benchmark sampling
src/deploy/        Deployment calibration contracts
src/evaluation/    Generation, metrics, paired benchmark, OOD evaluation
src/gating/        Optional legacy gates
src/models/        Phi-2 wrapper and frozen-model construction
src/routing/       Legacy and adaptive routing models
src/steering/      SteerNet variants and layer norm loading
src/training/      RFT, fixed-route, on-policy, checkpoint, metric, gate, bank, router, batching, and optimizer utilities
src/utils/         Shared logging and command-line runtime setup
scripts/           CLI orchestration for the supported experiment paths
notebooks/         Kaggle execution and isolated debugging notebooks
```
