# Adaptive Residual Steering (ARS)

ARS studies small trainable residual-stream adapters for mathematical reasoning with a frozen `microsoft/phi-2` backbone. The final study compares a fixed adapter, a validation-selected action from an independent route bank, and a prompt-conditioned router against the unmodified model.

The repository contains the training and evaluation code, the final-study notebooks, editable architecture diagrams, and the research presentation. The completed evaluation currently covers GSM8K, GSM-Plus, and ARC-Challenge at 0-shot and 2-shot. The results are mixed: fixed steering improves 0-shot accuracy on all three evaluated sets, while the current static bank and router do not outperform fixed steering; no reliable 2-shot gain is established. See [the measured results](docs/final_results.md) and [limitations](docs/limitations.md) for the claim boundaries.

The trained adapter release is hosted at [Hugging Face](https://huggingface.co/Abdullah182155/Adaptive_Residual_Steering). It requires the separate Phi-2 base model. See [model release notes](docs/model_release.md) for the checkpoint contents, provenance, and local manifest checksums.

The research deck is [`ARS_Research_Presentation.pptx`](ARS_Research_Presentation.pptx). It explains the research question, method, measured results, statistical evidence, resource constraints, and remaining limitations. Dataset and related-work sources are listed in [References](docs/references.md).

## Final-study workflow

| Order | Notebook | Purpose | Input | Main output |
|---|---|---|---|---|
| 1 | `notebooks/01a_train_fixed_and_bank.ipynb` | Fresh fixed-route adapter, GSM8K rollout collection, and norm calibration | No previous model or rollout artifacts; CUDA and internet | `ars_fixed_adapter_bundle.zip` |
| 2 | `notebooks/01b_train_router_and_package.ipynb` | Alpha calibration, independent route-bank training, static action selection, and router training | ZIP or extracted Stage A handoff | `ars_training_bundle.zip` |
| Recovery | `notebooks/01c_resume_router_and_package.ipynb` | Resume router training from a saved Stage B run without repeating bank training | Matching partial Stage B output | Updated `ars_training_bundle.zip` |
| 3 | `notebooks/02_gsm8k_in_distribution.ipynb` | Four-arm GSM8K evaluation | Complete training bundle | `ars_gsm8k_results.zip` |
| 4 | `notebooks/03_math_ood_evaluation.ipynb` | Numeric math transfer | Complete training bundle | `ars_math_ood_results.zip` |
| 5 | `notebooks/04_multiple_choice_transfer.ipynb` | Multiple-choice transfer | Complete training bundle | `ars_mcq_results.zip` |
| 6 | `notebooks/05_router_ablations.ipynb` | Read-only route-use and paired-result diagnostics | Any available evaluation bundles | Tables and diagnostic JSON |
| 7 | `notebooks/06_paper_figures.ipynb` | Figures and tables from saved records | Any available evaluation bundles | `ars_paper_assets.zip` |

Notebook 01a starts from fresh data and checkpoints. Notebook 01b consumes its checksummed handoff and does not repeat rollout collection or fixed-adapter training. Notebook 01c is only for resuming a matching partial Stage B run. Later notebooks require a complete Stage B training bundle. See [the notebook guide](notebooks/README.md) for inputs, outputs, run slices, and recovery behavior.

## Current Stage B training profile

Notebook 01b currently uses `SMOKE_TEST=False` and `TRAINING_PROFILE="extended_quality"`.

| Setting | Value |
|---|---:|
| Bank epochs | 2 |
| Examples per route expert | 768 |
| Held-out bank-screen rows | 24 |
| Alpha-calibration questions | 400 |
| Pair / triple / quad route caps | 8 / 8 / 8 |
| Router epochs / maximum rows | 3 / 2,048 |
| Static-action screen / task-validation rows | 128 / 84 |
| Route cost | 0.01 |
| Router gain grid | 1.0, 1.25, 1.5, 1.75, 2.0 |

The candidate layers come from the Stage A manifest. Bank questions and per-question training-row limits also come from that manifest. Router questions are capped at `min(2,000, bank questions)`. On a two-GPU runtime, alpha calibration and bank training run concurrently; on one GPU they run sequentially. The notebook prints the effective profile before launching work.

## Four evaluation arms

1. **Baseline:** frozen Phi-2 without steering.
2. **Fixed steering:** one trained adapter on route `[12, 15, 18, 21]`, with gain selected by validation calibration.
3. **Static bank:** one route and gain selected from the route bank on held-out data.
4. **Router:** one route and gain selected from a finite catalog using prompt-only features and prompt context.

All arms receive the same questions, shot condition, prompt schedule, and decoding settings within each paired comparison. Route diversity is reported as a diagnostic and is interpreted together with paired accuracy.

## Repository map

```text
src/                       Model, data, training, routing, and evaluation code
scripts/                   CLI entry points and research asset generation
notebooks/                 Versioned training, evaluation, diagnostics, and figures
diagrams/                  Editable Draw.io sources and matching PNG figures
docs/                      Current methodology, architecture, contracts, and protocol
docs/archive/               Superseded designs and historical experiment log
tests/                     Unit and contract tests
```

The notebooks listed above form the maintained run sequence. Scratch notebooks remain ignored by Git.

## Runtime

The training and benchmark notebooks need a CUDA GPU and internet access for source and model/data retrieval. The reported training run used Kaggle Tesla T4 GPUs; the longest stage exceeded a 12-hour session and required a router-only resume. Evaluation time depends strongly on dataset, shot count, generated-answer length, and platform load. Peak host RAM and peak VRAM were not recorded, so the project does not claim measured memory requirements. Install `requirements.txt` in a Python environment with a compatible PyTorch build. For local code checks, run from the repository root:

```powershell
python -m pytest tests/ -x -q
```

Do not commit generated weights, rollout JSONL, benchmark outputs, or bundle ZIPs. They remain local or in the notebook platform's attached artifacts. The final presentation and the small architecture figures are maintained repository deliverables.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Training pipeline](docs/training.md)
- [Evaluation protocol](docs/evaluation_protocol.md)
- [Artifact contracts](docs/artifact_contracts.md)
- [Reproducibility](docs/reproducibility.md)
- [Final measured results](docs/final_results.md)
- [Limitations and claim boundaries](docs/limitations.md)
- [Trained model release](docs/model_release.md)
- [Datasets and related-work references](docs/references.md)
- [Diagram sources](diagrams/README.md)
- [Historical material](docs/archive/README.md)
