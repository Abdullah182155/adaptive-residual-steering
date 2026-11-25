# ARS documentation

This folder describes the maintained final-study workflow. Notebook code and handoff manifests define run-specific values; this documentation explains their meaning and contracts. Historical prototype descriptions are preserved under `archive/` and do not define current behavior.

## Current reference documents

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | Frozen backbone, fixed adapter, independent route bank, router, and inference flow |
| [Training pipeline](training.md) | Stage A and Stage B procedures, objectives, current profile, and completion gates |
| [Evaluation protocol](evaluation_protocol.md) | Four arms, datasets, shot conditions, paired statistics, and performance metrics |
| [Final measured results](final_results.md) | Verified accuracy, paired uncertainty, answer quality, router behavior, runtime, and run provenance |
| [Trained model release](model_release.md) | Hugging Face link, checkpoint contents, checksums, and reproduction identity |
| [Artifact contracts](artifact_contracts.md) | Required handoff files, manifests, checksums, and ZIP contents |
| [Reproducibility](reproducibility.md) | Seeds, commits, hashes, run provenance, and resume boundaries |
| [Limitations](limitations.md) | Interpretation rules and claims the design does not establish by itself |
| [References](references.md) | Official dataset and model sources, plus related methodological papers |
| [Changelog](CHANGELOG.md) | Changes to the maintained final-study package |

## Where to start

1. Read the root [README](../README.md) for the run order and current profile.
2. Use `notebooks/README.md` for each notebook's required input and output.
3. Use `artifact_contracts.md` before transferring a ZIP or extracted Kaggle Model input.
4. Use `final_results.md` for completed benchmark evidence, `limitations.md` for claim boundaries, and `evaluation_protocol.md` for how it was produced.

Raw run bundles and the analysis archive are retained locally under `results/` and are excluded from Git. The result tables in `final_results.md` are the versioned summary; the exact run commits and artifact fingerprint are recorded there. Datasets or conditions without a saved result are explicitly marked unevaluated.
