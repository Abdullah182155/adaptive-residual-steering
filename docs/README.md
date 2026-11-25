# ARS documentation

This folder describes the maintained final-study workflow. Notebook code and handoff manifests define run-specific values; this documentation explains their meaning and contracts. Historical prototype descriptions are preserved under `archive/` and do not define current behavior.

## Current reference documents

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | Frozen backbone, fixed adapter, independent route bank, router, and inference flow |
| [Training pipeline](training.md) | Stage A and Stage B procedures, objectives, current profile, and completion gates |
| [Evaluation protocol](evaluation_protocol.md) | Four arms, datasets, shot conditions, paired statistics, and performance metrics |
| [Artifact contracts](artifact_contracts.md) | Required handoff files, manifests, checksums, and ZIP contents |
| [Reproducibility](reproducibility.md) | Seeds, commits, hashes, run provenance, and resume boundaries |
| [Limitations](limitations.md) | Interpretation rules and claims the design does not establish by itself |
| [Changelog](CHANGELOG.md) | Changes to the maintained final-study package |

## Where to start

1. Read the root [README](../README.md) for the run order and current profile.
2. Use `notebooks/README.md` for each notebook's required input and output.
3. Use `artifact_contracts.md` before transferring a ZIP or extracted Kaggle Model input.
4. Use `evaluation_protocol.md` to interpret saved records after the benchmark runs finish.

Results are intentionally absent from the current docs until the pending evaluation records are available and checked.
