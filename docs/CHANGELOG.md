# Changelog

This changelog tracks the maintained final-study package. Historical prototype results and design notes are preserved in [`archive/legacy_experiment_changelog.md`](archive/legacy_experiment_changelog.md).

## 2026-11-25

### Changed

- Reorganized current architecture, training, evaluation, artifact, and reproducibility documentation around the two-stage final-study workflow.
- Recorded the current `extended_quality` Stage B configuration and aligned Notebook 01b with the user-provided run settings.
- Moved the maintained notebooks to `notebooks/`, documented the router-only recovery notebook, and versioned the full workflow.
- Added latency, throughput, token, runtime, and artifact-hash provenance to benchmark records and summary tables.
- Added ASDiv numeric transfer support and AQuA-RAT parsing for the multiple-choice evaluation.
- Versioned the maintained research presentation and its editable diagram sources with matching PNGs.
- Replaced legacy architecture drawings with current editable diagrams and matching PNGs.
- Rebuilt the presentation narrative while retaining the existing black, white, and orange theme, Calibri typography, and 16:9 layouts.
- Added the checked GSM8K, GSM-Plus, and ARC-Challenge result summary, paired inference, router-use analysis, answer-quality metrics, runtime figures, and explicit resource/claim limitations.
- Added official dataset/model links and related methodological references; distinguished completed evaluation sets from datasets supported by code but not run.
- Corrected the run provenance note: the final Stage B bundle is a router-only resume and reuses a 300-question alpha calibration artifact, while the current notebook profile is configured for 400.
- Documented the maintainer-reported Hugging Face release, its local manifest checksums, and the fact that the Phi-2 base weights remain separate.

### Release note

- The raw evaluation ZIPs and trained PyTorch weights remain outside Git. They are preserved locally and are intended for the separate model-hosting release.
