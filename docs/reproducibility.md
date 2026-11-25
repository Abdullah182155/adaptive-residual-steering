# Reproducibility and provenance

ARS uses run IDs, pinned source commits, deterministic question sampling, and checksummed artifacts to tie a benchmark result to the exact trained model and evaluation implementation.

## Provenance chain

1. **Stage A:** record the repository commit, run ID, seed, model name, fixed route, candidate layers, training configuration, collection policy, phase status, and hashes for output files and code snapshots.
2. **Stage B:** validate the Stage A manifest, content checksums, architecture contract, and source snapshot before training. Record the source handoff identity, Stage B profile, independent bank catalog, router, static action, selected gain, hashes, and phase status.
3. **Evaluation:** record training and evaluation code commits separately, hash the required training artifacts, derive an artifact fingerprint, and record the dataset source/split, question IDs, seed, shots, prompt settings, and per-arm outputs.
4. **Analysis:** consume saved records and preserve a manifest mapping output tables and figures to source files and missing conditions.

## Seeds and matched data

The final-study notebooks use seed 42 for the main data and evaluation sampling. Stage A alternates unique question assignments across two rollout shards. Stage B splits by source question so completions from one question do not leak between training and validation partitions. Each benchmark arm receives the same ordered question IDs and prompt schedule within a shot condition.

Record actual row counts and dataset split names. A requested cap is a maximum and does not guarantee the source dataset contains that many eligible rows.

## Code revision rules

Notebook 01a fetches the configured development branch and records the resolved commit in its manifest. Notebook 01b checks out the exact Stage A commit and compares the source files with the handoff hashes. Evaluation notebooks record their current evaluation-code commit separately from the training-code commit embedded in the manifest. This separation allows benchmark code updates without silently replacing or relabeling the model that was trained.

If the Stage A or Stage B notebook code changes, preserve the run's printed commit and profile with its output bundle. Do not infer the code revision from the current branch after the run.

## Resume boundaries

- A completed Stage A handoff is the resume point for Stage B. Do not recollect rollouts to continue bank/router work.
- A completed Stage B bundle is the resume point for benchmark notebooks.
- Evaluation notebooks merge previous result bundles and isolate records by artifact fingerprint and code revision. Attach the latest merged bundle to continue missing slices.
- The benchmark test set is not a training or model-selection resume source.

## Current run configuration source

The active Stage B `extended_quality` profile is recorded in [the training guide](training.md) and printed in Notebook 01b. Values derived from the handoff manifest (candidate layers, maximum bank questions, and rows per question) belong to the specific Stage A run and must be read from that run's `manifest.json` rather than copied from a prior session.
