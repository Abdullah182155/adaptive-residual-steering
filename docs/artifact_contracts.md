# Artifact contracts

The training stages exchange explicit, checksummed bundles. A Kaggle Model may expose a ZIP or an extracted directory; the root manifest determines whether that input belongs to this pipeline.

## Stage A fixed-adapter handoff

File name: `ars_fixed_adapter_bundle.zip`.

The archive root contains `manifest.json`. The manifest declares format `ars_fixed_adapter_handoff_v1`, architecture ID/version, run ID, code commit, model, route, candidate layers, rank, `rho_max`, norm statistic, seed, shot weights, collection policy, phase status, and SHA-256 hashes.

Required members include:

| Member | Purpose |
|---|---|
| `checkpoints/fixed_route_adapter.pt` | Fresh fixed-route adapter checkpoint |
| `calibration/fixed_route_p99_norms.json` | Fixed-route p99 references |
| `calibration/candidate_norms_p99.json` | Candidate-layer p99 references for the bank |
| `training/rollouts/shard0.jsonl` | First verified rollout shard |
| `training/rollouts/shard1.jsonl` | Second verified rollout shard |
| `training/collection_plan.json` | Requested rollout policy |
| `training/collection_summary.json` | Validated shard and positive-yield summary |
| `code/...` | Exact source snapshot required to continue the run |

The manifest records hashes for model artifacts, rollout files, collection metadata, and code. Stage B rejects missing members, path traversal, unsupported architecture identity, and hash mismatches.

## Stage B training bundle

File name: `ars_training_bundle.zip`.

The archive root contains `manifest.json` and the completion status. Required model/calibration members are:

| Member | Purpose |
|---|---|
| `checkpoints/fixed_route_adapter.pt` | Stage A fixed adapter |
| `checkpoints/session3_steering_bank.pt` | Independent route experts and route catalog |
| `checkpoints/session3_router.pt` | Prompt-conditioned catalog router |
| `calibration/fixed_route_p99_norms.json` | Fixed-route norm references |
| `calibration/candidate_norms_p99.json` | Candidate-layer norm references |
| `selection/session3_static_bank_action.json` | Held-out-selected static route and gain |
| `training/alpha_calibration.json` | Fixed-adapter gain sweep and selected gain |

The bundle also preserves the parent handoff content identity, training profile, route catalog, code snapshot references, per-artifact hashes, rollout shards, phase statuses, and useful logs. It must report `training_ready: true` before evaluation uses it.

## Evaluation bundles

Each evaluation notebook writes its JSON and per-question records, log, and summaries. The result ZIP may also include CSV summaries for accuracy/performance, paired comparisons, route usage, and a manifest of available and missing conditions. Output names include the suite, shot count, sample budget, and seed. Notebooks merge prior result bundles before adding new slices, so keep the latest ZIP when moving to the next session.

The analysis notebook creates `ars_paper_assets.zip` from saved evaluation records. It contains PNG figures, CSV tables, a JSON summary, and a manifest linking figures to source records. It does not run model inference or invent missing measurements.

## Safe transfer rules

1. Keep each ZIP intact during transfer; Kaggle also accepts an extracted Stage A or Stage B directory when the root manifest is present.
2. Attach only one training bundle or one matching extracted tree to an evaluation run.
3. Preserve the current evaluation result ZIP before starting another slice.
4. Do not substitute a checkpoint or norms file from another run without updating and validating the manifest.
5. Keep model weights, rollout data, benchmark outputs, and ZIP bundles out of Git. The small code snapshots inside research bundles are separate provenance copies.
