# Trained model release

## Public model repository

The project maintainer reports that the trained ARS files have been uploaded to the [Hugging Face model repository](https://huggingface.co/Abdullah182155/Adaptive_Residual_Steering). The repository link is the distribution location; the versioned local Stage B manifest is the provenance and checksum source for the evaluated model.

This is an adapter release for **Microsoft Phi-2**, not a standalone language model. Download the matching base model separately under its [model card and license terms](https://huggingface.co/microsoft/phi-2). Use the repository code at the training revision below and retain the matching calibration, static-action, and router metadata when reproducing the four evaluation arms.

## Evaluated artifacts

The final Stage B training bundle records `training_ready: true` and contains four model tensors:

| Artifact | Local bundle path | Size |
|---|---|---:|
| Fixed-route adapter | `checkpoints/fixed_route_adapter.pt` | 1,315,061 bytes |
| Independent route bank | `checkpoints/session3_steering_bank.pt` | 28,258,219 bytes |
| Router | `checkpoints/session3_router.pt` | 1,490,365 bytes |
| Route-teacher targets | `checkpoints/session3_route_teacher_targets.pt` | 4,200,011 bytes |
| **Total** |  | **35,263,656 bytes (33.6 MiB)** |

The bundle also includes p99 norm calibration, selected static-bank action, alpha calibration, router training metadata, a manifest, source snapshots, and checksums. The original bundle is `results/training_handoff/ars_training_bundle.zip`; it is intentionally excluded from Git.

## Reproducibility identity

- Base model: `microsoft/phi-2`
- Training code commit: `5bd78d89a79e375d8330edc04066c05a4d99837c`
- Evaluation code commit: `f0d102c21af3272d07bf8be74156eba2bdd8c8d6`
- Evaluation artifact fingerprint: `47a758610030`
- Training seed: 42
- Final Stage B run: `20261122_234525_20261124_111803_router_resume_20261124_200410`
- Source Stage A run: `20261122_234525`

The final Stage B run reused the trained bank, static action, and a 300-question alpha-calibration artifact, then resumed router training. The notebook profile listed a cap of 400 alpha-calibration questions; the artifact records the actual reused value of 300. This is documented to keep the reproducibility record aligned with the file actually evaluated.

## Integrity check

The SHA-256 values below come from the local Stage B manifest. When mirroring the release or moving files between systems, compare the downloaded files against these values or the full manifest in the bundle.

| Artifact | SHA-256 |
|---|---|
| Fixed-route adapter | `2f15d8ad1a74426c61ca314e5ccf45b4d15b599b9dfa04a41e4f2273502e8797` |
| Route bank | `b12fef3dcc1715b66fae10094f2728bdd8fdb93292ea68c9b0082bacdfde0ce8` |
| Router | `4af56ca7ca62896d4b2a4b47fc47b3ed17fad722c217b974b034360d7c27e7e7` |
| Fixed-route p99 norms | `94c81a6d06f8737403deeab095203f2bc6400f2fda994c67e00f6219bf88b954` |
| Candidate-layer p99 norms | `e4acb0307f381a50146c2bcf31af5c6b4f9e398271703223ca275ab5abd5fd24` |
| Static-bank action | `2adca849936525f54daf09ea6e471d764059f4329bc828c45478d54cd9fcc1e9` |
| Alpha calibration | `6e144c3c3b438f7717990979445acec8e3b3aaffa3dbea370e1ceb97306d64a7` |
| Router training metadata | `dae9ea80ba19987bbdcc7dfaffcfeb8a5cd8060edde50bdd72aaf10dff62651f` |
| Route-teacher targets | `a9e01097b04dfd512c382581c52944211390dabcbee7dad5443db90935917334` |

## What the evaluation supports

On the saved GSM8K, GSM-Plus, and ARC-Challenge evaluations, fixed steering improved 0-shot accuracy over baseline. The static bank and router also improved 0-shot point accuracy over baseline, but neither beat fixed steering. The 2-shot comparisons do not establish a reliable gain. These limits apply to the released checkpoint; see [the full result record](final_results.md) and [claim boundaries](limitations.md).
