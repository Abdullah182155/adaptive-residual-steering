# Final benchmark results

## Scope and provenance

This report is the measured evidence currently available for the handoff. It covers three dataset families, two shot settings, and four matched arms. It does not claim results for datasets whose result bundles are absent.

The corresponding adapter and route-policy artifacts are distributed from the [Hugging Face model repository](https://huggingface.co/Abdullah182155/Adaptive_Residual_Steering); see [release contents and checksums](model_release.md). The base Phi-2 model is separate.

| Dataset | Split and sample | 0-shot | 2-shot | Scoring |
|---|---|---:|---:|---|
| GSM8K | Official test split; 1,319 questions | 1,319 | 1,319 | Lenient numeric answer accuracy; strict and marker metrics retained |
| GSM-Plus | Test evaluation sample; 500 questions | 500 | 500 | Lenient numeric answer accuracy; strict and marker metrics retained |
| ARC-Challenge | Test split; seed-42 sample of 500 | 500 | 500 | Exact multiple-choice answer accuracy |

The four arms use the same question IDs, shot condition, prompt schedule, decoding settings, and trained artifacts within each paired comparison:

1. `baseline`: frozen Phi-2 without steering.
2. `steer_only`: fixed route `[12, 15, 18, 21]`, calibrated gain 1.0.
3. `static_bank`: one validation-selected bank action, route `[22]`, gain 1.0.
4. `router`: per-question action from the trained route/gain catalog.

Evaluation seed: 42. Training code commit: `5bd78d89a79e375d8330edc04066c05a4d99837c`. Evaluation code commit: `f0d102c21af3272d07bf8be74156eba2bdd8c8d6`. Artifact fingerprint: `47a758610030`. The per-condition result archives and `ars_paper_assets.zip` are stored in [`results/evaluations/`](../results/evaluations/).

## Accuracy

Percent correct. Delta is the paired percentage-point difference from baseline; the interval is the paired bootstrap 95% confidence interval. GSM8K and GSM-Plus use lenient numeric answer accuracy; ARC uses exact choice accuracy.

| Dataset | Shots | n | Baseline | Fixed steering | Static bank | Router |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 0 | 1,319 | 46.47% | **57.32%** (+10.84; 7.96 to 13.72) | 52.99% (+6.52; 3.71 to 9.40) | 51.48% (+5.00; 2.96 to 7.13) |
| GSM8K | 2 | 1,319 | 56.33% | 58.07% (+1.74; -0.23 to 3.79) | 55.42% (-0.91; -3.18 to 1.29) | 56.10% (-0.23; -0.68 to 0.23) |
| GSM-Plus | 0 | 500 | 31.60% | **41.20%** (+9.60; 5.20 to 14.20) | 37.40% (+5.80; 1.20 to 10.40) | 36.20% (+4.60; 0.80 to 8.20) |
| GSM-Plus | 2 | 500 | 44.40% | 44.40% (0.00; -3.40 to 3.60) | 43.40% (-1.00; -4.80 to 2.80) | 44.60% (+0.20; -0.80 to 1.20) |
| ARC-Challenge | 0 | 500 | 43.60% | **68.40%** (+24.80; 20.40 to 29.20) | 50.80% (+7.20; 2.60 to 12.00) | 51.20% (+7.60; 3.80 to 11.40) |
| ARC-Challenge | 2 | 500 | 76.20% | 75.40% (-0.80; -2.20 to 0.60) | 76.80% (+0.60; -0.60 to 1.80) | 76.40% (+0.20; 0.00 to 0.60) |

### Paired inference

The evaluator reports two-sided exact McNemar tests and paired bootstrap intervals. Benjamini–Hochberg correction was applied across all 18 arm-versus-baseline comparisons (three datasets × two shot settings × three intervention arms).

All nine 0-shot gains over baseline remain below q = 0.05 after that correction. The 2-shot comparisons do not: all nine adjusted q-values exceed 0.18. This supports a reproducible 0-shot improvement on these three tested sets, but it does not support a general few-shot improvement.

| 0-shot comparison vs baseline | GSM8K q | GSM-Plus q | ARC-Challenge q |
|---|---:|---:|---:|
| Fixed steering | 7.0e-13 | 9.7e-5 | 5.9e-25 |
| Static bank | 2.3e-5 | 0.0192 | 0.0050 |
| Router | 1.2e-5 | 0.0192 | 0.00023 |

The bank and router both improve on the unsteered baseline in 0-shot, but neither beats fixed steering overall. On GSM8K, bank and router are respectively 4.32 and 5.84 points below fixed steering; on ARC-Challenge they are 17.60 and 17.20 points below it. GSM-Plus also favors fixed steering by 3.80 and 5.00 points, though those direct comparisons are less precise. Treat the bank/router point estimates as evidence that the current route-selection system has not converted its larger search space into an accuracy advantage.

## What the router actually selected

The router had an 11-route action catalog, including `k=0` (no steering), paired with a discrete gain grid. The final benchmark did show more than one action; it did not use the catalog broadly.

| Dataset | Shots | `k=0` share | Most-used route | Most-used share | Unique routes |
|---|---:|---:|---|---:|---:|
| GSM8K | 0 | 44.4% | `[22]` | 51.6% | 4 |
| GSM8K | 2 | **96.8%** | `k=0` | 96.8% | 3 |
| GSM-Plus | 0 | 35.4% | `[22]` | 61.4% | 4 |
| GSM-Plus | 2 | **92.8%** | `k=0` | 92.8% | 3 |
| ARC-Challenge | 0 | 35.0% | `[22]` | 49.8% | 4 |
| ARC-Challenge | 2 | **81.6%** | `k=0` | 81.6% | 6 |

At 0-shot the router mostly chooses a one-layer route or abstains. At 2-shot it usually chooses no steering. It selects no four-layer action in these six test conditions. Gain selection is also concentrated: alpha is 1.0 on every GSM8K and GSM-Plus example and on 96.2% of ARC 0-shot examples. This is measurable action diversity, but not evidence that the policy identifies a reliably better action per question.

## Answer format and output length

The primary numeric-task result is lenient answer accuracy. The saved records also include strict accuracy, answer-marker accuracy, answer-extraction rate, generated tokens, and generation-cap frequency. These measures should be read together: the baseline frequently fails the expected final-answer format, especially at 0-shot, so a gain in answer accuracy can include improved answer production and extraction as well as improved mathematical solution behavior.

| Dataset, 0-shot | Arm | Strict accuracy | Lenient accuracy | Extraction rate | Mean output tokens | Hit token cap |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | Baseline | 5.99% | 46.47% | 11.4% | 235.1 | 75.9% |
|  | Fixed steering | 57.32% | 57.32% | 98.2% | 84.1 | 1.0% |
|  | Static bank | 49.43% | 52.99% | 89.5% | 112.8 | 8.3% |
|  | Router | 29.95% | 51.48% | 55.3% | 168.8 | 38.1% |
| GSM-Plus | Baseline | 5.60% | 31.60% | 14.4% | 231.2 | 74.0% |
|  | Fixed steering | 41.00% | 41.20% | 97.8% | 97.9 | 1.0% |
|  | Static bank | 34.40% | 37.40% | 84.2% | 131.4 | 13.6% |
|  | Router | 24.20% | 36.20% | 60.0% | 169.9 | 34.4% |

For ARC-Challenge, all four arms hit the 64-token output limit on every 2-shot question. That cap is a material limitation on the ARC 2-shot result and may truncate otherwise complete responses.

## Runtime and throughput

Generation latency includes tokenization, generation, stopping, and decoding within the timed generation call. It excludes dataset loading, answer scoring, and file writing. Timing coverage was 100% in the summary records. The table shows mean seconds per question and output tokens per second for 0-shot; compare only within these runs, since absolute speed depends on the GPU and runtime load.

| Dataset | Arm | Mean seconds / question | Output tokens / second |
|---|---|---:|---:|
| GSM8K | Baseline / Fixed / Bank / Router | 8.43 / 3.37 / 4.22 / 6.24 | 27.9 / 24.9 / 26.8 / 27.0 |
| GSM-Plus | Baseline / Fixed / Bank / Router | 8.89 / 4.21 / 5.30 / 6.81 | 26.0 / 23.3 / 24.8 / 25.0 |
| ARC-Challenge | Baseline / Fixed / Bank / Router | 2.40 / 2.61 / 2.11 / 2.09 | 26.0 / 23.2 / 24.8 / 24.6 |

For GSM8K and GSM-Plus, the fixed arm emits substantially fewer tokens and finishes faster than the baseline, with higher 0-shot accuracy. This is an end-to-end answer-generation result, not evidence that each generated token is processed faster. In ARC, where outputs are capped at 64 tokens, fixed steering is not faster than baseline.

## Training snapshot and artifact sizes

- Stage A evaluated 5,000 unique GSM8K training questions for rollouts. It retained 2,293 strict-positive completions from 1,495 questions; 592 questions yielded two distinct positives. The remaining 3,505 questions had no verified positive after the configured retry budget and supplied no positive target to the likelihood objective.
- Stage B contains 38 independently parameterized route experts across 14 candidate layer locations; the router sees an 11-route catalog including `k=0`.
- The saved router run used 400 teacher-scored question rows, recorded 51 route/gain action scores, and ran three epochs. Router checkpoint metadata contains 50 held-out validation question groups. The bank preference-validation set contains 19 matched pairs.
- Bank preference training recorded 503 matched correct/well-formatted-wrong pairs; the preference term is therefore available only on a subset of examples.
- The four `.pt` files total 35,263,656 bytes (33.6 MiB): fixed adapter, route bank, router, and route-teacher targets. The base Phi-2 weights are separate and are not included in this project artifact total.

## Results files

Each per-condition ZIP contains the prediction records and paired summaries used by the analysis. `ars_paper_assets.zip` contains the consolidated summary tables and the 12 rendered black/orange figures used by the paper notebook. The files are the source of truth; values in this document are rounded for readability.
