# ARS final-study notebook suite

This suite separates fresh training, in-distribution testing, transfer testing, diagnostics, and figure generation. Notebook 01a starts from scratch; later notebooks consume its verified handoff bundle.

## Run order and inputs

| Notebook | Purpose | Required Kaggle input | Main output |
|---|---|---|---|
| `01a_train_fixed_and_bank.ipynb` | Fresh fixed-route low-rank adapter, rollout, and steering-bank training | No previous artifacts; internet and CUDA GPU(s) | `ars_bank_ready_bundle.zip` |
| `01b_train_router_and_package.ipynb` | Resume from the bank-ready bundle, train only the router, and package all final artifacts | `ars_bank_ready_bundle.zip` from Notebook 01a; same pinned code revision and CUDA GPU | `ars_training_bundle.zip` |
| `02_gsm8k_in_distribution.ipynb` | Four-arm GSM8K comparison; 0-shot uses all 1,319 test questions; 1/2/4/8-shot use 500 each | `ars_training_bundle.zip` | `ars_gsm8k_results.zip` |
| `03_math_ood_evaluation.ipynb` | Four-arm 0/2-shot evaluation on GSM-Plus, SVAMP, and MultiArith, up to 500 questions per dataset | `ars_training_bundle.zip` | `ars_math_ood_results.zip` |
| `04_multiple_choice_transfer.ipynb` | Four-arm 0/2-shot evaluation on ARC-Challenge or MMLU-Pro, one dataset per session | `ars_training_bundle.zip` | `ars_mcq_results.zip` |
| `05_router_ablations.ipynb` | Read-only route diversity, active-layer-count, gain, and paired-accuracy diagnostics | Any completed result bundles from Notebooks 2-4 | JSON and CSV diagnostics |
| `06_paper_figures.ipynb` | CPU-only presentation figures and data tables from saved results | Any completed result bundles from Notebooks 2-4 | `ars_paper_assets.zip` |

Notebook 01a trains architecture `ars.fixed_route_low_rank_residual` (version `1.0.0`) from fresh GSM8K rollouts and fresh p99 calibration. It does not load old adapter, calibration, rollout, bank, or router files. Notebook 01b consumes only the verified `ars_bank_ready_bundle.zip` produced by 01a. The separate prompt-conditioned architecture is not selected by this fixed-route bank workflow.

The bank stage creates a separately parameterized adapter for every retained route. It trains all single-layer experts first, screens pair/triple/quad routes with a three-layer minimum spacing, keeps a small set of two-layer-spacing controls, initializes each larger route from its best measured proper subset, and fine-tunes that route without changing any other expert. The final bank retains up to 8 pairs, 6 triples, and 4 quads plus the singleton experts. Router supervision uses a compact, validation-ranked catalog: k=0, up to 4 diverse singleton routes, and up to 2 routes at each larger cardinality. The fixed adapter remains a distinct benchmark arm and does not seed bank weights.

After Notebook 01a completes, attach its `ars_bank_ready_bundle.zip` to Notebook 01b. Attach the resulting `ars_training_bundle.zip` to Notebooks 2-4, then attach completed result bundles to Notebooks 5-6. Those notebooks merge earlier result bundles before writing a new bundle, so a later session can add its slice without losing earlier conditions.

## Important evaluation boundaries

- All arms use the same test IDs within each condition. Shot counts are recorded separately.
- Notebook 2's slices are `zero_shot_all`, `fewshot_1_2`, and `fewshot_4_8`; run one slice per Kaggle session and carry the result zip forward.
- Notebook 3 uses numeric-answer datasets compatible with the existing paired scorer. Its `N_PER_DATASET` is a maximum; each record reports the actual paired question count. MATH-500 is excluded until a symbolic-answer equivalence grader is implemented; converting its symbolic answers to floats would invalidate the score.
- Notebook 4 uses ARC-Challenge and MMLU-Pro. Demonstrations come only from ARC train or MMLU-Pro validation, never from the test split.
- Dataset references: [GSM-Plus](https://huggingface.co/datasets/qintongli/GSM-Plus), [SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP), [MultiArith](https://huggingface.co/datasets/ChilleD/MultiArith), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), and [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro). GSM-Plus is used only for evaluation, consistent with its dataset card.
- Notebook 5 reports route diversity as a diagnostic. More unique routes alone do not demonstrate a better router; compare accuracy, paired intervals, and route cost.
- Notebook 6 creates figures only from saved records. Missing measurements remain missing and are listed in the manifest.

## Runtime notes

Notebooks 1-4 need a Kaggle GPU and internet access for model/dataset downloads. Notebooks 5-6 do not load the model and need no GPU. For long benchmarks, select one notebook slice per Kaggle session and preserve the generated output zip before starting the next slice.

- The training pipeline is deliberately split at the saved steering-bank checkpoint: if the router session needs a separate Kaggle session, attach `ars_bank_ready_bundle.zip` and run Notebook 01b only. This avoids repeating rollouts, fixed-adapter training, and bank training.
