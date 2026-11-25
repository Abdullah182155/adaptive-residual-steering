# Evaluation protocol

The final evaluation compares four arms on matched questions and prompt conditions. GSM8K is the in-distribution benchmark. Separate notebooks measure mathematical transfer and multiple-choice transfer. Evaluation records contain per-question predictions and provenance so comparisons can be reconstructed.

## Arms

| Arm | Configuration |
|---|---|
| `baseline` | Frozen Phi-2 without steering |
| `steer_only` | Fixed-route adapter on `[12, 15, 18, 21]` with the selected calibrated gain |
| `static_bank` | One held-out-selected bank route and gain applied to all questions |
| `router` | Per-question route and gain selected by the trained router |

All arms use the same ordered test IDs, shot condition, prompt-template schedule, and decoding parameters within each comparison. Every arm uses the same trained artifact bundle. Demonstrations come from training data, never the evaluated test rows.

## In-distribution GSM8K

Notebook `02_gsm8k_in_distribution.ipynb` uses seed 42 and separates conditions into resumable slices:

| Slice | Shot settings | Question budget |
|---|---|---:|
| `zero_shot_all` | 0-shot | Entire official test split, documented as 1,319 questions |
| `fewshot_1_2` | 1-shot and 2-shot | Same seeded 500-question subset per condition |
| `fewshot_4_8` | 4-shot and 8-shot | Same seeded 500-question subset per condition |

The completed result bundles differ from that planned slice table: the archived GSM8K 0-shot and 2-shot records each contain 1,319 paired test questions. The saved results, not the planned caps above, define what was actually evaluated.

Zero-shot and 2-shot are the predeclared primary comparisons. The 1-, 4-, and 8-shot conditions are secondary shot-sensitivity results. Reports keep each shot condition separate.

## Mathematical transfer

Notebook `03_math_ood_evaluation.ipynb` runs one dataset per session, with up to 500 paired questions and both 0-shot and 2-shot:

- **GSM-Plus:** related GSM8K-style variation; useful for robustness to changed numbers and phrasing, with limited domain independence.
- **SVAMP:** arithmetic word problems with variation in linguistic structure and operation cues.
- **ASDiv:** broader collected arithmetic word-problem pool. The selected public release uses its `validation` split as an external evaluation pool, not as a canonical test split.

The actual source split, test IDs, and question count are saved with each result. The numeric scorer applies only to compatible numeric-answer datasets. Symbolic MATH-500 remains outside this protocol until a verified symbolic-equivalence grader is added.

## Multiple-choice transfer

Notebook `04_multiple_choice_transfer.ipynb` runs one dataset per session at 0-shot and 2-shot, with a maximum of 500 questions:

- **AQuA-RAT:** algebraic word problems in multiple-choice form.
- **ARC-Challenge:** science-question transfer to a distinct subject and answer format.

The actual paired count is determined by the selected dataset split and recorded in the result. The notebook takes demonstrations from the training split and never from the test split.

## Accuracy and paired inference

The evaluator preserves per-question outcomes and reports arm-level accuracy plus paired arm-to-arm comparisons. For each pair, summaries include wins, losses, ties, accuracy difference in percentage points, a paired bootstrap 95% interval, and McNemar's test. The pairwise record is the basis for the comparison; independently rounded arm percentages are not.

The evaluation also retains strict, marker, and lenient answer accuracy where the scorer supports them, answer-extraction and output-format rates, stop reasons, generation-limit frequency, arithmetic/equation diagnostics where available, guard trips, and nonfinite counts. Dataset-specific fields may be absent when they do not apply.

## Runtime and output quality

Each model-generation record can include generation latency, prompt tokens, generated tokens, and stop reason. The aggregate tables report timing coverage, mean/median/p90 generation latency, total generation time, questions per second, output tokens per second, and mean prompt/output lengths. CUDA timings synchronize around measurement; one unscored warm-up per arm and shot is excluded. Dataset loading, scoring, and file writes are excluded. Rows without timings remain in accuracy summaries and are excluded from throughput denominators; coverage is reported.

## Router diagnostics

For the router arm, the reports record observed unique routes, route frequencies, route entropy, most-used-route share, active layer-count (`k`) counts, per-layer counts, and gain counts. These are behavior diagnostics. Higher route count or entropy alone does not establish a better policy. Interpret them with paired accuracy, confidence intervals, and route cost.

## Completed result status

The completed saved results currently cover GSM8K (1,319 questions), GSM-Plus (500), and ARC-Challenge (500), at 0-shot and 2-shot. Each condition includes the four matched arms. The summary values, paired intervals, exact McNemar tests, answer-quality fields, timing, and run fingerprints are in [final_results.md](final_results.md). The raw result ZIP files are local project artifacts and are not committed to Git.

No saved results are present for SVAMP, ASDiv, MATH-500, or AQuA-RAT. Their support in evaluation code or notebooks does not count as an executed benchmark. Do not label them as completed.
