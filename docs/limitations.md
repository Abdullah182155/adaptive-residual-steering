# Limitations and claim boundaries

This note records what the completed runs establish, what remains uncertain, and the practical resource limits observed during the project. The numerical results are summarized in [final_results.md](final_results.md); this document focuses on interpretation.

## Empirical scope

- The available final evaluation covers three sets: the 1,319-question GSM8K test split, a 500-question GSM-Plus test sample, and a seed-42 sample of 500 ARC-Challenge test questions. Each was evaluated at 0-shot and 2-shot with four matched arms.
- No result bundles are present for SVAMP, ASDiv, MATH-500, or AQuA-RAT. Do not describe those datasets as evaluated.
- These are three benchmarks, not a broad survey of reasoning domains. GSM-Plus is closely related to GSM8K; ARC-Challenge adds subject and answer-format transfer but is not a mathematical reasoning test.
- The 0-shot results are statistically stronger than the 2-shot results. The study does not establish a reliable few-shot gain.
- The test sets are finite samples. The reported paired confidence intervals quantify sampling uncertainty under the saved evaluation procedure; they do not cover all training-seed, prompt, model, or platform variation.

## Answer scoring and generation

- GSM8K and GSM-Plus use numeric answer parsing with lenient accuracy as the primary metric. The baseline has low answer extraction and strict-format rates and often reaches the generation cap. Consequently, some of the fixed adapter's gain may come from producing shorter, more extractable final answers, in addition to changes in solution behavior. The records do not isolate these mechanisms causally.
- ARC-Challenge is scored by exact multiple-choice answer. Every arm reached the 64-token output cap on all 500 ARC 2-shot questions. This censors those outputs and limits interpretation of the ARC 2-shot comparison.
- Greedy benchmark accuracy measures one decoding outcome per question; it does not estimate pass@k or robustness across sampling seeds.
- The current records do not include a human audit of every extracted answer or a blinded qualitative error analysis.

## Bank and router

- The bank has 38 independently parameterized route experts, but the router sees only an 11-route action catalog including no-op. The search is curated, not exhaustive.
- The router's observed benchmark use is concentrated. At 2-shot, no-op is selected on 81.6%–96.8% of questions, depending on the dataset. At 0-shot, four routes are observed per set; the singleton `[22]` is the most frequent route. No four-layer action is selected in the evaluated conditions.
- More observed routes do not prove better question-level decisions. The router and static-bank point estimates improve over baseline at 0-shot, but both remain below fixed steering on every evaluated dataset in point accuracy. The current evidence does not show that per-question routing adds value over the fixed adapter.
- The bank's learned experts and router are trained from a finite GSM8K-derived dataset and teacher-score process. Preference examples are sparse relative to positive NLL examples, and the bank validation preference set contains 19 matched pairs. These limits may affect the reliability of route ranking; the present run does not establish which one is the dominant cause.
- Router training used up to 400 teacher-question IDs according to the resume profile, but the saved Stage B manifest and source outputs should be treated as the authoritative record. Selection may be sensitive to the training/validation sample and route catalog.

## Training data and generalization

- Stage A collected rollouts from 5,000 unique GSM8K training questions. A strict-positive completion was obtained for 1,495 questions; 3,505 remained unresolved after the configured retry budget. Those unresolved cases remain in the collection audit but do not provide a positive target for the positive-completion likelihood objective.
- The 0/2/4-shot collection mixture is weighted toward zero-shot, but it is still a single dataset and one base model. No claim is made that the learned adapter generalizes to arbitrary prompts or model families.
- The 2-shot transfer examples come from GSM8K training data. Results should be described as cross-dataset few-shot context under this protocol, not as an independent demonstration set drawn from each target benchmark.

## Compute and resource limits

- The recorded training environment exposed two Tesla T4 GPUs with 15 GiB each. Stage B ran gain calibration and bank training concurrently when two GPUs were available; the resumed router stage used one visible GPU.
- Bank training completed, but route-teacher scoring was expensive. A saved run required roughly 36,253 forward evaluations and about two hours for that scoring pass. An earlier Stage B session exceeded the 12-hour notebook limit and was resumed from saved bank/calibration/rollout artifacts; the final bundle records a `router_only_resume` profile.
- The notebooks were designed around GPU-backed hosted runtimes. There is no benchmark on CPU-only hardware or a production serving stack. Absolute latency numbers are not portable across GPU models, software versions, concurrency, or provider load.
- Peak host RAM and GPU VRAM were not logged in the saved evidence. State the GPU model and memory capacity available, but do not invent peak memory or claim that memory was not a bottleneck.
- Inference metrics time the generation call and exclude dataset loading, scoring, and file output. Token throughput is not a measure of end-to-end application throughput.

## Reproducibility and release

- The final training bundle identifies Phi-2, seed 42, training/evaluation code commits, artifact checksums, selected route action, and route catalog. The evaluation bundles use a shared artifact fingerprint and paired question IDs within each condition.
- Raw archives and generated weights are excluded from Git. The repository contains a checked summary in `final_results.md`; preserve the source ZIP files and model bundle separately for reruns.
- The published adapter files are not a standalone language model. Inference requires the matching Phi-2 base model, repository code at the recorded commit, calibration files, and the correct route-selection metadata.
- The project maintainer reports that the trained files have been uploaded to the [Hugging Face model repository](https://huggingface.co/Abdullah182155/Adaptive_Residual_Steering). This workspace records the local Stage B manifest checksums; remote file-level checksums were not independently inspected here.

## Appropriate conclusion

The defensible conclusion is narrow: on the three evaluated datasets, the fixed residual adapter improves zero-shot answer accuracy over unmodified Phi-2, with paired gains that survive the reported multiple-comparison correction. The current bank and router do not surpass the fixed adapter, and the 2-shot comparisons do not establish a reliable gain. These findings justify a careful project report and a clear account of the failed adaptive-policy hypothesis; they do not establish a general improvement in LLM reasoning or readiness for a peer-reviewed claim without further replication and broader evaluation.
