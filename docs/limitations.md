# Limitations and claim boundaries

This document records interpretation rules for the final study. It does not contain benchmark results; those will be added after the pending runs finish.

## What the system design does not establish

- A larger bank or a more diverse route catalog does not by itself show that selected routes improve answer accuracy.
- A router's teacher agreement, loss, or route entropy does not establish that its test-time decisions help. Paired answer outcomes are required.
- A route that is best on a screening split can still lose on a held-out benchmark. Static selection and router validation must remain separate from test evaluation.
- Positive-only RFT data does not teach the adapter how to correct every failure. It trains the adapter to increase likelihood of collected correct completions; the bank preference term adds same-prompt correct/wrong comparisons only where valid pairs exist.
- GSM-Plus is related to GSM8K and should not be described as a fully independent domain.
- The selected ASDiv validation pool is an external evaluation pool, not a canonical test split.
- Numeric answer matching is not a valid grader for symbolic-answer benchmarks such as MATH-500 without a verified symbolic-equivalence evaluator.
- Runtime performance depends on hardware and load. Latency and token throughput must include timing coverage and use the same measurement definition across arms.

## Claims for the paper

Use paired deltas, confidence intervals, disagreement counts, and the exact McNemar result for claims about differences. Report zero-shot and few-shot conditions separately. State the evaluated sample count, split, seed, prompts, artifact fingerprint, and code revision. Treat a non-significant or wide interval as uncertain, even if the point estimate favors one arm.

Keep numerical conclusions and causal explanations blank until the corresponding saved evaluation records have been checked. A successful training status means the pipeline completed its gates; it does not mean that steering, the bank, or the router improved the benchmark.
