# References and source links

This list separates the benchmark sources from papers that motivate related ideas. Citing a related paper does not mean ARS reproduces that paper's method.

## Evaluated datasets

1. **GSM8K** — OpenAI's grade-school math word-problem benchmark. Dataset card: [Hugging Face](https://huggingface.co/datasets/openai/gsm8k). Paper: Cobbe et al., “Training Verifiers to Solve Math Word Problems,” [arXiv:2110.14168](https://arxiv.org/abs/2110.14168).
2. **GSM-Plus** — Perturbation-based robustness benchmark for grade-school math problems. Dataset card: [Hugging Face](https://huggingface.co/datasets/qintongli/GSM-Plus). Paper: Li et al., “GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers,” [ACL Anthology](https://aclanthology.org/2024.acl-long.163/), [arXiv:2402.19255](https://arxiv.org/abs/2402.19255).
3. **ARC-Challenge** — AI2 Reasoning Challenge multiple-choice science questions. Dataset card: [Hugging Face](https://huggingface.co/datasets/allenai/ai2_arc). Paper: Clark et al., “Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge,” [arXiv:1803.05457](https://arxiv.org/abs/1803.05457).

## Base model

- **Microsoft Phi-2** — the frozen backbone used in this study. Model card and usage terms: [Hugging Face](https://huggingface.co/microsoft/phi-2). Microsoft overview: [Phi-2: The surprising power of small language models](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/).

## Related methodological work

- Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). Related low-rank parameter-efficient adaptation; ARS uses its own nonlinear residual adapter and is not standard LoRA.
- Turner et al., “Activation Addition: Steering Language Models Without Optimization,” [arXiv:2308.10248](https://arxiv.org/abs/2308.10248). Related activation-space intervention work; ARS trains a bounded residual adapter instead of applying a fixed activation vector.
- Zou et al., “Representation Engineering: A Top-Down Approach to AI Transparency,” [arXiv:2310.01405](https://arxiv.org/abs/2310.01405). Related work on interpreting and intervening on representations; it is context for the research question, not a direct implementation dependency.

## Project artifacts and code

- Source repository: [Adaptive Residual Steering](https://github.com/Abdullah182155/adaptive-residual-steering).
- Trained adapter release: [Hugging Face model repository](https://huggingface.co/Abdullah182155/Adaptive_Residual_Steering). The project maintainer reports that the release files have been uploaded. The local training bundle manifest records the canonical SHA-256 values; see [model release notes](model_release.md).
- Exact training code commit: `5bd78d89a79e375d8330edc04066c05a4d99837c`.
- Exact evaluation code commit: `f0d102c21af3272d07bf8be74156eba2bdd8c8d6`.
- Evaluation artifact fingerprint: `47a758610030`.

## Reporting notes

Dataset license and terms must be followed independently of the model's license. The Phi-2 base weights are not included in the ARS adapter release. Report the source split, question count, shot count, sample seed, scoring rule, and model artifact fingerprint with every result.
