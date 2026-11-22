# Training and Evaluation Workflow

This guide follows the current command-line and Kaggle execution paths. Per-run values belong in the notebook configuration or CLI arguments; historical experiment settings in audit reports are not defaults.

## Shared rules

- Freeze every Phi-2 backbone parameter. Only steering and routing modules are trainable.
- Keep Phase 1 learning rate at `1.5e-4`; `RSCConfig.validate()` enforces the lock.
- Phase 1 RFT trains on strictly verified positive completions with token-mean NLL. Prompt and padding labels are masked from the loss.
- Use paired evaluation: all arms receive the same questions, prompts, decoding parameters, and test seed.
- Keep run outputs separate from source. Each run should retain its configuration, shard statistics, checkpoints, paired report, and final status.

## Session 1: fixed-route RFT

The Session 1 workflow is orchestrated by `notebooks/kaggle_ars_runner.ipynb` and `scripts/run_rft_pipeline.py`:

1. **Equivalence probe** compares batch-1 and batched generation. A mismatch is diagnostic for the Session 1 path unless strict equivalence was explicitly requested; the notebook's configured inference path must match the benchmark path.
2. **Gate 0** compares sampled pass@k with greedy decoding. Production quality checks are reported separately from smoke-mode diagnostics.
3. **Collection** samples GSM8K training questions and verifies generated completions. Two workers can collect disjoint shards on separate GPUs. The CLI enforces a minimum verified-positive yield unless `--allow_low_yield` is passed; the selected policy is recorded in shard stats.
4. **Training** combines positive rows from the shard files and optimizes the fixed route `[17, 20, 23, 26]`. The backbone stays frozen; the magnitude head is disabled, giving unit magnitude. Checkpoint selection uses the paired validation accuracy delta; completion-length ratio is recorded as a diagnostic.
5. **Evaluation** compares baseline and the isolated `steer_only` arm with paired per-question statistics. Calibration and evaluation are skipped if no valid trained checkpoint is available.

The CLI supports explicit `equivalence`, `gate0`, `collect`, and `train` stages as well as `all`. Use `--n_questions` for probe and collection sizes, `--skip_collection --rollout_dir PATH` to reuse existing shards, and `--allow_partial_rollouts` only when intentionally training from available nonempty shards. Preserve the shard JSONL files and their stats if the run may be resumed.

## Session 2: fixed-route V2

Session 2 changes the adapter architecture while retaining the RFT objective and frozen backbone. It uses route `[12, 15, 18, 21]`, rank 16, `rho_max=0.10`, no magnitude head, no convolution, and p99 norm calibration. Reusing rollout shards avoids recollecting data; the training CLI validates shard availability and calibration before training.

Session 1 and Session 2 checkpoints are not interchangeable: their adapter topology, route metadata, rank, and calibration contracts differ. Keep the checkpoint metadata and corresponding norm file together.

## Session 3: adaptive steering bank and router

The Session 3 notebook runs `scripts/run_session3_router.py` in three stages:

1. **Steering bank.** Split rollout questions before screening. Train a separate expert for each singleton route from verified positive RFT rows. Build pair, triple, and quad candidates with a three-layer minimum separation plus a bounded set of two-layer controls. Initialize each larger expert from its best measured trained proper-subset experts, then optimize it with its own parameters while freezing every other expert. Use token-mean NLL and a weighted pairwise margin between each positive completion and a same-prompt, well-formatted wrong-answer completion. Screen routes on held-out questions; use positive-completion NLL when matched preference evidence is unavailable. Retain routes across depth patterns, preserve `k=0` as a no-intervention control, and validate the checkpoint catalog against its exact expert weights and norm references before use.
2. **Static-action selection.** Use its own held-out question partition. Matched preference scores shortlist route/gain candidates; when enough questions are available, a disjoint subset is used for greedy 0-shot/2-shot accuracy to select the final fixed bank route and gain. These questions are not part of the benchmark test subset.
3. **Router.** Score catalog actions on the router-training partition using matched positive/wrong-answer preferences where available, and positive-completion NLL for remaining questions, with a small route cost. Router input is a pooled representation of question tokens only, plus normalized shot count and question length; completion tokens are never router features. Validation compares greedy answers against the selected static-bank action and the unsteered baseline on separate held-out questions. Checkpoint selection uses configured weights of 0.7 for 0-shot, 0.2 for 2-shot, and 0.1 for 4-shot. Weights are renormalized over shot settings actually evaluated; the current 0/2-shot pilot therefore uses 0.7778 × 0-shot plus 0.2222 × 2-shot. The router receives a small discrete gain grid (`[1.0, 1.5, 2.0]`) and can select a question-conditioned gain; neither gain adaptation nor route diversity is guaranteed to improve accuracy without benchmark evidence.
4. **Paired benchmark.** Require baseline, fixed V2, static bank, and router to all complete on the same deterministic GSM8K subset and shot conditions. Optional numeric OOD suites use deterministic seeded samples and report status separately. Inspect paired deltas, intervals, route counts, active-k distribution, layer counts, and the most-used-route share. High teacher diversity does not imply the learned router will use diverse actions.

Calibration outputs, the V2 checkpoint, rollout shards, steering bank, router checkpoint, and benchmark files must all refer to compatible layer routes and metadata. Training, bank screening, static-action selection, router validation, and test questions are split by question. The notebook verifies the exact four-arm list and configured question count before marking the GSM8K benchmark complete. It records OOD suite failures separately; a missing OOD result must not be described as a completed OOD evaluation.

## Evaluation interpretation

- Accuracy on a sampled subset is a preliminary estimate for that deterministic subset, not the full GSM8K test-set result.
- Prefer paired deltas over subtracting independently rounded arm accuracies.
- Report confidence intervals and the paired disagreement counts. A positive point estimate with a wide interval is uncertain.
- Report 0-shot and 2-shot separately. An improvement in one condition does not establish improvement in the other.
- A router that beats baseline but loses to fixed steering has not established the value of adaptive routing.
- Check output format, guard trips, invalid answers, and route collapse in addition to lenient accuracy.

## Verification

Run the unit and contract suite from the repository root:

```powershell
python -m pytest tests/ -x -q
```

For a Kaggle run, retain the final phase-status object and the JSON benchmark report with the corresponding per-question generation files. These artifacts are needed to reproduce and interpret the comparison.
