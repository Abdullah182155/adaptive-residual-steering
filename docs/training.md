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

1. **Steering bank.** Build positive RFT train/validation rows from rollout shards. Train all candidate single-layer adapters, screen candidate cross-depth pairs, then grow selected pairs into triples and quads. The route curriculum prunes nearby same-band combinations and carries an explicit empty route for no steering. Save the bank checkpoint and its route catalog.
2. **Router.** Score the route catalog on training examples. For each question, build prompt-only router features and teacher labels from the measured action scores. Train a route-value model to predict those scores. Completion tokens are used only to measure training targets, never as router input. Validation selects the best checkpoint.
3. **Paired benchmark.** Compare baseline, fixed V2, and router arms on the same deterministic GSM8K subset and shot conditions. Inspect paired deltas and intervals alongside route counts, layer counts, active-k distribution, and most-used-route share. High teacher diversity does not imply the learned router will use diverse actions.

Calibration outputs, the V2 checkpoint, rollout shards, steering bank, router checkpoint, and benchmark files must all refer to compatible layer routes and metadata. The notebook prints the resolved paths and records phase failures; a failed prerequisite should be treated as a failed run phase, not as a valid comparison.

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
