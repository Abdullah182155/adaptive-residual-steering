# Historical alternate design: prompt-conditioned residual steering

> **Archived design.** This document describes a separate prototype and is not the architecture or training contract used by the maintained `notebooks` workflow. See [`../architecture.md`](../architecture.md) for the current system.

## Purpose

This document describes the fixed-route steering architecture used for the next
training run. It replaces the ambiguous experiment label `v2` with the stable
architecture identifier `ars.prompt_conditioned_residual` and a semantic
architecture version (`1.0.0`). The old `v2` name remains accepted only as a
compatibility alias for existing checkpoints and notebooks.

## Architecture

The Phi-2 backbone remains frozen. The adapter is attached to transformer
layers `[12, 15, 18, 21]`; these are backbone locations, not the adapter's own
neural-network layers.

At each selected backbone layer, the adapter:

1. Projects the hidden state from the backbone width into a rank-32 latent space.
2. Refines that latent representation with a residual MLP block.
3. Projects the result back to the backbone width to produce a steering update.
4. Predicts one bounded strength per example from the masked mean of prompt
   latents. Each selected layer has its own strength head. Its prediction is
   reused during autoregressive decoding for that example.
5. Applies the p99-calibrated residual norm cap with `rho_max=0.10`.

The strength is multiplied into the residual before the cap:

```text
latent = GELU(W_down @ hidden)
refined = latent + W_residual_out(GELU(W_residual_in(latent)))
strength = strength_max * sigmoid(W_strength @ mean(prompt_latent) + b_strength)
delta = norm_cap(strength * W_up @ refined, rho_max * calibrated_p99_norm)
```

The prompt mask excludes completion tokens during RFT, so the strength head does
not read the gold answer. At inference the prompt prefill produces one strength
per selected layer and generation reuses those values for every decode step.
The head receives the pre-adapter hidden state at its layer, including updates
from earlier routed layers. The strength-teacher feature pass uses the same
state, avoiding a mismatch between training features and inference inputs.

## Training contract

- The backbone is frozen; only the adapter and strength heads are optimized.
- Positive rollout completions use token-mean causal NLL.
- Training rows are split by source question, with 0-shot receiving the largest
  sampling weight and 2-shot/4-shot examples retained for generalization.
- After RFT, a separate training-only teacher sweep compares a bounded strength
  grid using greedy exact-answer correctness. For questions with at least one
  correct candidate, the target is the correct strength closest to neutral
  strength `1.0`; questions with no correct candidate retain neutral strength.
  Per-layer heads are fitted in transformer order and the inputs for later
  layers are refreshed after upstream heads are updated.
- The model is initialized from scratch. No old adapter, bank, or router
  checkpoint is loaded.
- RFT epoch selection uses paired held-out generation, prioritizing 0-shot and
  rejecting material 2-shot regressions.
- The first benchmark is a paired 200-question GSM8K check at 0-shot and 2-shot.
  It reports baseline, learned prompt strength, and a fixed-strength `1.0`
  reference on the same questions.

## Artifact identity

New checkpoints record:

- `architecture_id`: `ars.prompt_conditioned_residual`
- `architecture_version`: `1.0.0`
- `steering_rank`: `32`
- `strength_conditioning`: `masked_prompt_mean`
- `strength_scope`: `per_example_per_layer`
- the fixed route, p99 norm statistic, and residual cap

Human-facing stage names use `rollout_quality_probe`; `gate0` remains an
accepted CLI and artifact alias for older runs. Use
`--steering_architecture prompt_conditioned_residual` for new runs. The older
`v2` architecture label remains a compatibility option for legacy checkpoints;
it does not refer to this architecture version.

## Limits of the first implementation

Exact-answer outcomes on a sampled training subset provide noisy strength
targets. The teacher is a training signal, not an estimate of generalization;
only the paired validation and test comparisons can establish whether learned
strength helps. A later bank/router stage must explicitly load this
architecture identifier and rank; it must not assume the legacy
`steernet_v2` state-dict layout.
