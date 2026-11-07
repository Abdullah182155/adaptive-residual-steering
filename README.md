# Adaptive Residual Steering (ARS)

**Lightweight residual adapters that steer a frozen LLM toward structured mathematical reasoning — no fine-tuning, no RL from human feedback.**

ARS injects rank-16 causal adapters (SteerNet V5) at selected transformer layers of `microsoft/phi-2` (2.7B) and trains them to improve chain-of-thought accuracy on GSM8K. A learned router then selects per-question layer subsets at inference time.

## Method

```
                        ┌──────────┐
  prompt ──► Phi-2 ──►  │ SteerNet │ ──► ... ──► answer
             frozen     │ adapters │
                        └──────────┘
                     layers selected by
                     JointLayerRouter
```

**Phase 1 (S1)** — Fixed-route SteerNet training:
- 10 candidate layers: `[8, 11, 13, 15, 17, 19, 20, 21, 22, 24]`
- Typed NLL loss with per-category weighting (format, prose, operator, operand, result, final)
- Learned Kendall uncertainty balancing across NLL, KL-divergence, and L1 objectives
- On-policy data collection with gold-answer fallback for hard questions
- Prompt shielding preserves few-shot demonstrations from steering interference

**Phase 2 (S2)** — Per-question adaptive routing:
- JointLayerRouter with two-pass cross-attention and learned synergy matrix
- RLOO policy gradient with Cooperative Plackett-Luce subset sampling
- Selects K layers (2-8) per question based on probe hidden states
- Synergy-aware greedy inference matches training-time selection

## Architecture

### SteerNet V5

Each adapter is a rank-16 bottleneck with causal depthwise convolution:

```
h ──► LayerNorm ──► W_down (D→16) ──► CausalDWConv1d(k=4) ──► GELU ──► W_up (16→D)
                                                                  │
                                                           mag_head ──► sigmoid ──► magnitude
```

- **Relative amplitude**: `delta = rho_max / sqrt(K) * magnitude * ||h_centered|| * direction`
- **Smooth cap envelope**: `C_l = kappa * N_l` (calibrated per-layer norm bounds)
- **Position-0 shielding**: no steering on the attention-sink token

### JointLayerRouter

```
probe_hidden ──► pool ──► [scalars, depth, semantic] ──► Transformer(2-pass)
                              │                                │
                              ▼                                ▼
                         k_head ──► K distribution      utility_head ──► layer scores
                                                               │
                                                        synergy_proj ──► S_ij cooperation matrix
```

## Results

Preliminary results on GSM8K (1000 training examples, 5 epochs, 50 test):

| Condition | Baseline | ARS (Fixed) | Delta |
|-----------|----------|-------------|-------|
| 0-shot    | 42.0%    | 52.0%       | **+10.0 pp** |
| 2-shot    | 56.0%    | 64.0%       | **+8.0 pp** |
| 4-shot    | 58.0%    | 60.0%       | +2.0 pp |

Format compliance (0-shot): 16% baseline → 76% with ARS.

## Project Structure

```
src/
├── configs/config.py          # RSCConfig dataclass with all hyperparameters
├── data/
│   ├── dataset.py             # GSM8K loading, tokenization, multi-context splits
│   └── prompts.py             # Unified prompt builder with shield masks
├── models/wrapper.py          # Phi2WithRSC: hooks, routing override, generation
├── steering/steernet.py       # SteerNet V5 with CausalDWConv1d
├── routing/
│   ├── router.py              # JointLayerRouter with synergy-aware inference
│   └── rloo.py                # RLOO policy gradient with Plackett-Luce sampling
├── training/
│   ├── trainer.py             # Phase 1 training loop + on-policy collection
│   ├── router_trainer.py      # Phase 2 router training + checkpoint loading
│   └── losses.py              # Typed NLL, focal weighting, Kendall balancing
├── evaluation/
│   ├── benchmark.py           # Multi-arm benchmark (baseline/fixed/router)
│   ├── answers.py             # Numeric answer extraction and scoring
│   └── diagnostics.py         # Paired plots, training curves, generation viewer
└── deploy/calibrate.py        # Deployment contract calibration

scripts/
├── run_ars_pipeline.py        # Main entry: S1 → S2 → S5 → bench
└── calibrate_norms.py         # Per-layer norm calibration for cap envelope

notebooks/
├── kaggle_ars_runner.ipynb    # End-to-end Kaggle execution notebook
└── ood_evaluation.ipynb       # Out-of-distribution evaluation (SVAMP, ASDiv, etc.)

tests/                         # 22 unit tests covering core components
```

## Quick Start

```bash
# 1. Install dependencies
pip install transformers>=4.36.0 datasets>=2.16.0 accelerate>=0.25.0 tqdm

# 2. Calibrate per-layer norms
python scripts/calibrate_norms.py --output norms_calibration.json \
    --candidate_layers 8 11 13 15 17 19 20 21 22 24

# 3. Run full pipeline (S1 training → S2 router → S5 calibration → benchmark)
python scripts/run_ars_pipeline.py --stage all --steernet_arch v5 \
    --s1_epochs 5 --s1_lr 5e-4 --s1_train_n 7473 \
    --s_fixed 8,11,13,15,17,19,20,21,22,24

# 4. Run tests
pytest tests/ -q
```

For Kaggle execution with GPU, use `notebooks/kaggle_ars_runner.ipynb`.

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rho_max` | 0.06 | Maximum relative steering amplitude |
| `steer_kappa` | 0.04 | Cap envelope coefficient (C_l = kappa * N_l) |
| `lora_rank` | 16 | SteerNet bottleneck rank |
| `phase1_epochs` | 5 | S1 training epochs |
| `kl_weight` | 2.0 | KL divergence penalty (preserves base model on easy inputs) |
| `focal_gamma` | 1.0 | Focal loss concentration on hard tokens |
| `router_max_k` | 8 | Maximum layers the router can select |
| `phase2_epochs` | 3 | S2 router training epochs |
| `on_policy_start_epoch` | 2 | First epoch for on-policy data collection |

## Design Principles

1. **Zero heuristics** — No manual thresholds, no static layer tables. Everything is learned or calibrated.
2. **Frozen backbone** — The base Phi-2 model is never modified. All steering is additive through the residual stream.
3. **Relative amplitude** — Steering magnitude scales with the local hidden-state norm, making it resolution-invariant across layers.
4. **Prompt shielding** — Few-shot demonstrations are protected from steering to prevent format regression in multi-shot settings.

## Citation

This is an active research project. If you use this work, please cite the repository.

## License

MIT
