# Training Methodology

This document describes the complete training pipeline for Adaptive Residual Steering (ARS), covering data preparation, loss functions, optimization schedules, and both training phases. All formulas reflect the implementation in `src/training/losses.py`, `src/training/trainer.py`, `src/training/router_trainer.py`, and `src/routing/rloo.py`.

---

## Table of Contents

1. [Training Overview](#1-training-overview)
2. [Phase 1: Fixed-Route SteerNet Training](#2-phase-1-fixed-route-steernet-training)
3. [On-Policy Data Collection](#3-on-policy-data-collection)
4. [Phase 2: RLOO Router Training](#4-phase-2-rloo-router-training)
5. [Loss Function Details](#5-loss-function-details)

---

## 1. Training Overview

ARS training proceeds in two sequential phases that together produce a steered language model with adaptive per-question routing:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ARS Training Pipeline                        │
│                                                                     │
│  ┌───────────────┐    ┌────────────────┐    ┌───────────────────┐  │
│  │  Norm Calib.   │───►│   Phase 1 (S1)  │───►│   Phase 2 (S2)    │  │
│  │  per-layer N_l │    │   SteerNet V5   │    │   JointLayerRouter│  │
│  └───────────────┘    │   Fixed route   │    │   RLOO + PL       │  │
│                        │   5 epochs      │    │   3 epochs        │  │
│                        │   + on-policy   │    │                   │  │
│                        └────────────────┘    └───────────────────┘  │
│                                                                     │
│  Frozen backbone: microsoft/phi-2 (2.7B) — never modified          │
└─────────────────────────────────────────────────────────────────────┘
```

**Phase 1 (S1)** trains the SteerNet V5 residual adapters on a fixed set of 10 candidate layers using a typed reasoning loss. The backbone remains frozen; only adapter parameters and learned loss weights are optimized. Starting from epoch 2, on-policy data collection augments the training set with the model's own correct completions.

**Phase 2 (S2)** freezes the trained SteerNet adapters and trains a JointLayerRouter to select per-question layer subsets. The router is optimized with RLOO (Leave-One-Out REINFORCE) policy gradients using Cooperative Plackett-Luce subset sampling.

### Data

- **Dataset**: GSM8K official train split (7,473 examples by default)
- **Validation split**: 8% held out (`val_split = 0.08`)
- **Tokenizer**: Phi-2 tokenizer with `max_tok_len = 512`
- **Multi-context mixture**: stratified shot distribution `[0.50, 0.10, 0.20, 0.05, 0.15]` for 0/1/2/3/4-shot prompts
- **Prompt template**: unified format with optional preface and V5 exemplars (`v5_demo_frac = 0.90`)

### Hardware Configuration

- Mixed precision (FP16/FP32) with `GradScaler` on CUDA
- Gradient checkpointing on the frozen backbone (saves ~30–40% activation VRAM)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation OOM

---

## 2. Phase 1: Fixed-Route SteerNet Training

Phase 1 optimizes the SteerNet V5 adapters at 10 fixed transformer layers while the Phi-2 backbone remains frozen.

### 2.1 Architecture Recap

Each SteerNet V5 adapter is a rank-16 bottleneck:

```
h ──► LayerNorm ──► W_down (D→16) ──► CausalDWConv1d(k=4) ──► GELU ──► W_up (16→D)
                                                                  │
                                                           mag_head ──► sigmoid ──► magnitude
```

The steering correction at layer $l$ is computed in **relative amplitude mode**:

$$\delta_l = \frac{\rho_{\max}}{\sqrt{K}} \cdot m_l \cdot \|h_l^{\text{centered}}\| \cdot \hat{d}_l$$

where:
- $\rho_{\max} = 0.06$ is the maximum relative amplitude
- $K$ is the number of active layers (10 during Phase 1)
- $m_l = \sigma(\text{mag\_head}(x_l)) \in [0, 1]$ is the learned per-token magnitude
- $h_l^{\text{centered}} = h_l - \bar{h}_l$ is the mean-centered hidden state
- $\hat{d}_l = d_l / \|d_l\|$ is the unit direction from `W_up`

A **smooth cap envelope** prevents any single layer's correction from growing unbounded:

$$\delta_l' = \delta_l \cdot \frac{C_l}{\sqrt{\|\delta_l\|^2 + C_l^2}}, \quad C_l = \kappa \cdot N_l$$

where $\kappa = 0.04$ and $N_l$ is the calibrated median hidden-state norm at layer $l$.

**Position-0 shielding** zeros the steering correction at the first token (attention sink).

### 2.2 Candidate Layers

```
Layer indices: [8, 11, 13, 15, 17, 19, 20, 21, 22, 24]
```

These 10 layers span mid-to-late transformer blocks of Phi-2 (32 layers total), covering the reasoning-critical region of the residual stream.

### 2.3 Optimizer and Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$) |
| Learning rate | $5 \times 10^{-4}$ |
| Weight decay | 0.01 (applied to `W_down`, `W_up`, and `mag_head.bias`; excluded from LayerNorm and conv) |
| Batch size | 2 |
| Gradient accumulation | 8 (effective batch = 16) |
| Gradient clipping | max norm 1.0 |
| Epochs | 5 |
| Warmup | 6% of total steps (linear ramp) |
| Decay | Cosine to 10% of peak LR |

The learning rate schedule follows linear warmup then cosine decay:

$$\eta(t) = \begin{cases}
\eta_{\max} \cdot \frac{t+1}{T_{\text{warm}}} & t < T_{\text{warm}} \\
\eta_{\min} + \frac{\eta_{\max} - \eta_{\min}}{2}\left(1 + \cos\frac{\pi(t - T_{\text{warm}})}{T - T_{\text{warm}}}\right) & t \geq T_{\text{warm}}
\end{cases}$$

where $\eta_{\min} = 0.10 \cdot \eta_{\max}$.

**Parameter groups** separate adapter parameters into three optimizer groups:
1. **Decay group**: `W_down.weight`, `mag_head.bias` — receive weight decay
2. **No-decay group**: LayerNorm parameters, conv parameters — no weight decay
3. **Up group**: `W_up.weight` — no weight decay (direction is scale-invariant due to normalization)

Additionally, the `LearnedLossWeights` module (8 parameters) is included as a fourth group with no weight decay.

### 2.4 Training Loop

```
for epoch in 1..5:
    shuffle data via length-bucketed mega-chunks (16 batches per chunk)

    for each gradient-accumulation window (8 micro-batches):
        compute window token counts per category
        for each micro-batch:
            if KL needed:
                forward pass with hooks disabled → base_logits (moved to CPU)
            forward pass with hooks enabled → steered_logits
            compute typed_objective(steered_logits, base_logits, ...)
            compute magnitude L1 penalty on prose tokens
            combine via Kendall weighting (if learned_weights) or fixed weights
            backward pass with GradScaler

        unscale gradients
        clip grad norm to 1.0
        optimizer step (skipped if scaler detects inf)
        scheduler step (skipped if optimizer step was skipped)

    evaluate on validation set → per-category NLL, critical loss, result_top1
    checkpoint if critical loss improves

    if on-policy is due (epoch ≥ 2, every 2 epochs, not last epoch):
        collect on-policy data → append to training set
        re-initialize scheduler for remaining epochs
```

**Length-bucketed batching**: within mega-chunks of `batch_size × 16` samples, examples are sorted by attended length to minimize padding waste. The resulting batches are then shuffled to preserve stochasticity.

**Checkpoint selection**: the model state is saved when the **critical loss** (weighted combination of per-category validation NLL on result and final tokens) reaches a new minimum.

### 2.5 Dual Forward Pass

Each micro-batch requires two forward passes through the backbone:

1. **Base pass** (hooks disabled, no gradient): produces `base_logits` for the frozen backbone — used as the focal difficulty reference and as the KL divergence anchor. Moved to CPU immediately after computation to free VRAM.

2. **Steered pass** (hooks enabled, gradient): produces `steered_logits` through the backbone with SteerNet corrections injected. This is the only pass that contributes gradients.

### 2.6 Token Type System

Every target token is classified into one of 7 categories via regex-based character annotation:

| ID | Category | Description | Example Tokens |
|----|----------|-------------|----------------|
| 0 | `NONE` | Padding / prompt | (masked with -100) |
| 1 | `FORMAT` | Structural markers | `Step`, `1:`, `Final answer:`, `####` |
| 2 | `PROSE` | Natural language reasoning | `She`, `needs`, `total`, `of` |
| 3 | `OPERATOR` | Arithmetic operators | `+`, `-`, `*`, `/`, `=` |
| 4 | `OPERAND` | Numeric values in expressions | `15`, `$3.50`, `120` |
| 5 | `RESULT` | Computation results (after `=`) | `= 45`, `= $12.00` |
| 6 | `FINAL` | Final answer value | `Final answer: 42` |

Annotation proceeds in priority order (later passes overwrite earlier ones): PROSE → OPERAND → OPERATOR → RESULT → FINAL → FORMAT. Each token inherits the highest-priority category among its constituent characters.

---

## 3. On-Policy Data Collection

Starting from epoch 2 and running every 2 epochs (but never on the last epoch), ARS augments its training set with on-policy completions — the model's own generated solutions to training questions.

### 3.1 Procedure

```
┌───────────────────────────────────────────────────┐
│              On-Policy Collection                   │
│                                                     │
│  1. Sample 25% of training questions                │
│  2. Build mixed-shot prompts (60/20/20 for 0/1/2)  │
│  3. Generate k=4 completions per question (T=0.7)   │
│  4. For each question:                              │
│     ├─ If any completion is correct → keep first    │
│     └─ If all fail AND gold CoT exists → use gold   │
│  5. Tokenize new rows with type annotations         │
│  6. Append to training set                          │
│                                                     │
│  On-policy generation uses the current SteerNet     │
│  with fixed-route steering enabled.                 │
└───────────────────────────────────────────────────┘
```

### 3.2 Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `on_policy_frac` | 0.25 | Fraction of training questions sampled |
| `on_policy_k` | 4 | Pass@k: number of completions per question |
| `on_policy_temperature` | 0.7 | Sampling temperature for diversity |
| `on_policy_start_epoch` | 2 | First epoch to run collection |
| `on_policy_interval` | 2 | Run every N epochs |
| `on_policy_gen_batch` | 4 | Batch size for generation |

### 3.3 Pass@k with Gold Fallback

For each sampled question, the model generates $k = 4$ completions at temperature $T = 0.7$. Each completion is scored against the gold answer using lenient numeric matching:

- **If any completion is correct**: the first correct completion is tokenized and added to the training set. This provides on-distribution training signal from the model's own successful reasoning chains.

- **If all completions fail but gold CoT exists**: the gold chain-of-thought answer is used as a fallback. This ensures hard questions still contribute training data rather than being silently dropped.

The mix of on-policy successes and gold fallbacks creates a curriculum that gradually shifts toward the model's own distribution while maintaining coverage of difficult problems.

### 3.4 Prompt Format During Collection

On-policy generation uses a different shot distribution than training: `[60%, 20%, 20%]` for `[0-shot, 1-shot, 2-shot]`. This simpler mixture reduces generation cost while maintaining diversity.

### 3.5 Scheduler Re-initialization

After appending on-policy rows, the training set grows. The LR scheduler is re-initialized for the remaining epochs with:
- Warmup steps set to 0 (training is already in the cosine regime)
- Total steps recalculated based on the expanded dataset
- Floor remains at 10% of peak LR

---

## 4. Phase 2: RLOO Router Training

Phase 2 freezes the Phase 1 SteerNet adapters and trains a JointLayerRouter to select per-question layer subsets via policy gradient optimization.

### 4.1 JointLayerRouter Architecture

```
                      ┌──────────────────────────────────────────────────┐
                      │              JointLayerRouter                     │
                      │                                                  │
  probe_hidden ──►    │  Pool ──► [mean_rms, rms_std, coverage]          │
  (B, T, 2560)        │             │                                    │
                      │             ▼                                    │
                      │  summary_proj ──► semantic (B, 16) ──► per-cand  │
                      │             │            semantic (B, N, 8)      │
                      │             ▼                                    │
                      │  Concat [scalars(3), depth(1), semantic(8)]      │
                      │  + identity_embed(N, 32)                         │
                      │             │                                    │
                      │        ┌────▼────┐                               │
                      │        │ Pass 1  │  TransformerEncoder(d=32,h=2) │
                      │        │         │──► utility_head ──► u_1       │
                      │        └────┬────┘                               │
                      │             │ + refine_proj(softmax(u_1))        │
                      │        ┌────▼────┐                               │
                      │        │ Pass 2  │  TransformerEncoder(d=32,h=2) │
                      │        │         │──► utility_head ──► u_final   │
                      │        └────┬────┘                               │
                      │             │                                    │
                      │        ┌────▼────┐                               │
                      │        │ k_head  │──► K distribution (B, K_max)  │
                      │        └─────────┘                               │
                      │             │                                    │
                      │        ┌────▼────┐                               │
                      │        │synergy  │──► S_ij = (eW)(e^T)/√d       │
                      │        │  proj   │    symmetric, zero diagonal   │
                      │        └─────────┘                               │
                      └──────────────────────────────────────────────────┘
```

**Input features** per candidate layer $i$:
- **Scalar statistics**: mean RMS, RMS std, attention coverage — computed from probe hidden states
- **Depth encoding**: normalized layer position $d_i / (L - 1)$
- **Semantic features**: per-candidate semantic vector (8-dim), derived from a shared semantic summary (16-dim) projected from the pooled probe representation
- **Identity embedding**: learned 32-dim embedding per candidate slot

**Two-pass refinement**: Pass 1 produces initial utility scores $u_1$. The softmax of $u_1$ is fed back as additional input for Pass 2, which produces the final utility logits $u$ and the K-distribution logits. This iterative design allows the router to refine its decisions after seeing initial preferences.

**Synergy matrix**: $S_{ij} = \frac{(e_i W_{\text{syn}})(e_j^T)}{\sqrt{d}}$, symmetrized and with zero diagonal. Captures cooperative interactions between layer pairs.

### 4.2 Probe Hook

The router reads from a **probe layer** — the middle candidate layer (default: `fixed_layers[5]`, i.e., layer 19). A `ProbeCapture` forward hook captures the hidden state at this layer during a no-steering forward pass, providing the router with a representation of the input before any steering is applied.

### 4.3 RLOO Policy Gradient

The router is optimized with Reinforced Leave-One-Out (RLOO), a variance-reduced policy gradient estimator that uses paired rollouts as mutual baselines.

#### 4.3.1 Sampling

For each training example, two independent routes $(A, B)$ are sampled:

1. **K sampling** via Gumbel-max on `k_logits`:

$$K \sim \text{Categorical}(\text{softmax}(k\_logits)), \quad K \in \{1, \ldots, K_{\max}\}$$

with $K$ clamped to $[K_{\min}, K_{\max}] = [2, 8]$.

2. **Subset sampling** via Cooperative Plackett-Luce:

For each of $K$ selection steps, a layer is chosen without replacement using Gumbel-max on the utility logits, optionally boosted by synergy with already-selected layers:

$$\text{step } t: \quad \text{score}_i^{(t)} = u_i + \sum_{j \in S_{<t}} S_{ij}$$

$$i^* = \arg\max_i \left[\log \text{softmax}(\text{score}^{(t)})_i + G_i\right], \quad G_i \sim \text{Gumbel}(0,1)$$

The log-probability of each sample is accumulated across steps for the policy gradient.

#### 4.3.2 Reward

Each sampled route is evaluated by running a no-gradient forward pass with that layer subset active:

$$R = -\text{NLL}_{\text{answer}} - \lambda_{\text{cost}} \cdot (K - 1)$$

where $\lambda_{\text{cost}} = 0.0$ by default (no cost penalty).

#### 4.3.3 RLOO Advantage

The leave-one-out baseline uses the paired sample:

$$A_A = R_A - R_B, \quad A_B = R_B - R_A$$

The policy gradient loss is:

$$\mathcal{L}_{\text{policy}} = -\mathbb{E}\left[(\log \pi_A - \log \pi_B) \cdot A_A\right]$$

where $\log \pi = \log p(K) + \log p(S | K)$.

### 4.4 Entropy Bonuses

Two decoupled entropy terms prevent policy collapse:

**K-distribution entropy**:

$$H_K = -\sum_{k=1}^{K_{\max}} p(k) \log p(k)$$

**Subset entropy** (sequential Plackett-Luce):

$$H_S = \frac{1}{K_{\max}} \sum_{t=1}^{K_{\max}} H\left(\text{softmax}(\text{score}^{(t)}_{\text{remaining}})\right)$$

The total loss combines these with configurable coefficients:

$$\mathcal{L} = \mathcal{L}_{\text{policy}} - \beta_K \cdot H_K - \beta_S \cdot H_S + \lambda_{\text{bal}} \cdot \mathcal{L}_{\text{balance}}$$

| Coefficient | Default | Purpose |
|-------------|---------|---------|
| $\beta_K$ (`router_beta_entropy_k`) | 0.05 | Prevents K from collapsing to a single value |
| $\beta_S$ (`router_beta_entropy_subset`) | 0.10 | Prevents subset from always picking the same layers |
| $\lambda_{\text{bal}}$ (`router_balance_weight`) | 0.10 | Anti-monopoly: penalizes uneven layer utilization |

### 4.5 Anti-Monopoly Balance Loss

To prevent a few layers from dominating all routes:

$$\mathcal{L}_{\text{balance}} = \text{MSE}\left(\bar{s},\ \frac{\bar{K}}{N}\right)$$

where $\bar{s}$ is the mean selection rate across the batch and $\bar{K}/N$ is the expected uniform selection rate.

### 4.6 Warmup

During the first epoch (`router_warmup_epochs = 1`), K is fixed to `router_min_k = 2`:
- The K-distribution head does not receive gradients
- Only the subset selection policy is trained
- Subset entropy bonus is still active

This warmup prevents early K collapse before the utility logits have learned meaningful layer preferences.

### 4.7 Collapse Detection and Recovery

After each epoch, a diversity health check evaluates:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `k_ge2_rate` | ≥ 0.85 | Fraction of examples selecting $K \geq 2$ |
| `layer_entropy_bits` | ≥ 0.55 | Entropy of layer selection distribution |
| `mean_k` | ≥ `router_min_k` | Average number of selected layers |

If any threshold is violated (health = COLLAPSED), the subset entropy coefficient is doubled:

$$\beta_S \leftarrow 2 \cdot \beta_S$$

This automatic recovery mechanism increases exploration pressure until diversity is restored.

### 4.8 Phase 2 Training Loop

```
for epoch in 1..3:
    if epoch ≤ warmup_epochs:
        fix K = router_min_k (no K gradient)

    for each batch:
        probe forward (no steering) → capture hidden states at probe layer
        router(probe_h, mask) → utility logits u, k_logits, synergy S

        sample route A: K_a ~ Gumbel-max(k_logits), S_a ~ CoopPL(u, K_a, S)
        sample route B: K_b ~ Gumbel-max(k_logits), S_b ~ CoopPL(u, K_b, S)

        evaluate NLL_a via forward with route A (no gradient)
        evaluate NLL_b via forward with route B (no gradient)

        compute RLOO advantage and policy gradient loss
        compute entropy bonuses (H_K, H_S) and balance loss
        backward, clip grad norm to 1.0, optimizer step

    diversity health check → double beta_S if collapsed
    evaluate validation NLL under router routing
    checkpoint if val_nll improves

restore best router checkpoint
install synergy-aware greedy selector for inference
```

### 4.9 Optimizer (Phase 2)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$) |
| Learning rate | $5 \times 10^{-5}$ |
| Schedule | Cosine annealing to 10% of peak |
| Gradient clipping | max norm 1.0 |
| Epochs | 3 |

### 4.10 Synergy-Aware Greedy Inference

At inference time, the router uses a greedy selection procedure that incorporates the learned synergy matrix:

```python
for each example:
    K = argmax(k_logits) + 1, clamped to [min_k, max_k]
    scores = u.clone()
    selected = {}
    for step in 1..K:
        chosen = argmax(scores[remaining])
        selected.add(chosen)
        scores = u + synergy[chosen]  # boost from cooperation with chosen layer
```

This synergy-aware greedy procedure matches the Cooperative Plackett-Luce sampling used during training, ensuring consistency between training and inference routing decisions.

---

## 5. Loss Function Details

### 5.1 Typed NLL Loss

The core Phase 1 loss is a **category-weighted, focal, window-normalized** negative log-likelihood:

$$\mathcal{L}_{\text{NLL}} = \sum_{c \in \mathcal{C}} \lambda_c \cdot \frac{\sum_{t \in c} w_t \cdot \text{NLL}_t}{|\text{window}_c|}$$

where:

- $\mathcal{C} = \{\text{FORMAT}, \text{PROSE}, \text{OPERATOR}, \text{OPERAND}, \text{RESULT}, \text{FINAL}\}$
- $\lambda_c$ are per-category importance weights (learned or fixed)
- $w_t = (1 - p_t^{\text{base}})^\gamma$ is the focal weight ($\gamma = 2.0$)
- $p_t^{\text{base}} = \exp(-\text{NLL}_t^{\text{base}})$ is the baseline model's confidence
- $|\text{window}_c|$ is the count of category-$c$ tokens in the gradient-accumulation window

**Window normalization** divides each category's loss sum by the total number of tokens of that category across the entire gradient-accumulation window (8 micro-batches). This prevents categories with few tokens (like FINAL) from having disproportionately large gradients.

### 5.2 Focal Weighting

Focal weighting concentrates the loss on tokens the base model finds difficult:

$$w_t = (1 - p_t^{\text{base}})^\gamma$$

When `base_logits` are available (from the KL pass), $p_t^{\text{base}}$ comes from the frozen backbone. This makes the loss independent of the adapter's current performance — the difficulty reference is always the unsteered model.

With $\gamma = 2.0$:
- A token the base model predicts correctly ($p \approx 1.0$): $w \approx 0$ — near-zero loss contribution
- A token the base model finds hard ($p \approx 0$): $w \approx 1$ — full loss contribution

### 5.3 Category Weights

#### Fixed Weights (Fallback)

| Category | $\lambda_c$ | Rationale |
|----------|-------------|-----------|
| FORMAT | 0.02 | Structural tokens — easy to learn, low priority |
| PROSE | 0.05 | Reasoning language — abundant, moderate importance |
| OPERATOR | 0.08 | Arithmetic operators — moderately important |
| OPERAND | 0.15 | Numeric inputs — important for computation accuracy |
| RESULT | 0.40 | Intermediate results — critical for chain correctness |
| FINAL | 0.35 | Final answer — the ultimate target |

#### Learned Weights (Default: `loss_mode = "typed"`)

When `LearnedLossWeights` is active, category weights are parameterized as softmax over 6 learnable logits, initialized to approximate the fixed weights:

$$\lambda_c = \text{softmax}(\ell_c), \quad \ell = [-1.85,\ -2.50,\ -0.47,\ 0.16,\ 1.14,\ 1.01]$$

These initial logits produce weights close to `[FORMAT: 0.02, PROSE: 0.01, OPERATOR: 0.08, OPERAND: 0.15, RESULT: 0.40, FINAL: 0.35]`, matching the fixed-weight design intent while allowing the optimizer to refine them.

### 5.4 KL Divergence Anchor

The KL term penalizes the steered model for deviating from the base model on prose tokens, preserving fluency on natural language while steering computation tokens:

$$\mathcal{L}_{\text{KL}} = \frac{1}{|\text{window}_{\text{PROSE}}|} \sum_{t \in \text{PROSE}} \text{KL}(q_t \| p_t^{\text{base}})$$

where $q_t = \text{softmax}(z_t^{\text{steered}})$ is the steered model's distribution and $p_t^{\text{base}} = \text{softmax}(z_t^{\text{base}})$ is the frozen backbone's distribution. This is forward KL:

$$\text{KL}(q \| p) = \sum_v q_v (\log q_v - \log p_v)$$

A secondary arithmetic KL term (`kl_arithmetic_weight = 0.30`) applies the same penalty to OPERAND and OPERATOR tokens, preventing over-steering of numeric content.

### 5.5 Magnitude L1 Regularization

An L1 penalty on the steering magnitude of prose tokens encourages selectivity:

$$\mathcal{L}_{L1} = \frac{1}{|\text{layers}| \cdot |\text{micro}|} \sum_{l} \text{mean}(m_l[\text{prose, active}])$$

where $m_l$ is the magnitude head output for layer $l$ and "active" means $m > 0$. This regularizer pushes the adapter to have near-zero magnitude on prose tokens (where the base model is already competent) while allowing large magnitudes on computation tokens.

### 5.6 Kendall Uncertainty Balancing

The three loss components (NLL, KL, L1) are combined using **homoscedastic uncertainty weighting** (Kendall et al., CVPR 2018):

$$\mathcal{L}_{\text{total}} = \frac{1}{2}\left(\sigma_{\text{NLL}}^{-2} \cdot \mathcal{L}_{\text{NLL}} + \sigma_{\text{KL}}^{-2} \cdot \mathcal{L}_{\text{KL}} + \sigma_{L1}^{-2} \cdot \mathcal{L}_{L1}\right) + \frac{1}{2}\sum_i \log \sigma_i^2$$

Parameterized via 3 log-variance parameters $s_i = \log \sigma_i^2$:

$$\mathcal{L}_{\text{total}} = \frac{1}{2}\left(e^{-s_0} \mathcal{L}_{\text{NLL}} + e^{-s_1} \mathcal{L}_{\text{KL}} + e^{-s_2} \mathcal{L}_{L1}\right) + \frac{1}{2}(s_0 + s_1 + s_2)$$

**Initialization**: $s = [0.0, 0.0, 0.916]$, giving initial precisions $e^{-s} \approx [1.0, 1.0, 0.40]$. The L1 init of 0.916 ($\approx \log 2.5$) starts it at an effective weight of ~0.20, matching `mag_l1_weight`.

The log-variance parameters are learned alongside the adapter weights, automatically balancing the relative importance of each objective during training. When one component's loss decreases (easier to optimize), its precision increases, shifting the optimizer's attention to the remaining objectives.

### 5.7 Complete Phase 1 Loss

Combining all components with Kendall weighting:

```
                    ┌─────────────────────────────────────────────┐
                    │          Phase 1 Total Loss                  │
                    │                                              │
                    │  L_total = Kendall(                          │
                    │    L_NLL   = Σ_c λ_c · focal_weighted_nll_c │
                    │    L_KL    = KL_prose + 0.3 · KL_arithmetic  │
                    │    L_L1    = mean(magnitude[prose, active])  │
                    │  )                                           │
                    │                                              │
                    │  where:                                      │
                    │    λ_c     = softmax(learned_logits)[c]      │
                    │    focal   = (1 - p_base)^γ · NLL            │
                    │    window  = normalize by per-category count │
                    │    Kendall = σ⁻² weighting + log σ² penalty  │
                    └─────────────────────────────────────────────┘
```

### 5.8 Validation Metrics

The validation loop computes per-category NLL and two aggregate metrics:

**Critical loss** (checkpoint selection):

$$\text{crit} = \sum_{c \in \mathcal{C}} \lambda_c \cdot \overline{\text{NLL}}_c$$

**Result top-1 accuracy**: fraction of RESULT and FINAL tokens where the model's argmax prediction matches the ground truth — a direct measure of computation accuracy.

---

## Hyperparameter Reference

### Phase 1

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name` | `microsoft/phi-2` | Frozen backbone |
| `lora_rank` | 16 | SteerNet bottleneck dimension |
| `rho_max` | 0.06 | Maximum relative steering amplitude |
| `steer_kappa` | 0.04 | Cap envelope coefficient |
| `phase1_lr` | 5e-4 | Peak learning rate |
| `phase1_epochs` | 5 | Training epochs |
| `batch_size` | 2 | Micro-batch size |
| `grad_accum` | 8 | Gradient accumulation steps |
| `clip_grad` | 1.0 | Max gradient norm |
| `focal_gamma` | 2.0 | Focal loss exponent |
| `kl_weight` | 2.0 | KL divergence coefficient (fixed-weight mode) |
| `kl_arithmetic_weight` | 0.30 | Arithmetic KL coefficient |
| `mag_l1_weight` | 0.20 | L1 penalty coefficient (fixed-weight mode) |
| `max_tok_len` | 512 | Maximum sequence length |
| `n_samples` | 7473 | Training examples from GSM8K |
| `seed` | 42 | Random seed |

### On-Policy

| Parameter | Value | Description |
|-----------|-------|-------------|
| `on_policy_frac` | 0.25 | Fraction of questions sampled |
| `on_policy_k` | 4 | Completions per question |
| `on_policy_temperature` | 0.7 | Sampling temperature |
| `on_policy_start_epoch` | 2 | First eligible epoch |
| `on_policy_interval` | 2 | Epoch interval |

### Phase 2

| Parameter | Value | Description |
|-----------|-------|-------------|
| `phase2_lr` | 5e-5 | Peak learning rate |
| `phase2_epochs` | 3 | Training epochs |
| `router_max_k` | 8 | Maximum active layers |
| `router_min_k` | 2 | Minimum active layers |
| `router_semantic_dim` | 16 | Shared semantic dimension |
| `router_d_model` | 32 | Transformer hidden dimension |
| `router_n_heads` | 2 | Attention heads |
| `router_beta_entropy_k` | 0.05 | K entropy bonus coefficient |
| `router_beta_entropy_subset` | 0.10 | Subset entropy bonus coefficient |
| `router_balance_weight` | 0.10 | Anti-monopoly balance coefficient |
| `router_warmup_epochs` | 1 | Epochs with fixed K |
| `lambda_cost` | 0.0 | Layer-count cost penalty |
