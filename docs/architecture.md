# ARS Architecture

Comprehensive technical reference for every component in the Adaptive Residual Steering system.

---

## System Overview

ARS steers a frozen `microsoft/phi-2` (2.7B) toward structured chain-of-thought reasoning by injecting learned residual corrections into the transformer's residual stream. No backbone parameter is ever modified.

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   Phi2WithRSC Wrapper                   │
                    │                                                         │
  input_ids ──►     │   ┌─────────┐   hook    ┌───────────┐                  │
  attn_mask ──►     │   │ Phi-2   │ ────────► │ SteerNet  │──► δ (delta)     │
                    │   │ Layer i │           │ V5 @ i    │                  │
                    │   └────┬────┘           └───────────┘                  │
                    │        │                      │                         │
                    │        │    h_new = h + δ     │                         │
                    │        ◄──────────────────────┘                         │
                    │                                                         │
                    │   Repeated for each layer i ∈ active_route             │
                    │                                                         │
                    │   Route selected by:                                    │
                    │     Phase 1: fixed_layers (all 10 candidates)          │
                    │     Phase 2: JointLayerRouter (per-question subset)    │
                    └─────────────────────────────────────────────────────────┘
```

### Candidate Layers

10 layers selected from Phi-2's 32 transformer blocks:

```
Layer:  0  1  2 ... 7  [8] 9 10 [11] 12 [13] 14 [15] 16 [17] 18 [19] [20] [21] [22] 23 [24] 25 ... 31
                        ▲         ▲        ▲        ▲        ▲       ▲    ▲    ▲    ▲        ▲
                     candidate candidate  ...  (10 candidates total, bracketed above)
```

Layers were chosen to span the mid-to-late transformer depth where Phi-2 performs reasoning and answer formation. Early layers (0-7) handle tokenization-level representations and are left untouched.

---

## SteerNet V5

**File**: `src/steering/steernet.py`

Each candidate layer gets its own SteerNet instance — a rank-16 bottleneck adapter with causal temporal mixing.

### Architecture

```
h (B, T, D=2560)
   │
   ▼
LayerNorm(D)
   │
   ▼
W_down: Linear(D → 16, no bias)      ← Xavier uniform, gain=1.0
   │
   ▼
GELU activation
   │
   ▼
CausalDWConv1d(channels=16, k=4)     ← Causal depthwise conv for temporal mixing
   │
   ├──────────────────────┐
   ▼                      ▼
W_up: Linear(16 → D)   mag_head: Linear(16 → 1) → sigmoid
   │                      │
   ▼                      ▼
direction (D)          magnitude ∈ [0, 1]
   │                      │
   └──────┬───────────────┘
          ▼
   δ = (ρ_max / √K) × magnitude × ‖h_centered‖ × direction / ‖direction‖
          │
          ▼
   Smooth cap: δ_capped = δ × C_l / √(‖δ‖² + C_l²)
          │
          ▼
   Position-0 shield × Prompt shield × Injection mask
          │
          ▼
   h_new = h + δ_capped
```

### Key Design Choices

**Relative Amplitude** (`amplitude_mode="relative"`):
- Steering magnitude scales with the local hidden-state norm: `ρ_max × magnitude × ‖h - mean(h)‖`
- This makes steering resolution-invariant across layers (layer 8 norms ~52, layer 24 norms ~90)
- `rho_max=0.06` is the ONLY amplitude knob — `lora_alpha` and `steer_scale` have no effect in relative mode

**√K Scaling**:
- When K layers are active, each layer's amplitude scales by `1/√K`
- Prevents total steering energy from growing linearly with route size
- At K=4: each layer contributes 50% of its solo amplitude; total energy = 2× solo

**Smooth Cap Envelope**:
- Per-layer cap: `C_l = κ × N_l` where `N_l` is the calibrated median hidden-state norm at layer l
- Soft saturation: `f(n) = n × C_l / √(n² + C_l²)` — approaches `C_l` asymptotically, no hard clipping
- `κ=0.04` keeps steering within ~4% of the layer's natural norm range

**Magnitude Head Initialization**:
- `bias = -3.0` → `sigmoid(-3) ≈ 0.047` — starts near-zero (selective)
- Typed loss lifts result/final tokens while L1 penalty keeps prose tokens suppressed
- This learned selectivity replaces manual per-type magnitude tables

### CausalDWConv1d

**File**: `src/steering/steernet.py`, class `CausalDWConv1d`

Depthwise 1D convolution with strict causality:

```
Prefill mode:
  [0, 0, 0, x₁, x₂, x₃, x₄, x₅]  ← left-pad with k-1 zeros
         ↓ conv1d (groups=channels)
  [y₁, y₂, y₃, y₄, y₅]             ← each yᵢ sees only x₁..xᵢ

Decode mode (streaming):
  cache = [x_{t-3}, x_{t-2}, x_{t-1}]   ← rolling buffer of last k-1 tokens
  input = [x_t]
  concat → conv → output → update cache
```

- **Initialization**: Dirac-centered — current-token tap = 1.0, history taps = 0.01 noise
- **Purpose**: Lets the adapter see local temporal context (4 tokens) to detect multi-step reasoning patterns without global attention
- **Prefill primes the cache** so decode mode starts with real context, not zeros

### Position-0 Shielding

The first token in every sequence (position 0) receives zero steering. In Phi-2, this is the attention-sink token — all layers attend to it heavily, so perturbing it destabilizes the entire sequence.

---

## JointLayerRouter

**File**: `src/routing/router.py`

Scores all candidate layers JOINTLY and selects a per-question subset at inference time.

### Architecture

```
probe_hidden (B, T, D)
       │
       ▼
  ┌─── _base_features ───┐
  │                       │
  │  h_det = h.detach()   │
  │  rms = √(mean(h²))   │    Scalars: [mean_rms, rms_std, coverage]
  │  pool = weighted_mean │    (3 values per batch item)
  │                       │
  │  summary_proj(pool)   │    Semantic: (B, N, 8)
  │  → N × 8 candidate   │    (per-candidate learned readout)
  │    semantic features  │
  └───────────────────────┘
            │
            ▼
  tokens_in = [scalars(3) ‖ depth(1) ‖ semantic(8)]  →  input_proj(12 → 32)
            │
            + identity_embed(N, 32)   ← learnable per-candidate identity
            │
            ▼
  ┌─── Pass 1 ───┐
  │ TransformerEncoderLayer(d=32, heads=2, ff=64, GELU) │
  │ utility_head: Linear(32 → 1) → u₁ (initial scores)  │
  └──────────────┘
            │
            ▼
  soft_select = softmax(u₁)
  tokens_2 = attended_1 + refine_proj(soft_select)
            │
            ▼
  ┌─── Pass 2 (Refinement) ───┐
  │ TransformerEncoderLayer(d=32, heads=2, ff=64, GELU)  │
  │ utility_head_refine: Linear(32 → 1) → u_final        │
  │ k_head: Linear(32 → K_max) → k_logits                │
  │ synergy_proj: Linear(32 → 32) → S_ij                  │
  └────────────────────────────┘
```

### Two-Pass Design

**Pass 1** scores each candidate independently (with cross-attention for context). **Pass 2** feeds back the soft-selection distribution from Pass 1, allowing the router to refine scores based on "who else is likely selected." This iterative refinement captures layer interactions without exponential subset enumeration.

### Synergy Matrix

```
S_ij = (e_i × W_syn × e_j^T) / √d_model

S is symmetrized: S = 0.5 × (S + S^T)
Diagonal zeroed: S_ii = 0
```

- `S_ij > 0`: layers i and j cooperate — selecting both yields bonus utility
- `S_ij < 0`: layers i and j interfere — avoid co-selecting them
- Used during both training (Cooperative Plackett-Luce) and inference (synergy-aware greedy)

### K Distribution

The `k_head` outputs logits over `{1, 2, ..., K_max}` where `K_max=8`. During training, K is sampled via Gumbel-max. During inference, K is the argmax, clamped to `[min_k=2, max_k]`.

### Synergy-Aware Greedy Inference

```python
def selector(batch_size, device):
    u, k_logits, synergy = router.recompute_from_cache(return_synergy=True)
    k = k_logits.argmax(-1) + 1

    for each batch item b:
        remaining = all candidates
        scores = u[b]                    # initial utility scores
        for step in range(k[b]):
            chosen = argmax(scores, remaining)
            mark chosen as selected
            scores = u[b] + synergy[b, chosen]  # update scores with synergy from chosen layer
```

This greedy procedure matches the Cooperative Plackett-Luce sampling used during training — first pick is by raw utility, subsequent picks incorporate synergy with already-selected layers.

### Diagnostic Tracking

The router maintains exponential moving averages (α=0.02) for:
- `logit_ema` / `logit_sq_ema`: Per-layer utility logit statistics
- `selection_rate_ema`: Per-layer selection frequency
- `k_dist_ema`: Distribution over K values
- `selected_forwards` / `seen_forwards`: Absolute selection counts

Health is assessed via `route_diversity_stats()`:
- `mean_k`: Expected number of active layers
- `k_ge2_rate`: Fraction of selections with K ≥ 2
- `layer_entropy_bits`: Binary entropy of per-layer selection rates (max 1.0 bit)

---

## RLOO Policy Gradient

**File**: `src/routing/rloo.py`

Trains the router using Leave-One-Out REINFORCE with Cooperative Plackett-Luce subset sampling.

### Algorithm

```
For each training batch:
  1. Forward router on cached probe hidden states → (u, k_logits, synergy)
  2. Sample (K_a, subset_a) and (K_b, subset_b) independently
  3. Evaluate NLL for each route (backbone forward with SteerNet, no gradient)
  4. Compute RLOO advantage: advantage_a = reward_a - reward_b
  5. Policy gradient: L_policy = -E[(logπ_a - logπ_b) × advantage_a]
  6. Add entropy bonuses and balance regularization
```

### Cooperative Plackett-Luce Sampling

Sequential without-replacement sampling where each step's probabilities are boosted by synergy with already-selected layers:

```
For step t in range(K):
  log_p = log_softmax(u + synergy_boost, remaining only)
  sample via Gumbel-max (no gradient through choice, gradient through log_p)
  update synergy_boost from newly selected layer
```

This couples the subset distribution with the learned synergy matrix — layers that cooperate well are more likely to be co-selected.

### Loss Components

```
L = L_policy - β_k × H(K) - β_subset × H(subset) + λ_balance × L_balance

L_policy:   REINFORCE with RLOO baseline (A vs B)
H(K):       Entropy of K distribution (prevents K collapse to a single value)
H(subset):  Sequential Plackett-Luce entropy (prevents layer monopoly)
L_balance:  MSE between mean selection rate and uniform target
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| `β_k` (entropy K) | 0.05 | Prevent K from collapsing to one value |
| `β_subset` (entropy subset) | 0.10 | Prevent one layer from dominating selection |
| `λ_balance` | 0.10 | Anti-monopoly: push selection rates toward uniform |
| `λ_cost` | 0.00 | K-cost penalty (disabled — let router freely choose K) |

### Warmup

During the first `router_warmup_epochs` epochs, K is fixed to `min_k` and the K-distribution entropy bonus is disabled. This lets the utility logits and synergy matrix stabilize before introducing the K-selection policy.

---

## Phi2WithRSC Wrapper

**File**: `src/models/wrapper.py`

Orchestrates steering injection without modifying the Phi-2 backbone.

### Hook Mechanism

```
1. register_forward_pre_hook on base_model:
   - Detect prefill vs. decode (T=1 → decode)
   - Call _select_route() → per-batch boolean mask (B, N_candidates)
   - Reset guard_state and conv cache on prefill
   - Activate prompt shield mask

2. register_forward_hook on each candidate layer:
   - If layer is active for this batch item: compute SteerNet delta, inject
   - If layer is inactive: pass through unchanged
```

### Routing Modes

| Mode | When | Behavior |
|------|------|----------|
| `"only"` + fixed_layers | Phase 1 training | All candidates active |
| `"matrix"` | RLOO evaluation | Per-item boolean mask from sampled subsets |
| `RouteSelector` | Phase 2 inference | Router's `make_selector()` callable |
| `"none"` | Baseline eval | No steering (clean backbone) |

### Prompt Shield

When `prompt_shielding=True` and few-shot exemplars are present:
- A boolean mask marks all tokens belonging to the preface + demonstrations
- Shielded tokens receive zero steering delta (set via `injection_mask`)
- Prevents steering from corrupting the format patterns in few-shot demos

### Guard State

Per-forward tracking of:
- `rel_max`: Maximum relative steering magnitude (‖δ‖/‖h‖) across all layers and tokens
- `nonfinite`: Whether any non-finite values were produced

If `guard_revert_to_k0=True` and guard detects anomalies, the system can fall back to no steering.

---

## Typed NLL Loss

**File**: `src/training/losses.py`

### Token Type System

Every target token is classified into one of 7 categories:

| ID | Name | Description | Example |
|----|------|-------------|---------|
| 0 | T_NONE | Non-target / padding | Prompt tokens |
| 1 | T_FORMAT | Structural markers | `Step 1:`, `####` |
| 2 | T_PROSE | Natural language reasoning | "There are 5 apples..." |
| 3 | T_OPERATOR | Arithmetic operators | `+`, `-`, `*`, `/`, `=` |
| 4 | T_OPERAND | Numbers in expressions | `42`, `$3.50` |
| 5 | T_RESULT | Computation results after `=` | `= 15` |
| 6 | T_FINAL | The final answer | `Final answer: 42` |

Classification uses regex-based character annotation with ascending priority — if a character matches multiple patterns, the highest-priority category wins.

### Loss Weighting

Each category has a learned weight via `LearnedLossWeights` (Kendall uncertainty):

```
L_total = 0.5 × exp(-s_nll) × L_nll + 0.5 × exp(-s_kl) × L_kl + 0.5 × exp(-s_l1) × L_l1 + 0.5 × Σ s_i

L_nll = Σ_c  λ_c × Σ_t∈c [focal_weight(t) × NLL(t)] / count(c)
```

- **Focal weighting**: `w(t) = (1 - p_base(t))^γ` using the frozen base model's confidence
- **Window normalization**: Each category's loss is normalized by its token count in the gradient-accumulation window, not the batch — prevents rare categories (T_FINAL) from being drowned out

### KL Divergence

Two selective KL terms preserve base-model behavior where steering should be minimal:
- **KL_prose** (`kl_weight=2.0`): Penalizes divergence on prose tokens (reasoning text)
- **KL_arithmetic** (`kl_arithmetic_weight=0.30`): Light regularization on operand/operator tokens

---

## Configuration

**File**: `src/configs/config.py`

All hyperparameters live in a single `RSCConfig` dataclass with:
- Type-safe defaults certified for Phi-2
- `validate()` method enforcing Phase 1 contracts
- `apply_overrides()` for CLI `key=value` overrides with type-aware parsing

### Critical Constraints

```python
steernet_arch == "v5"           # Only V5 supported
objective == "nll"              # Token-weighted NLL only
k_scale_mode == "sqrt"          # √K scaling enforced
amplitude_mode ∈ {"relative", "capped"}
fixed_layers ⊂ candidate_layers
```

---

## Data Pipeline

**File**: `src/data/dataset.py`

### Multi-Context Stratified Mixture

Training data uses a stratified shot distribution:

| Shots | Probability | Purpose |
|-------|-------------|---------|
| 0-shot | 50% | Hardest condition, forces format learning |
| 1-shot | 10% | Minimal demonstration |
| 2-shot | 20% | Standard few-shot |
| 3-shot | 5% | Extended context |
| 4-shot | 15% | Maximum context |

### Prompt Construction

```
[optional] System preface (PREFACE)
[optional] V5 exemplar demonstrations (90% from curated bank, 10% from train set)
Question: {question}
Answer: Let's think step by step.
Step 1: {reasoning}
...
Final answer: {number}
```

### On-Policy Data Collection

After `on_policy_start_epoch` (epoch 2), every `on_policy_interval` (2) epochs:
1. Sample questions from the training set
2. Generate `pass@k` (k=4) completions at temperature 0.7
3. Extract and verify answers
4. **Correct completions** → add to training set as on-policy examples
5. **All failed** → fall back to gold CoT from the `answer` field

The gold fallback ensures hard questions still contribute supervision even when the model can't solve them yet.

---

## File Map

```
src/
├── configs/config.py          RSCConfig dataclass, CLI overrides
├── data/
│   ├── dataset.py             GSM8K loading, tokenization, multi-context splits
│   └── prompts.py             Prompt builder with shield masks, V5 exemplars
├── models/wrapper.py          Phi2WithRSC: hooks, routing, guard, generation
├── steering/steernet.py       SteerNet V5, CausalDWConv1d, smooth cap
├── routing/
│   ├── router.py              JointLayerRouter, synergy matrix, diagnostics
│   └── rloo.py                RLOO policy gradient, Cooperative Plackett-Luce
├── training/
│   ├── trainer.py             Phase 1 loop: typed loss, on-policy, LR schedule
│   ├── router_trainer.py      Phase 2 loop: RLOO, warmup, health monitoring
│   └── losses.py              Typed NLL, focal weighting, Kendall balancing
├── evaluation/
│   ├── benchmark.py           Multi-arm benchmark (baseline/fixed/router)
│   ├── answers.py             Numeric extraction and scoring
│   └── diagnostics.py         Plots, training curves, generation viewer
└── deploy/calibrate.py        Deployment contract calibration
```
