# Methodology & Algorithmic Formulation

This document provides the formal mathematical formulation and algorithmic specification for **Adaptive Residual Steering (ARS)**.

---

## 1. SteerNet: $K$-Conditioned Low-Rank Residual Correction

SteerNet operates as a non-invasive residual-stream intervention attached via PyTorch forward hooks at selected transformer blocks $l \in \mathcal{L}_{\text{target}}$. 

### 1.1 Architectural Formulation
For hidden state $h_l^{(t)} \in \mathbb{R}^d$ at layer $l$ and sequence position $t$, SteerNet computes candidate correction $\Delta h_l^{(t)}$:

$$\tilde{x}_l^{(t)} = \text{Dropout}\left(\text{GELU}\left(W_{\text{down}}^{(l)} \cdot \text{LN}\left(h_l^{(t)}\right)\right)\right) \in \mathbb{R}^r$$

To prevent signal dilution when multiple layers steer concurrently, SteerNet explicitly conditions on the realized number of active layers $K = |S| \in \{1, \dots, K_{\max}\}$:

$$\tilde{x}_{K, l}^{(t)} = \tilde{x}_l^{(t)} + W_K^{(l)} \cdot \left(\frac{K}{K_{\max}}\right) \in \mathbb{R}^r$$

The candidate residual update vector is then projected back to model dimension $d$:

$$\Delta h_l^{(t)} = W_{\text{up}}^{(l)} \cdot \left(\tilde{x}_{K, l}^{(t)} + \text{Dropout}\left(\text{GELU}\left(W_{\text{mid}}^{(l)} \cdot \tilde{x}_{K, l}^{(t)}\right)\right)\right) \times \left(\frac{\alpha_{\text{lora}}}{r}\right) \in \mathbb{R}^d$$

### 1.2 Parameter Constraints
- **Rank $r$**: Low-rank bottleneck (`lora_rank = 16`, where $d = 2560$ for Phi-2).
- **Zero-Initialization**: $W_K$ is initialized to zero ($W_K \leftarrow 0$) so that initialization matches standard execution.
- **Scaling Factor**: $\frac{\alpha_{\text{lora}}}{r} = \frac{32.0}{16} = 2.0$, ensuring stable gradient scale across rank variations.

---

## 2. Dynamic Usefulness Token Gate

The Token Gate module evaluates whether applying the proposed steering delta $\Delta h_l^{(t)}$ improves reasoning confidence at sequence position $t$.

### 2.1 Multi-Feature Concatenation
The gate evaluates four complementary geometric and semantic features:
$$f_l^{(t)} = \left[ \text{LN}\left(h_l^{(t)}\right), \; \text{LN}\left(\Delta h_l^{(t), \text{detach}}\right), \; \left\|\Delta h_l^{(t)}\right\|_2, \; \cos\left(h_l^{(t)}, \Delta h_l^{(t)}\right) \right] \in \mathbb{R}^{2d + 2}$$

### 2.2 Activation Computation
$$\alpha_l^{(t)} = \sigma\left(W_2 \cdot \text{GELU}\left(W_1 f_l^{(t)} + b_1\right) + b_2\right) \in [0, 1]$$

### 2.3 Regularization Objectives
During Stage 2B training, the gate is optimized using a composite regularized objective:
$$\mathcal{L}_{\text{gate}} = \mathcal{L}_{\text{usefulness}} + \lambda_{\text{L1}} \mathcal{L}_{\text{sparsity}} + \lambda_{\text{antisat}} \mathcal{L}_{\text{antisat}} + \lambda_{\text{div}} \mathcal{L}_{\text{diversity}} + \lambda_{\text{inv}} \mathcal{L}_{\text{invariance}}$$

1. **L1 Sparsity Loss**: Encourages dormant states on non-critical tokens:
   $$\mathcal{L}_{\text{sparsity}} = \frac{1}{T} \sum_{t=1}^T \alpha_t$$
2. **Anti-Saturation Loss**: Prevents gate collapse to binary limits ($0$ or $1$):
   $$\mathcal{L}_{\text{antisat}} = \frac{1}{T} \sum_{t=1}^T \left(\alpha_t^2 (1 - \alpha_t) + \alpha_t (1 - \alpha_t)^2\right)$$
3. **Diversity Floor Loss**: Enforces minimum variance across the sequence to prevent all-zero collapse:
   $$\mathcal{L}_{\text{diversity}} = \max\left(0, \tau_{\text{floor}} - \text{Var}_t(\alpha_t)\right)$$
4. **Prompt Invariance Loss**: Forces consistent gating across paraphrased CoT prompt templates.

---

## 3. Cooperative Layer Router & Anchor-Grounded 3-Arm RLOO

### 3.1 Two-Pass Attention Router Architecture
The router evaluates candidate layers $l \in \mathcal{L}_{\text{target}}$ through a 2-pass cross-attention mechanism:
1. **Pass 1 (Layer Utility Head)**: Computes unnormalized layer utilities $u_l \in \mathbb{R}$ for each candidate layer.
2. **Pass 2 ($K$-Distribution Head)**: Computes logits over the cardinality distribution $P(K) = \text{Softmax}(\pi_K)$ for $K \in \{1, \dots, K_{\max}\}$.

### 3.2 Tanh-Normalized Pairwise Synergy Matrix
Cross-layer interaction is modeled via a learnable symmetric matrix $W_{\text{syn}} \in \mathbb{R}^{N \times N}$. To prevent gradient saturation, the synergy matrix is strictly bounded:
$$S_{ij} = \tanh\left(\frac{W_{\text{syn}} + W_{\text{syn}}^\top}{2}\right) \in [-1, 1]$$
- $S_{ij} > 0$: Synergistic cooperation (both layers activate concurrently).
- $S_{ij} < 0$: Antagonistic interference (one layer suppresses the other).

### 3.3 Anchor-Grounded 3-Arm RLOO Estimator
At each bootstrap step, three distinct arms are evaluated:
- **Arm 0 (Anchor Baseline)**: Empty action $S_0 = \emptyset$ (frozen backbone, unsteered baseline reward $R_0$).
- **Arm A**: Sampled layer subset $S_A \sim P(S)$.
- **Arm B**: Independent sample $S_B \sim P(S)$.

The counterfactual leave-one-out baselines incorporate the anchor $R_0$:
$$b_A = \frac{R_B + R_0}{2}, \quad b_B = \frac{R_A + R_0}{2}$$
$$\Delta R_A = R_A - b_A, \quad \Delta R_B = R_B - b_B$$

The policy gradient loss is computed as:
$$\mathcal{L}_{\text{RLOO}} = -\frac{1}{2}\left[\Delta R_A^{\text{detach}} \log P(S_A) + \Delta R_B^{\text{detach}} \log P(S_B)\right]$$

### 3.4 Direct Pairwise Synergy Alignment Loss
The synergy matrix $S_{ij}$ is directly aligned with empirical RLOO reward differentials:
$$\mathcal{L}_{\text{synergy\_align}} = -0.5 \left[\Delta R_A^{\text{detach}} \cdot \bar{S}(S_A) + \Delta R_B^{\text{detach}} \cdot \bar{S}(S_B)\right]$$
where $\bar{S}(S)$ denotes the mean pairwise synergy among active layers in subset $S$.

### 3.5 Anti-Monopoly Balance Loss
To guarantee exploratory coverage and prevent layer monopolization:
$$\mathcal{L}_{\text{balance}} = \text{MSE}\left(p_{\text{batch}}, \; \frac{1}{N} \mathbf{1}\right)$$
where $p_{\text{batch}} \in \mathbb{R}^N$ is the batch-averaged layer selection probability.

---

## 4. Prompt Shielding Forward Pass Integration

The full end-to-end forward pass at target layer $l$ is defined as:

$$h_l^{(t)} \leftarrow h_l^{(t)} + \mathbb{I}(l \in S) \cdot M_{\text{shield}}(t) \cdot \alpha_l^{(t)} \cdot \Delta h_l^{(t)}$$

where:
- $\mathbb{I}(l \in S) \in \{0, 1\}$: Binary indicator whether layer $l$ is in active subset $S$.
- $M_{\text{shield}}(t) \in \{0, 1\}$: Prompt shield mask ($0$ for few-shot demonstrations, $1$ for target reasoning).
- $\alpha_l^{(t)} \in [0, 1]$: Dynamic token usefulness gate.
- $\Delta h_l^{(t)} \in \mathbb{R}^d$: $K$-conditioned low-rank residual steering delta.
