# Methodology & Algorithmic Formulation

## 1. SteerNet: $K$-Conditioned Low-Rank Residual Correction
For candidate layer $l$, the correction delta is computed via a low-rank bottleneck MLP:
$$\tilde{x} = \text{Dropout}(\text{GELU}(W_{\text{down}} \cdot \text{LN}(h)))$$
$$\tilde{x}_K = \tilde{x} + W_K \left(\frac{K}{4}\right)$$
$$\Delta h = W_{\text{up}} \left(\tilde{x}_K + \text{Dropout}(\text{GELU}(W_{\text{mid}} \tilde{x}_K))\right) \times \frac{\alpha_{\text{lora}}}{r}$$

Where:
- $r$ is the low-rank dimension (`lora_rank = 16`).
- $W_K$ is zero-initialized so that default execution without $K$-information remains unperturbed.
- $\frac{\alpha_{\text{lora}}}{r} = \frac{32}{16} = 2.0$ acts as a constant scaling factor.

## 2. Usefulness Gate: Per-Token Activation
The gate module evaluates whether applying the steering delta $\Delta h$ is beneficial at position $t$:
$$f_t = [\text{LN}(h_t), \text{LN}(\Delta h_t^{\text{detach}}), \|\Delta h_t\|_2, \cos(h_t, \Delta h_t)] \in \mathbb{R}^{2d + 2}$$
$$\alpha_t = \sigma(W_2 \cdot \text{GELU}(W_1 f_t + b_1) + b_2)$$

Key design constraints:
- **Detached Delta:** $\Delta h$ is detached so the gate cannot backpropagate gradients into SteerNet.
- **Conservative Initialization:** Output bias $b_2$ is initialized to $-6.0$, giving $\alpha_{\text{init}} \approx \sigma(-6.0) \approx 0.0025$, requiring the gate to earn activation.

## 3. Joint Layer Router: RLOO Counterfactual Subset Selection
Rather than independent per-layer scalar cutoffs, the router operates over candidate subsets:
- **Candidate Pool:** 8 evenly spaced layers across transformer depth (e.g. layers 9 to 24 for Phi-2).
- **Subset Sampling:** Two subsets $S_a, S_b$ are sampled per step using Plackett-Luce or catalog distributions.
- **Reward Function:**
  $$R(S) = -\text{NLL}_{\text{answer}}(S) - \lambda_{\text{cost}} \cdot |S|$$
- **RLOO Advantage (Leave-One-Out for $N=2$):**
  $$A(S_a) = R(S_a) - R(S_b) = -A(S_b)$$
- **Policy Gradient Update:**
  $$\mathcal{L}_{\text{router}} = -(\log P(S_a) - \log P(S_b)) \cdot A(S_a)^{\text{detach}} - \beta_{\text{entropy}} \mathcal{H}(P)$$

This formulation eliminates convergence to margin zero ($\mu \approx \tau$) and prevents catastrophic subset collapse.
