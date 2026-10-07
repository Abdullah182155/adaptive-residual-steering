# Research Background & Problem Formulation

## 1. Problem Statement
Autoregressive Large Language Models (LLMs) often exhibit reasoning failures on multi-step arithmetic, symbolic, and logic problems (such as those in the GSM8K benchmark). While standard remedies involve:
1. **Full fine-tuning (SFT):** Modifying all weights, which is computationally prohibitive and risks catastrophic forgetting of general capabilities.
2. **Parameter-Efficient Fine-Tuning (PEFT / LoRA):** Adapting attention or MLP projections directly, which still alters the frozen model dynamics globally across all generation steps.
3. **Prompt engineering (CoT / Few-Shot):** Relying solely on input formatting without fixing internal latent calculation drift.

## 2. The Steering Hypothesis
Instead of updating model parameters $W$, **Residual Steering** modifies hidden states $h_l^{(t)}$ dynamically inside the residual stream at transformer layer $l$:
$$h_l^{(t)} \leftarrow h_l^{(t)} + \Delta h_l^{(t)}$$

Prior static steering approaches apply fixed steering vectors across all tokens and layers. However, this causes two critical failure modes:
1. **Uniform degradation:** Intervening on syntax/formatting tokens (e.g. whitespace, commas, boilerplate prefixes) corrupts language fluency.
2. **Layer interference / Negative transfer:** Intervening indiscriminately across all layers creates conflicting representations, leading to performance drops (such as the observed Stage B $\to$ Stage C regression).

## 3. The Adaptive Residual Steering (ARS) Hypothesis
ARS addresses these challenges through a principled, selective architecture:
- **Token-Level Selectivity (Usefulness Gating):** Compute a per-token gate $\alpha_t \in [0, 1]$ conditioned on the hidden state $h_t$ and proposed delta $\Delta h_t$, training it to fire only when steering improves task likelihood.
- **Layer-Level Dynamic Compute (Catalog / RLOO Routing):** Treat layer selection as a discrete subset decision. A learned policy chooses an optimal subset $S \subseteq \{l_1, \dots, l_N\}$ per problem instance, with an explicit option for $S = \emptyset$ (no-op) when the base model is already confident.
- **$K$-Aware Compensation:** Condition low-rank MLPs (SteerNet) on the number of concurrently active layers $K = |S|$, preventing signal dilution when multiple layers steer simultaneously.
