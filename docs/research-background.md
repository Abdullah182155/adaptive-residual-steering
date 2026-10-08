# Research Background & Problem Formulation

## 1. Introduction & Foundational Motivation

Autoregressive Large Language Models (LLMs) parameterized by billions of transformer weights demonstrate remarkable fluency and emergent linguistic capabilities [1, 2]. However, on complex multi-step reasoning benchmarks—such as grade-school mathematics (GSM8K [3]), symbolic multi-step arithmetic (SVAMP [4]), competition mathematics (MATH-500 [5]), and scientific problem solving (ARC [6])—even state-of-the-art foundation models frequently suffer from catastrophic reasoning failures.

These failures rarely stem from vocabulary limitations or syntax errors; rather, they arise from **latent calculation drift**: intermediate mathematical operations and logical deductions wander off-course within deep internal representation spaces [7, 8].

To rectify latent reasoning errors, the community has predominantly relied upon three standard paradigms:
1. **Supervised Full Fine-Tuning (SFT)**: Updating all model parameters $W \in \mathbb{R}^{d \times d}$. While expressive, full fine-tuning requires substantial compute, degrades the general conversational capabilities of the base model via catastrophic forgetting, and causes over-fitting to specific training distributions [9].
2. **Parameter-Efficient Fine-Tuning (PEFT / LoRA)**: Freezing the backbone and training low-rank adapter matrices $\Delta W = B \cdot A$ injected into attention or feed-forward projections [10]. While parameter-efficient, LoRA permanently alters the model dynamics across *every* generated token and *every* forward pass, lacking temporal, semantic, or difficulty-aware selectivity.
3. **Prompt Engineering & Chain-of-Thought (CoT)**: Guiding generation through step-by-step prefixes ("Let's think step by step") [11]. While non-invasive, prompt engineering cannot modify internal latent trajectories once an error begins propagating.

---

## 2. The Residual Stream Mechanics & Activation Intervention

In standard Decoder-only Transformer architectures (e.g., GPT, LLaMA, Phi [12, 13]), each layer $l \in \{1, \dots, L\}$ reads from and writes to a shared high-dimensional representation channel known as the **Residual Stream**:

$$h_l^{(t)} = h_{l-1}^{(t)} + \text{Attn}_l\left(\text{LN}(h_{l-1}^{(\le t)})\right) + \text{FFN}_l\left(\text{LN}(h_{l-1}^{(t)})\right)$$

where $h_l^{(t)} \in \mathbb{R}^d$ denotes the hidden representation at sequence position $t$.

### 2.1 The Promise and Pitfalls of Activation Addition
Recent work in mechanistic interpretability and representation engineering [14, 15] has demonstrated that internal representations encode high-level semantic concepts, truthfulness vectors, and reasoning directions. **Activation Addition (ActAdd)** [16] and **Contrastive Activation Addition (CAA)** [17] intervene directly on the residual stream without modifying model weights:

$$h_l^{(t)} \leftarrow h_l^{(t)} + c \cdot v_l$$

where $v_l \in \mathbb{R}^d$ is a pre-computed static steering vector and $c \in \mathbb{R}$ is an intervention coefficient.

However, static activation addition exhibits two fundamental pathologies:
1. **Syntactic and Fluency Degradation**: Applying a fixed steering vector across *every token position* $t$ corrupts syntax tokens (e.g., whitespace, punctuation, prepositions), causing repetition loops, linguistic degeneration, and fluency collapse.
2. **Multi-Layer Destructive Interference**: Intervening across all layers simultaneously causes antagonistic cross-layer interference, where interventions in early layers disrupt representations expected by intermediate layers, frequently leading to performance worse than the unsteered baseline.

---

## 3. The Adaptive Residual Steering (ARS) Paradigm

**Adaptive Residual Steering (ARS)** resolves these fundamental limitations by establishing a unified, non-invasive, three-tiered adaptive control mechanism:

```
                      ┌────────────────────────────────────────┐
                      │         Input Prompt Sequence          │
                      └──────────────────┬─────────────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │   Cooperative Layer Router    │
                         │   (3-Arm Anchor-Grounded)     │
                         └───────┬───────────────┬───────┘
                                 │               │
                  Active Layer S │               │ Empty S = ∅ (No-op)
                                 ▼               ▼
                      ┌──────────────────┐  [Pure Frozen Baseline]
                      │ Target Layer l   │
                      └────────┬─────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
            ▼                                     ▼
 ┌──────────────────────┐              ┌──────────────────────┐
 │       SteerNet       │              │    Usefulness Gate   │
 │ (K-Conditioned MLP)  │              │ (Per-Token α_t ∈ [0,1)
 └──────────┬───────────┘              └──────────┬───────────┘
            │ Δh_l^(t)                            │ α_t
            └──────────────────┬──────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Prompt Shielding  │ ◄─── Protects Few-Shot Exemplars
                    └──────────┬──────────┘
                               │ M_shield * α_t * Δh_l^(t)
                               ▼
                    ═══════════════════════
                        Residual Stream
                    ═══════════════════════
```

### 3.1 Token-Level Usefulness Gating
Rather than applying constant steering, ARS introduces an active evaluator gate $\alpha_t \in [0, 1]$ conditioned jointly on the input hidden state $h_t$ and the candidate intervention vector $\Delta h_t$:
$$\alpha_t = \sigma\left(W_2 \cdot \text{GELU}(W_1 f_t + b_1) + b_2\right)$$
where $f_t = [\text{LN}(h_t), \text{LN}(\Delta h_t^{\text{detach}}), \|\Delta h_t\|_2, \cos(h_t, \Delta h_t)]$.
The gate is regularized via an L1 sparsity penalty and contrastive usefulness objective, guaranteeing that it fires **strictly on semantic calculations** (e.g., calculating $21 - 15 = 6$) while remaining completely dormant on boilerplate text.

### 3.2 Cooperative Layer Routing & Anchor-Grounded 3-Arm RLOO
Layer selection is treated as a combinatorial discrete subset optimization problem $S \subseteq \{l_1, \dots, l_N\}$. ARS introduces:
- **Pairwise Layer Synergy Modeling**: A learnable symmetric interaction matrix $S_{ij} \in [-1, 1]$ normalized by $\tanh$, explicitly quantifying synergistic cooperation ($S_{ij} > 0$) vs antagonistic interference ($S_{ij} < 0$).
- **Anchor-Grounded 3-Arm RLOO**: A policy-gradient advantage estimator utilizing Arm 0 (unsteered baseline anchor $R_0$) to establish an absolute variance-reduction floor:
  $$b_A = \frac{R_B + R_0}{2}, \quad \Delta R_A = R_A - b_A$$
- **Anti-Monopoly Regularization**: Entropy penalties that prevent the router from collapsing to single-layer dominance.

### 3.3 Prompt Shielding for In-Context Learning
To prevent the catastrophic demonstration representation drift observed when steering multi-shot contexts, ARS incorporates temporal prompt shielding:
$$h_l^{(t)} \leftarrow h_l^{(t)} + M_{\text{shield}}(t) \cdot \alpha_t \cdot \Delta h_l^{(t)}$$
This preserves few-shot exemplars with zero representation drift, allowing in-context learning and dynamic steering to reinforce one another cooperatively.

---

## 4. Academic References & Bibliography

### Activation Addition & Representation Engineering
1. **Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023).** *Activation Addition: Steering Language Models Without Fine-Tuning.* arXiv preprint arXiv:2308.10248.
2. **Rimsky, N., Gabrieli, N., Schulz, J., Megill, M., & Turner, A. (2023).** *Steering Llama 2 via Contrastive Activation Addition.* arXiv preprint arXiv:2312.06681.
3. **Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mantena, M., Sheng, P., Wang, P., & Hendrycks, D. (2023).** *Representation Engineering: A Top-Down Approach to AI Transparency and Control.* arXiv preprint arXiv:2310.01405.
4. **Subramani, N., Suresh, N., & Peters, M. E. (2022).** *Extracting Latent Steering Vectors from Pretrained Language Models.* In *Findings of the Association for Computational Linguistics: ACL 2022*, pages 1978–1986.

### Parameter-Efficient Adaptation & Mixture of Experts
5. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).** *LoRA: Low-Rank Adaptation of Large Language Models.* In *International Conference on Learning Representations (ICLR 2022)*.
6. **Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019).** *Parameter-Efficient Transfer Learning for NLP.* In *International Conference on Machine Learning (ICML 2019)*, pages 2790–2799.
7. **Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017).** *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* In *International Conference on Learning Representations (ICLR 2017)*.

### Reinforcement Learning, RLOO & Plackett-Luce Choice Models
8. **Kool, W., van Hoof, H., & Welling, M. (2019).** *Buy 4 REINFORCE Samples, Get a Baseline for Free!* In *Deep Reinforcement Learning Workshop, NeurIPS 2019*.
9. **Ahmadian, A., Cremer, C., Gallego, M., Perez, J., & Fadaee, S. (2024).** *Back to Basics: Revisiting REINFORCE Style Optimization for RLHF.* arXiv preprint arXiv:2402.14740.
10. **Luce, R. D. (1959).** *Individual Choice Behavior: A Theoretical Analysis.* John Wiley & Sons.
11. **Plackett, R. L. (1975).** *The Analysis of Permutations.* *Applied Statistics*, 24(2), 193–202.

### Reasoning Benchmarks & In-Context Learning
12. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2022).** *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* In *Advances in Neural Information Processing Systems (NeurIPS 2022)*, 35, 24824–24837.
13. **Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021).** *Training Verifiers to Solve Math Word Problems.* arXiv preprint arXiv:2110.14168 (GSM8K).
14. **Patel, A., Bhattamishra, S., & Goyal, N. (2021).** *Are NLP Models really able to Solve Simple Math Word Problems?* In *Proceedings of NAACL-HLT 2021*, pages 2080–2094 (SVAMP).
15. **Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O. (2018).** *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.* arXiv preprint arXiv:1803.05457.
16. **Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021).** *Measuring Mathematical Problem Solving with the MATH Dataset.* In *NeurIPS 2021 Datasets and Benchmarks Track*.
17. **Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., & Zettlemoyer, L. (2022).** *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* In *Proceedings of EMNLP 2022*, pages 11048–11064.
18. **Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020).** *Language Models are Few-Shot Learners.* In *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 1877–1901.
19. **Li, Y., Bubeck, S., Eldan, R., Del Giorno, A., Gunasekar, S., & Lee, Y. T. (2023).** *Textbooks Are All You Need II: phi-1.5 technical report.* arXiv preprint arXiv:2309.05463.
20. **Javaheripi, M., & Bubeck, S. (2023).** *Phi-2: The surprising power of small language models.* Microsoft Research Blog.
