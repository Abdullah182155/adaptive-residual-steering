# Methodology & Algorithmic Formulation

## 1. SteerNet: K-Conditioned Low-Rank Residual Correction

For candidate layer $l$ the correction is a low-rank bottleneck MLP:

$$\tilde{x} = \text{Dropout}(\text{GELU}(W_{\text{down}} \cdot \text{LN}(h))),\quad \tilde{x}_K = \tilde{x} + W_K\left(\tfrac{K}{4}\right)$$

$$\Delta h = W_{\text{up}}\left(\tilde{x}_K + \text{Dropout}(\text{GELU}(W_{\text{mid}}\tilde{x}_K))\right)\cdot\tfrac{\alpha_{\text{lora}}}{r}$$

$W_K$ is zero-initialised so that K-information starts as a no-op; $\alpha_{\text{lora}}/r = 2.0$.  The injected correction is divided by $\sqrt{K}$ so total steering energy stays bounded as more layers fire.

## 2. Usefulness Gate: Per-Token Activation

$$f_t = [\text{LN}(h_t), \text{LN}(\Delta h_t^{\text{detach}}), \|\Delta h_t\|_2, \cos(h_t,\Delta h_t), c]$$

$$\alpha_t = \sigma(W_2\,\text{GELU}(W_1 f_t + b_1) + b_2)$$

- $\Delta h$ is detached: the gate never back-propagates into SteerNet.
- **Initialisation:** $W_2 = 0$ and the output bias $b_2 = 2$, so $\alpha_{\text{init}} = \sigma(2) \approx 0.88$.  The gate starts nearly open, preserving the already-trained SteerNet correction, and must *learn* to close on tokens where steering does not help.
- **Dominance hinge (S3):** the gate is trained directly on the task loss; a hinge term penalises any configuration in which the gated model is worse than the ungated (const-gate) model, so the learned gate is only allowed to improve on passthrough.  Deployment re-checks this on the selection split (Section 5).

## 3. Catalog Router: Distillation over Counterfactual Subsets

Layer selection is a decision over a fixed **catalog** of subsets $\mathcal{C}$ (the empty subset is a legal action).

- **Counterfactual table (S0):** for every prompt and every $S \in \mathcal{C}$, the answer NLL under steering with subset $S$ is measured with the gate held constant.
- **Distillation (S2):** the router is trained to reproduce the best-subset distribution implied by $R(S) = -\text{NLL}_{\text{answer}}(S) - \lambda_{\text{cost}}|S|$, rather than by sampled policy gradients.
- **Question-span pooling:** router features are pooled only over the question span (prompt tokens, via the router context mask), so the decision is made before any answer token exists and is identical during prefill and cached decode.
- **Abstention:** a threshold $\theta_{\text{abs}}$ maps low-benefit predictions to $S=\emptyset$; $\theta=1.01$ disables abstention.

## 4. Failure Policy and Guards

Per injected token the relative norm $\|\delta_{\text{injected}}\|/\|h\|$ is monitored against `rel_norm_cap`.  Non-finite values, a cap violation, or more than `k_max` active layers trip the guard; a tripped item reverts to $K=0$ (the unsteered generation).  "Raw ARS" runs with guards and abstention disabled for ablation.

## 5. Deployment Calibration (S5)

All deployment hyper-parameters are fixed on a dedicated selection split ($n=120$) and frozen in `deployment_contract.json`:

1. $s^*_{\text{fixed}}$: lowest paired NLL subset; candidates within `calib_nll_tie_eps` are decided by greedy accuracy, then smaller $K$.
2. $\theta^*_{\text{abs}}$: grid search maximising paired NLL gain over baseline (ties: accuracy, then larger $\theta$).
3. Gate adoption: `learned` iff mean paired NLL diff $<0$ **and** gated accuracy $\ge$ ungated $- \text{tol}/n$; otherwise `passthrough`.
4. Envelope: quantiles of the relative injected norm under the learned routes/gates; `W_ENVELOPE_MISMATCH` is raised if the measured cap differs from 0.02 by more than $2\times$.

## 6. Evaluation Protocol (5 arms)

Baseline, Fixed-control, Router-only, Full ARS, Raw ARS on GSM8K test ($n=1319$), 0- and 8-shot.  Paired exact McNemar tests, paired bootstrap 95% CIs ($B=10{,}000$) and Holm-Bonferroni correction over the four treatments.
