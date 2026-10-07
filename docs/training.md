# Training & Staged Execution Pipeline

## Staged Curriculum
Because joint end-to-end gradient updates across SteerNet, Gate, and Router lead to destabilization, ARS utilizes a carefully staged curriculum:

### Phase 1: SteerNet Alone (Fixed Full Steering)
- **Gate:** Pinned at $\alpha = 1.0$.
- **Router:** Fixed single layer or uniform candidate subset.
- **Objective:** Standard cross-entropy task loss on GSM8K CoT trajectories.
- **Goal:** Learn a high-magnitude, valid arithmetic correction signal before introducing selectivity.

### Phase 2: Staged Refinement & Decomposition
1. **Phase 2A (Router Bootstrap):**
   - SteerNet & Gate are frozen.
   - Router policy is trained via counterfactual subset forwards under RLOO to discover layer synergies and identify when steering hurts.
2. **Phase 2B (SteerNet Fine-Tuning):**
   - Fine-tune SteerNet under the newly learned routing distributions with paraphrase invariance loss.
3. **Phase 2C (Gate Tuning):**
   - SteerNet and Router are frozen.
   - Gate is trained using auxiliary objectives (Policy Gradient / L1 sparsity / Contrastive token loss / Anti-saturation).
4. **Phase 2D (Joint Low-LR Fine-Tuning):**
   - Extremely low learning rate joint pass with accuracy-based early stopping.

## Evaluation Protocol
- **Stage A:** Frozen Base Model Baseline.
- **Stage B:** SteerNet Only (Phase 1 weights, Gate=1.0, single/fixed layer).
- **Stage C:** Full ARS (Trained SteerNet + Dynamic Router + Token Gate).
- **Stage D / Diagnostics (2x2 Grid):**
  - {Router: Fixed vs Dynamic} $\times$ {Gate: 1.0 vs Trained Live}.
  - Isolates whether performance regressions stem from router layer selection or gate token suppression.
