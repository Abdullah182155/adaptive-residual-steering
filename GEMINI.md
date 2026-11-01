# 🚀 Adaptive Residual Steering (ARS) — Project Instructions & Guidelines

## 1. Project Mission & Objectives
- **Core Mission:** Develop, validate, and optimize the **Adaptive Residual Steering (ARS)** architecture on `microsoft/phi-2` (2.7B parameters, 32 transformer layers, $d=2560$).
- **Success Criteria:** Outperform the baseline backbone by **$\ge +15.00\text{ pp}$ accuracy** on GSM8K across all shot evaluations (0-shot, 2-shot, 4-shot) and generalize to unseen reasoning benchmarks.
- **Current Phase:** **Phase 1: SteerNet-Only (Fixed Layers)**.
  - Candidate catalog layers: `[13, 15, 18, 20, 22]`.
  - Proven 3-layer synergistic subset: `s*_fixed = [13, 15, 22]` (Early L13 + Mid L15 + Late Anchor L22).
  - Router is DISABLED (Phase 2).
  - Gate is PASSTHROUGH / CONSTANT (Phase 3).

---

## 2. Hardware & Platform Constraints
- **Target Platform:** Kaggle Execution Environment (Single/Dual NVIDIA Tesla T4, 16 GB VRAM).
- **VRAM Budget:** Maximum active allocation $\le 12.0\text{ GB}$ (leaving $\ge 4.0\text{ GB}$ headroom).
- **Batch Size:** Micro-batch size $\le 4$ with gradient accumulation to keep activation footprint low.
- **Pipeline Runtime:** Total pipeline execution (Norm Calibration $\to$ S1 Training $\to$ S5 Calibration $\to$ Benchmark) must strictly complete in **$\le 20$ minutes**.

---

## 3. Steering Physics & Mathematical Principles
1. **SteerNet V5 Architecture (`src/steering/steernet_v5.py`):**
   - Rank-64 LoRA bottleneck with temporal depthwise causal convolution (`CausalDWConv1d`, kernel size 4).
   - **Dirac-Centered Identity Initialization:** $w[:, :, -1] = 1.0$ (current token tap), $w[:, :, :-1] \sim \mathcal{N}(0, 0.01^2)$ (past taps). Ensures pure identity signal flow at step 0 without attenuation.
   - **Per-Layer Hidden Norm Clamping:** Norm cap $C_l = \kappa \cdot N_l$ calibrated to median hidden layer norm $N_l$ with relative cap $\kappa = 0.08$.
   - **Warm Cache Hand-off:** Conv cache is primed with trailing prompt tokens during prefill so auto-regressive decode starts with true context.
2. **Shot-Aware Steering Energy:**
   - 0-shot starts cold without demonstrations $\to$ Requires full steering guidance ($\text{scale} = 1.0$).
   - 2-shot has 2 in-context exemplars $\to$ Requires gentle guidance ($\text{scale} = 0.50$).
   - 4-shot has strong in-context exemplar alignment $\to$ Requires subtle nudge ($\text{scale} = 0.35$).
3. **Prompt Shielding Dynamics:**
   - In 0-shot: The math question MUST be steered during prefill to prime the KV-cache into mathematical reasoning mode.
   - In few-shot: Demonstrations ($0 \dots \text{last\_match}-1$) are strictly shielded via `prompt_shield_mask` ($\Delta = 0$) to keep exemplar keys and values 100% pristine.
4. **S5 Deployment Calibration (`src/deploy/calibrate.py`):**
   - In Phase 1, `measure_envelope` must evaluate under the active fixed subset `s*_fixed = [13, 15, 22]` so the relative norm cap reflects real steering vectors ($\sim 0.12 - 0.15$) and eliminates false guard tripping.

---

## 4. Engineering Standards & Code Quality
- **Language Policy:**
  - **Zero Arabic in code:** All code, inline comments, docstrings, test names, logs, file names, and git commit messages MUST be strictly in English.
  - Arabic is used strictly for conversational pair-programming responses to the user.
- **Testing & Verification:**
  - **100% Green Test Suite:** All unit and contract tests in `tests/` must pass before pushing (`pytest tests/ -q -m "not slow"`).
  - Any architectural modification must update the cache via `python -c "from scripts.check_architecture_diagrams import compute_arch_hashes, get_repo_root, save_cached_record; r = get_repo_root(); save_cached_record('diagrams/.arch_hash', compute_arch_hashes(r))"`.
- **No Band-Aids / Placebos:**
  - Every modification must be mathematically justified, modular, and address root causes. Never introduce arbitrary hacks or bypass safety contracts.
- **Git Workflow:**
  - Active branch is `development`.
  - Follow Conventional Commits (`feat(...)`, `fix(...)`, `test(...)`, `perf(...)`, `refactor(...)`).
