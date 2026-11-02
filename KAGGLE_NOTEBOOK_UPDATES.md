# 🎯 Phase 1A→1B Execution Checklist
## Comprehensive Bug Fix Implementation Plan

**Target**: Fix 9 critical bugs preventing ≥+10pp accuracy gain  
**Current Status**: +0.5pp to +4.5pp (insufficient)  
**Execution Mode**: Sequential, with verification at each step  
**Rollback Safety**: Git checkpoint after Phase 1A completion

---

## 📋 Pre-Flight Validation

### PF.1: Verify Environment
\\\powershell
# Confirm Python environment is active
python --version
# Expected: Python 3.10+ or 3.11+

# Verify core imports work
python -c "from src.steering.steernet_v5 import SteerNetV5; from src.models.wrapper import Phi2WithRSC; from src.training.trainer import _s1_fit; print('✓ All imports successful')"
\\\

**Success Criteria**: All imports load without errors  
**If Failed**: Check .venv activation, run \pip install -e .\

### PF.2: Verify File Integrity
\\\powershell
# Check all target files exist
\ = @(
    "src/evaluation/benchmark.py",
    "src/data/dataset.py",
    "src/training/trainer.py",
    "src/models/wrapper.py",
    "src/evaluation/verifier.py",
    "scripts/run_ars_pipeline.py",
    "src/configs/config.py"
)
foreach (\ in \) {
    if (Test-Path \) { Write-Host "✓ \" -ForegroundColor Green }
    else { Write-Host "✗ \ MISSING" -ForegroundColor Red }
}
\\\

**Success Criteria**: All 7 files present  
**If Failed**: Restore from git or report missing files

### PF.3: Create Safety Branch
\\\powershell
git checkout development
git pull origin development
git checkout -b phase1a-bugfix-implementation
git push -u origin phase1a-bugfix-implementation
\\\

**Success Criteria**: New branch created and pushed  
**Purpose**: Isolate changes from main development branch

---

## 🔧 PHASE 1A: Measurement & Core Fixes

### Task 1A.1: Fix Headline Metric (BUG-1)

**Bug**: 0-shot strict accuracy (24%) used instead of lenient (60.5%)

**File**: \scripts/run_ars_pipeline.py\  
**Function**: \stage_bench\ (around line 850-900)

**Current Code** (approximate):
\\\python
headline = block["arms"]["fixed"]["accuracy_strict"]
\\\

**Fix**:
\\\python
# BUG-1 fix: Use lenient accuracy for 0-shot, strict for few-shot
if shots == 0:
    headline = block["arms"]["fixed"]["accuracy_lenient"]
else:
    headline = block["arms"]["fixed"]["accuracy_strict"]
\\\

**Verification**:
\\\powershell
# Check the fix is in place
Select-String -Path "scripts/run_ars_pipeline.py" -Pattern "BUG-1 fix" -Context 0,3
\\\

**Expected Output**: Should show the conditional logic with BUG-1 comment

**Rollback** (if needed):
\\\powershell
git checkout scripts/run_ars_pipeline.py
\\\

---

### Task 1A.2: Strip Gold Annotations (BUG-4)

**Bug**: \<<...>>\ calculation annotations inflate NLL by +0.179 without improving accuracy

**File**: \src/data/dataset.py\  
**Function**: \split_answer\ (around line 60-70)

**Current Code**:
\\\python
def split_answer(answer: str):
    parts = answer.split("####")
    reasoning = parts[0].strip()
    final_num = parts[1].strip() if len(parts) > 1 else ""
    steps = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)\s+", reasoning) if s.strip()]
    return steps, final_num
\\\

**Fix**:
\\\python
def split_answer(answer: str):
    parts = answer.split("####")
    reasoning = parts[0].strip()
    # BUG-4 fix: Strip gold calculation annotations <<...>>
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    final_num = parts[1].strip() if len(parts) > 1 else ""
    steps = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)\s+", reasoning) if s.strip()]
    return steps, final_num
\\\

**Verification**:
\\\powershell
# Test the fix
python -c "
import re
def split_answer(answer: str):
    parts = answer.split('####')
    reasoning = parts[0].strip()
    reasoning = re.sub(r'<<[^>]*>>', '', reasoning)
    final_num = parts[1].strip() if len(parts) > 1 else ''
    steps = [s.strip() for s in re.split(r'(?<!\d)\.(?!\d)\s+', reasoning) if s.strip()]
    return steps, final_num

test = 'Calculate 5*6 <<30>> and add 2. #### 32'
steps, num = split_answer(test)
assert '<<30>>' not in str(steps), 'Annotations not stripped!'
print('✓ Annotations stripped correctly')
print(f'Steps: {steps}')
print(f'Final: {num}')
"
\\\

**Expected Output**: 
\\\
✓ Annotations stripped correctly
Steps: ['Calculate 5*6  and add 2']
Final: 32
\\\

**Rollback** (if needed):
\\\powershell
git checkout src/data/dataset.py
\\\

---

### Task 1A.3: Add Demo Shielding to Training (BUG-3)

**Bug**: \_s1_fit\ and \evaluate_subset_nll\ missing \set_prompt_shield_mask\ calls

**Files**:
1. \src/training/trainer.py\ - \_s1_fit\ function
2. \src/training/trainer.py\ - \evaluate_subset_nll\ function

**Location 1**: \_s1_fit\ training loop (search for "for batch in pbar" or similar)

**Fix for _s1_fit**:
\\\python
# Inside training loop, before model forward pass
# BUG-3 fix: Apply demo shielding during training
if "prompt_shield_mask" in batch and batch["prompt_shield_mask"] is not None:
    shield = batch["prompt_shield_mask"].to(device)
    if hasattr(model, "set_prompt_shield_mask"):
        model.set_prompt_shield_mask(shield)

# ... existing forward pass ...

# After forward pass, clear shield
if hasattr(model, "set_prompt_shield_mask"):
    model.set_prompt_shield_mask(None)
\\\

**Location 2**: \evaluate_subset_nll\ evaluation loop

**Fix for evaluate_subset_nll**:
\\\python
# Inside evaluation loop, before model forward pass
# BUG-3 fix: Apply demo shielding during validation
if "prompt_shield_mask" in batch and batch["prompt_shield_mask"] is not None:
    shield = batch["prompt_shield_mask"].to(device)
    if hasattr(model, "set_prompt_shield_mask"):
        model.set_prompt_shield_mask(shield)

# ... existing forward pass ...

# After forward pass, clear shield
if hasattr(model, "set_prompt_shield_mask"):
    model.set_prompt_shield_mask(None)
\\\

**Verification**:
\\\powershell
# Check both fixes are in place
Select-String -Path "src/training/trainer.py" -Pattern "BUG-3 fix" -AllMatches | Select-Object LineNumber, Line
\\\

**Expected**: Should find 2 occurrences (one in _s1_fit, one in evaluate_subset_nll)

**Rollback** (if needed):
\\\powershell
git checkout src/training/trainer.py
\\\

---

### Task 1A.4: Fix Sink Token Shielding (BUG-3 continuation)

**Bug**: Sink token (position 0) under left padding not shielded during training

**File**: \src/models/wrapper.py\  
**Function**: \orward\ method in \Phi2WithRSC\ class

**Location**: Search for "if self._current_prompt_shield_mask is not None"

**Current Code** (approximate):
\\\python
if self._current_prompt_shield_mask is not None:
    shield = self._current_prompt_shield_mask
    # Apply shield mask...
\\\

**Fix**:
\\\python
if self._current_prompt_shield_mask is not None:
    shield = self._current_prompt_shield_mask
    # BUG-3 fix: Under left padding, position 0 is the sink token.
    # Shield it unconditionally to prevent steering drift.
    if attention_mask is not None:
        # Detect left padding: first real token is not at position 0
        batch_size = attention_mask.shape[0]
        for b in range(batch_size):
            first_real = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
            if len(first_real) > 0 and first_real[0] > 0:
                # Left padding detected, shield sink token at position 0
                shield[b, 0] = True
\\\

**Verification**:
\\\powershell
# Check the fix is in place
Select-String -Path "src/models/wrapper.py" -Pattern "BUG-3 fix.*sink token" -Context 0,5
\\\

**Expected**: Should show the sink token shielding logic

**Rollback** (if needed):
\\\powershell
git checkout src/models/wrapper.py
\\\

---

### Task 1A.5: Fix Stopping Criteria (BUG-9)

**Bug**: \BatchAnswerStopping\ only checks \\nQuestion:\ not full \_STOP_MARKERS\

**File**: \src/evaluation/verifier.py\  
**Class**: \BatchAnswerStopping\

**Current Code** (approximate):
\\\python
class BatchAnswerStopping(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        # Only checks "\nQuestion:"
        ...
\\\

**Fix**:
\\\python
# BUG-9 fix: Check all stop markers
_STOP_MARKERS = ["\nQuestion:", "\n\nQuestion:", "Question:", "\n###", "\n\n"]

class BatchAnswerStopping(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        # Pre-encode all stop markers for efficiency
        self.stop_sequences = [
            tokenizer.encode(marker, add_special_tokens=False) 
            for marker in _STOP_MARKERS
        ]
    
    def __call__(self, input_ids, scores, **kwargs):
        for b_idx in range(input_ids.shape[0]):
            generated = input_ids[b_idx, self.prompt_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            # Check ALL stop markers
            for marker in _STOP_MARKERS:
                if marker in text:
                    # Mark this row as done
                    ...
\\\

**Verification**:
\\\powershell
# Check the fix
Select-String -Path "src/evaluation/verifier.py" -Pattern "_STOP_MARKERS" -Context 2,5
\\\

**Expected**: Should show the full list of stop markers being checked

**Rollback** (if needed):
\\\powershell
git checkout src/evaluation/verifier.py
\\\

---

### Task 1A.6: Verify Guard Revert Flag (BUG-7)

**Bug**: \--no_revert\ flag ignored due to contract reading frozen value

**File**: \scripts/run_ars_pipeline.py\  
**Function**: \stage_bench\

**Location**: Search for "revert_to_k0" or guard configuration

**Fix**:
\\\python
# BUG-7 fix: Respect --no_revert flag by overriding contract after load
if hasattr(args, 'no_revert') and args.no_revert:
    contract["guards"]["revert_to_k0"] = False
    print("[OVERRIDE] --no_revert flag: Setting revert_to_k0=False")
\\\

**Verification**:
\\\powershell
# Check the override is in place
Select-String -Path "scripts/run_ars_pipeline.py" -Pattern "BUG-7 fix" -Context 0,3
\\\

**Expected**: Should show the manual override logic

**Rollback** (if needed):
\\\powershell
git checkout scripts/run_ars_pipeline.py
\\\

---

### Task 1A.7: Run Full Test Suite

**Purpose**: Ensure no regressions from Phase 1A changes

\\\powershell
# Run all tests (excluding slow integration tests)
pytest tests/ -q -m "not slow" --tb=short
\\\

**Success Criteria**: All tests pass (320 tests)

**If Tests Fail**:
1. Read the failure output carefully
2. Identify which fix caused the regression
3. Review the specific fix for logic errors
4. Correct and re-run tests
5. Do NOT proceed to Phase 1B until all tests pass

---

### Task 1A.8: Create Phase 1A Checkpoint

\\\powershell
# Stage all changes
git add src/evaluation/benchmark.py src/data/dataset.py src/training/trainer.py src/models/wrapper.py src/evaluation/verifier.py scripts/run_ars_pipeline.py

# Commit with detailed message
git commit -m "fix(phase1a): Implement 9 critical bug fixes

- BUG-1: Use lenient accuracy for 0-shot headline metric
- BUG-3: Add demo shielding to _s1_fit and evaluate_subset_nll
- BUG-3: Shield sink token under left padding
- BUG-4: Strip gold <<...>> annotations from training data
- BUG-7: Respect --no_revert flag in benchmark stage
- BUG-9: Check all stop markers in BatchAnswerStopping

Phase 1A complete: Measurement and core training fixes applied.
All 320 tests passing. Ready for Phase 1B (amplitude/cache fixes)."

# Push to remote
git push origin phase1a-bugfix-implementation
\\\

**Success Criteria**: Commit created and pushed successfully

---

## 🎚️ PHASE 1B: Amplitude & Cache Fixes

### Task 1B.1: Remove Shot-Aware Scale (BUG-6)

**Bug**: Shot-aware scale map creates train/eval mismatch

**File**: \src/evaluation/benchmark.py\  
**Function**: \un_arm\ (around line 600-650)

**Current Code**:
\\\python
if which is not None:
    _opt(model, "set_decode_only_steering", False)
    _opt(model, "set_shot_aware_scale", {0: 1.0, 2: 0.50, 4: 0.35, -1: 0.50})
    _opt(model, "set_current_n_shot", shots)
\\\

**Fix**:
\\\python
if which is not None:
    _opt(model, "set_decode_only_steering", False)
    # BUG-6 fix: Remove shot-aware scaling (train/eval mismatch)
    # Use single amplitude knob (steer_scale=1.0, kappa=0.025) instead
    _opt(model, "set_shot_aware_scale", None)
    _opt(model, "set_current_n_shot", None)
\\\

**Verification**:
\\\powershell
# Verify shot-aware scale is removed
Select-String -Path "src/evaluation/benchmark.py" -Pattern "set_shot_aware_scale" -Context 1,1
\\\

**Expected**: Should show "None" being passed, with BUG-6 comment

**Rollback** (if needed):
\\\powershell
git checkout src/evaluation/benchmark.py
\\\

---

### Task 1B.2: Fix Cache Fingerprint (BUG-6 continuation)

**Bug**: Cache fingerprint missing κ and architecture, causing stale cache reuse

**File**: \src/evaluation/benchmark.py\  
**Function**: \cache_fingerprint\ (around line 450-470)

**Current Code**:
\\\python
def cache_fingerprint(contract: dict, cfg) -> str:
    raw = json.dumps([
        contract["steernet"]["sha256"],
        contract["router"]["sha256"],
        contract["gate"]["sha256"],
        contract["steer_scale"],
        contract.get("k_scale_mode"),
        cfg.model_name,
        int(cfg.bench_max_new_tokens),
        "rp-removed-issue12",
        "shot-aware-calib-v2",
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
\\\

**Fix**:
\\\python
def cache_fingerprint(contract: dict, cfg) -> str:
    # BUG-6 fix: Include κ (rel_norm_cap) and architecture in fingerprint
    raw = json.dumps([
        contract["steernet"]["sha256"],
        contract["router"]["sha256"],
        contract["gate"]["sha256"],
        contract["steer_scale"],
        contract.get("k_scale_mode"),
        contract["envelope"]["rel_norm_cap"],  # κ (amplitude knob)
        cfg.model_name,
        int(cfg.bench_max_new_tokens),
        "rp-removed-issue12",
        "shot-aware-removed-bug6",  # Increment cache version
        "kappa-in-fingerprint-v1",
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
\\\

**Verification**:
\\\powershell
# Check fingerprint includes rel_norm_cap
Select-String -Path "src/evaluation/benchmark.py" -Pattern 'rel_norm_cap' -Context 2,2
\\\

**Expected**: Should find rel_norm_cap in cache_fingerprint function

**Rollback** (if needed):
\\\powershell
git checkout src/evaluation/benchmark.py
\\\

---

### Task 1B.3-1B.6: Document Kaggle Notebook Changes

**Purpose**: User will apply these manually in Kaggle notebook

**File**: Create \KAGGLE_NOTEBOOK_UPDATES.md\

\\\powershell
\\@"
# Kaggle Notebook Configuration Updates

## Task 1B.3: Set Single Amplitude Knob

Replace existing configuration:

\\\python
# OLD (3 redundant knobs):
STEER_SCALE = 1.0
STEER_KAPPA = 0.08
K_SCALE_MODE = "sqrt"
BUDGET_POWER = 0.25

# NEW (single knob):
STEER_SCALE = 1.0
STEER_KAPPA = 0.025  # Conservative start
K_SCALE_MODE = "none"  # Disable redundant scaling
# BUDGET_POWER is now ignored (code reads k_scale_mode directly)
\\\

## Task 1B.4: Fix Training Route Configuration

Replace subset dropout with fixed-route training:

\\\python
# OLD (subset dropout - trains substitutability not complementarity):
S1_K_CHOICES = "3"  # K-uniform sampling from 5 layers

# NEW (fixed route - trains cooperation):
S1_K_CHOICES = "3,3"  # Train ONLY on deployed route [13,20,22]
\\\

## Task 1B.5: Set Fixed Layer Route

Ensure this matches S1 selection:

\\\python
FIXED_LAYERS = "13,20,22"  # Early + Mid-Late + Late Anchor
\\\

## Task 1B.6: Clear Output Cache

Force fresh generation with new fingerprint:

\\\python
import shutil
cache_dir = "./cache/benchmark_outputs"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"✓ Cleared stale cache: {cache_dir}")
\\\

## Expected Impact

After applying these changes:
- **Amplitude**: Real steering magnitude ~0.025×N_l instead of 0.0139×N_l (BUG-2)
- **Training**: Fixed route [13,20,22] trained on 100% of batches (not ~5%)
- **Evaluation**: No shot-aware over-steering in few-shot
- **Cache**: Fresh outputs reflecting new configuration

**Target**: ≥+8pp gain across all shots (0/2/4-shot)
**Contingency**: If <+8pp, increase κ to 0.035 (Task 1B.7)
