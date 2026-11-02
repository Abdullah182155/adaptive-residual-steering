# 🔍 Phase 1A→1B Implementation Review

**Review Date**: November 2, 2026  
**Reviewer**: Kiro AI  
**Agent**: Devin (GitHub integration)  
**Branch**: main (should be phase1a-bugfix-implementation)  
**Status**: ⚠️ **PARTIALLY COMPLETE - CRITICAL ISSUES FOUND**

---

## ✅ **COMPLETED TASKS**

### Phase 1A: Core Fixes

| Task | File | Status | Notes |
|------|------|--------|-------|
| **1A.2** | `src/data/dataset.py` | ✅ **DONE** | Annotation stripping implemented correctly (`re.sub(r"<<[^>]*>>", "", reasoning)`) |
| **1A.3** | `src/training/trainer.py` | ✅ **DONE** | Demo shielding added to both `_s1_fit` and `evaluate_subset_nll` with proper `_demo_shield()` helper |
| **1A.5** | `src/evaluation/verifier.py` | ✅ **DONE** | Stop markers check updated to use `_STOP_MARKERS` array |

**Quality Assessment**: These 3 fixes are **correctly implemented** and follow the specification exactly.

---

## ❌ **MISSING / INCOMPLETE TASKS**

### Critical Issues

#### ❌ **Task 1A.1: Fix Headline Metric (BUG-1) - NOT FOUND**
- **File**: `scripts/run_ars_pipeline.py`
- **Expected**: Conditional logic to use lenient accuracy for 0-shot, strict for few-shot
- **Status**: File appears corrupted or empty - cannot verify
- **Impact**: **HIGH** - Headline metrics will still show incorrect 24% instead of 60.5% for 0-shot

#### ❌ **Task 1A.4: Fix Sink Token Shielding (BUG-3) - NOT VERIFIED**
- **File**: `src/models/wrapper.py`
- **Expected**: Shield position 0 under left padding
- **Status**: Cannot find sink token shielding logic in the reviewed section
- **Impact**: **MEDIUM** - Sink token may still cause steering drift under left padding

#### ❌ **Task 1A.6: Verify Guard Revert Flag (BUG-7) - NOT VERIFIED**
- **File**: `scripts/run_ars_pipeline.py`
- **Expected**: Override `contract["guards"]["revert_to_k0"]` when `--no_revert` flag set
- **Status**: File appears corrupted - cannot verify
- **Impact**: **MEDIUM** - `--no_revert` flag may still be ignored

#### ❌ **Task 1A.7: Run Test Suite - NOT EXECUTED**
- **Command**: `pytest tests/ -q -m "not slow"`
- **Status**: Test command failed to execute in PowerShell
- **Impact**: **HIGH** - No verification that changes don't break existing functionality

#### ❌ **Task 1A.8: Git Checkpoint - NOT CREATED**
- **Expected**: Commit on branch `phase1a-bugfix-implementation`
- **Actual**: Changes are on `main` branch, not staged, not committed
- **Impact**: **CRITICAL** - No rollback safety, working directly on main

### Phase 1B Issues

#### ❌ **Task 1B.1: Remove Shot-Aware Scale (BUG-6) - NOT DONE**
- **File**: `src/evaluation/benchmark.py` line 430
- **Current**: `_opt(model, "set_shot_aware_scale", {0: 1.0, 2: 0.50, 4: 0.35, -1: 0.50})`
- **Expected**: `_opt(model, "set_shot_aware_scale", None)`
- **Status**: ⚠️ **STILL HAS SHOT-AWARE SCALING** - Creates train/eval mismatch
- **Impact**: **CRITICAL** - Will cause over-steering in few-shot scenarios

#### ❌ **Task 1B.2: Fix Cache Fingerprint (BUG-6) - NOT DONE**
- **File**: `src/evaluation/benchmark.py` line 596-608
- **Current**: Missing `contract["envelope"]["rel_norm_cap"]` in fingerprint
- **Expected**: Should include κ in cache key
- **Status**: ⚠️ **STALE CACHE WILL BE REUSED** - Wrong results
- **Impact**: **CRITICAL** - Benchmark will use cached results from old configuration

#### ⚠️ **Task 1B.3-1B.6: Kaggle Documentation - WRONG FILE**
- **File**: Created `KAGGLE_NOTEBOOK_UPDATES.md`
- **Content**: Contains the Phase 1A→1B **CHECKLIST**, not the **KAGGLE CONFIGURATION UPDATES**
- **Status**: ⚠️ **WRONG CONTENT** - User needs actual configuration instructions
- **Impact**: **HIGH** - User cannot apply notebook changes without proper documentation

---

## 🚨 **CRITICAL PROBLEMS**

### Problem 1: Git Workflow Violation
- **Issue**: Working directly on `main` branch instead of `phase1a-bugfix-implementation`
- **Risk**: Changes not isolated, no clean rollback path
- **Fix Required**: Create proper branch and commit

### Problem 2: File Corruption
- **Issue**: `scripts/run_ars_pipeline.py` appears empty/corrupted
- **Risk**: Cannot verify BUG-1 and BUG-7 fixes
- **Fix Required**: Restore file and implement missing fixes

### Problem 3: Phase 1B Not Started
- **Issue**: Shot-aware scaling still active, cache fingerprint not fixed
- **Risk**: Benchmark results will be **INVALID** due to train/eval mismatch and stale cache
- **Fix Required**: Complete Task 1B.1 and 1B.2 **BEFORE** running pipeline

### Problem 4: No Test Verification
- **Issue**: No confirmation that 320 tests still pass
- **Risk**: Unknown regressions, broken functionality
- **Fix Required**: Run full test suite and fix any failures

---

## 📊 **IMPLEMENTATION SCORECARD**

| Phase | Tasks | Completed | Missing | Score |
|-------|-------|-----------|---------|-------|
| **Pre-Flight** | 3 | 0 | 3 | 0% |
| **Phase 1A** | 8 | 3 | 5 | 38% |
| **Phase 1B** | 4 | 0 | 4 | 0% |
| **TOTAL** | 15 | 3 | 12 | **20%** |

**Overall Status**: ⚠️ **INCOMPLETE AND RISKY**

---

## 🔧 **IMMEDIATE ACTION REQUIRED**

### Step 1: Fix Git Workflow ⚠️ **URGENT**

```powershell
# Stash current changes
git stash

# Create proper branch from development
git checkout development
git pull origin development
git checkout -b phase1a-bugfix-implementation

# Restore changes
git stash pop

# Verify we're on the right branch
git branch  # Should show * phase1a-bugfix-implementation
```

### Step 2: Fix `scripts/run_ars_pipeline.py` ⚠️ **URGENT**

The file appears corrupted. Need to:
1. Check if file exists and has content
2. If corrupted, restore from git: `git checkout HEAD scripts/run_ars_pipeline.py`
3. Re-apply BUG-1 and BUG-7 fixes

### Step 3: Complete Phase 1A Missing Tasks

#### Task 1A.1: Fix Headline Metric
**File**: `scripts/run_ars_pipeline.py`  
**Location**: Function `stage_bench` (search for `block["arms"]["fixed"]["accuracy_strict"]`)

**Add this code**:
```python
# BUG-1 fix: Use lenient accuracy for 0-shot, strict for few-shot
if shots == 0:
    headline = block["arms"]["fixed"]["accuracy_lenient"]
else:
    headline = block["arms"]["fixed"]["accuracy_strict"]
```

#### Task 1A.4: Fix Sink Token Shielding
**File**: `src/models/wrapper.py`  
**Location**: Search for `if self._current_prompt_shield_mask is not None`

**Add after shield assignment**:
```python
# BUG-3 fix: Shield sink token at position 0 under left padding
if attention_mask is not None:
    batch_size = attention_mask.shape[0]
    for b in range(batch_size):
        first_real = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
        if len(first_real) > 0 and first_real[0] > 0:
            shield[b, 0] = True
```

#### Task 1A.6: Verify Guard Revert Flag
**File**: `scripts/run_ars_pipeline.py`  
**Location**: Function `stage_bench`, after contract is loaded

**Add this code**:
```python
# BUG-7 fix: Respect --no_revert flag
if hasattr(args, 'no_revert') and args.no_revert:
    contract["guards"]["revert_to_k0"] = False
    print("[OVERRIDE] --no_revert flag: revert_to_k0 = False")
```

### Step 4: Run Test Suite ⚠️ **MANDATORY**

```powershell
# Activate virtual environment if not already active
.\.venv\Scripts\Activate.ps1

# Run tests
pytest tests/ -q -m "not slow" --tb=short

# Expected: All 320 tests should pass
# If failures occur: STOP and fix them before proceeding
```

### Step 5: Create Phase 1A Checkpoint

```powershell
# Stage Phase 1A files
git add src/data/dataset.py
git add src/training/trainer.py
git add src/evaluation/verifier.py
git add src/models/wrapper.py
git add scripts/run_ars_pipeline.py

# Commit
git commit -m "fix(phase1a): Implement critical bug fixes for BUG-1,3,4,7,9

- BUG-1: Use lenient accuracy for 0-shot headline metric
- BUG-3: Add demo shielding to _s1_fit and evaluate_subset_nll
- BUG-3: Shield sink token under left padding
- BUG-4: Strip gold <<...>> annotations from training data
- BUG-7: Respect --no_revert flag in benchmark stage
- BUG-9: Check all stop markers in BatchAnswerStopping

Phase 1A complete. All tests passing. Ready for Phase 1B."

# Push
git push -u origin phase1a-bugfix-implementation
```

### Step 6: Complete Phase 1B Tasks ⚠️ **CRITICAL FOR ACCURACY**

#### Task 1B.1: Remove Shot-Aware Scale
**File**: `src/evaluation/benchmark.py` line 430

**Current**:
```python
_opt(model, "set_shot_aware_scale", {0: 1.0, 2: 0.50, 4: 0.35, -1: 0.50})
```

**Replace with**:
```python
# BUG-6 fix: Remove shot-aware scaling (train/eval mismatch)
_opt(model, "set_shot_aware_scale", None)
```

#### Task 1B.2: Fix Cache Fingerprint
**File**: `src/evaluation/benchmark.py` line 596-608

**Current**:
```python
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
```

**Replace with**:
```python
def cache_fingerprint(contract: dict, cfg) -> str:
    # BUG-6 fix: Include κ in fingerprint to prevent stale cache reuse
    raw = json.dumps([
        contract["steernet"]["sha256"],
        contract["router"]["sha256"],
        contract["gate"]["sha256"],
        contract["steer_scale"],
        contract.get("k_scale_mode"),
        contract["envelope"]["rel_norm_cap"],  # ADD THIS LINE
        cfg.model_name,
        int(cfg.bench_max_new_tokens),
        "rp-removed-issue12",
        "shot-aware-removed-bug6",  # INCREMENT VERSION
        "kappa-in-fingerprint-v1",   # ADD THIS LINE
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

### Step 7: Create Proper Kaggle Documentation

**Delete current KAGGLE_NOTEBOOK_UPDATES.md** (it has wrong content)

**Create new file** with proper configuration updates:

```powershell
# Create the correct Kaggle documentation
@"
# Kaggle Notebook Configuration Updates

Apply these changes to your Kaggle notebook BEFORE running the pipeline.

## 1. Single Amplitude Knob (BUG-2 fix)

Replace:
\`\`\`python
STEER_SCALE = 1.0
STEER_KAPPA = 0.08
K_SCALE_MODE = "sqrt"
BUDGET_POWER = 0.25
\`\`\`

With:
\`\`\`python
STEER_SCALE = 1.0
STEER_KAPPA = 0.025  # Conservative start (real amplitude 0.025×N_l)
K_SCALE_MODE = "none"  # Disable redundant sqrt scaling
# BUDGET_POWER no longer used
\`\`\`

## 2. Fixed-Route Training (BUG-5 fix)

Replace:
\`\`\`python
S1_K_CHOICES = "3"  # Subset dropout
\`\`\`

With:
\`\`\`python
S1_K_CHOICES = "3,3"  # Train ONLY on deployed route [13,20,22]
\`\`\`

## 3. Deployed Route Configuration

Ensure:
\`\`\`python
FIXED_LAYERS = "13,20,22"  # Early L13 + Mid-Late L20 + Late Anchor L22
\`\`\`

## 4. Clear Stale Cache

Add at the beginning of your notebook:
\`\`\`python
import shutil
cache_dir = "./cache/benchmark_outputs"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"✓ Cleared cache: {cache_dir}")
\`\`\`

## Expected Results

After these changes:
- 0-shot lenient: ≥68.5% (baseline 60.5% + 8pp target)
- 2-shot strict: ≥64.5% (baseline 56.5% + 8pp target)
- 4-shot strict: ≥69.5% (baseline 61.5% + 8pp target)

If results < +8pp, increase κ to 0.035 and re-run.
"@ | Out-File -FilePath "KAGGLE_NOTEBOOK_UPDATES.md" -Encoding utf8
```

### Step 8: Run Tests Again

```powershell
pytest tests/ -q -m "not slow" --tb=short
```

### Step 9: Create Phase 1B Checkpoint

```powershell
git add src/evaluation/benchmark.py
git add KAGGLE_NOTEBOOK_UPDATES.md
git commit -m "fix(phase1b): Remove shot-aware scale and fix cache fingerprint

- BUG-6: Remove shot-aware scaling to eliminate train/eval mismatch
- BUG-6: Add κ (rel_norm_cap) to cache fingerprint
- BUG-2: Document single amplitude knob configuration
- BUG-5: Document fixed-route training configuration

Phase 1B complete. All tests passing. Ready for pipeline execution."

git push origin phase1a-bugfix-implementation
```

---

## ⚡ **WHAT YOUR AGENT SHOULD DO NOW**

Your agent should execute the 9 steps above in order:

1. ✅ Fix git workflow (create proper branch)
2. ✅ Restore/fix `scripts/run_ars_pipeline.py`
3. ✅ Complete missing Phase 1A tasks (1A.1, 1A.4, 1A.6)
4. ✅ Run test suite and verify all pass
5. ✅ Create Phase 1A git checkpoint
6. ✅ Complete Phase 1B tasks (1B.1, 1B.2)
7. ✅ Create proper Kaggle documentation
8. ✅ Run test suite again
9. ✅ Create Phase 1B git checkpoint

**Do NOT run the pipeline until all 9 steps are complete.**

---

## 🎯 **SUCCESS CRITERIA**

Before marking implementation complete, verify:

- [ ] All changes on branch `phase1a-bugfix-implementation` (not `main`)
- [ ] All 6 files modified: dataset.py, trainer.py, verifier.py, wrapper.py, run_ars_pipeline.py, benchmark.py
- [ ] Shot-aware scale set to `None` in benchmark.py line 430
- [ ] Cache fingerprint includes `rel_norm_cap`
- [ ] KAGGLE_NOTEBOOK_UPDATES.md has configuration instructions (not checklist)
- [ ] All 320 tests passing
- [ ] 2 git commits created (Phase 1A + Phase 1B)
- [ ] Branch pushed to remote

Only after ALL criteria met → User applies Kaggle changes → Run pipeline → Expect ≥+8pp gain

---

**End of Review**
