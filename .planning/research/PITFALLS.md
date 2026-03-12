# Pitfalls Research

**Domain:** Python Code Refactoring (PyTorch Lightning)
**Researched:** 2026-03-12
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Breaking LightningModule Interface Contract

**What goes wrong:**
Refactoring removes or changes the signature of required Lightning methods (`training_step`, `validation_step`, `forward`, `configure_optimizers`), causing runtime errors like "LightningModule ... has a `training_step` with missing arguments" or checkpoint loading failures.

**Why it happens:**
The `LightningModule` has strict interface requirements. Methods like `training_step(batch, batch_idx)` have specific signatures that PyTorch Lightning expects. Refactoring without understanding these contracts breaks the training loop.

**How to avoid:**
1. Never change method signatures of hooks: `training_step`, `validation_step`, `test_step`, `forward`, `configure_optimizers`, `setup`, `teardown`
2. If you need new parameters, use `self.hparams` (parsed from config) or instance variables
3. Maintain the same return structure from `training_step` (dict with `loss` key)
4. When extracting logic, keep wrapper methods intact and call extracted functions internally

**Warning signs:**
- Errors like "missing 1 required positional argument" in Lightning methods
- Checkpoints fail to load with "state_dict mismatch" errors
- Training runs but validation never executes (validation_step signature wrong)

**Phase to address:**
This must be addressed in the **restructuring phase** (Phase 2: Module Reorganization). Any refactoring of model classes must preserve the Lightning interface.

---

### Pitfall 2: Breaking Checkpoint Compatibility

**What goes wrong:**
After refactoring, existing checkpoints fail to load with errors like "Error loading checkpoint, missing key 'model.layer1.weight'" or "Unexpected keys in state_dict".

**Why it happens:**
- Renaming model attributes/layers
- Changing layer ordering
- Removing layers that exist in saved checkpoints
- Not handling checkpoint migration for architecture changes

**How to avoid:**
1. Before refactoring model classes, document all attribute names
2. Use Lightning's `load_from_checkpoint` with `map_location` and handle missing/unexpected keys
3. If renaming layers, implement checkpoint migration or use `strict=False` with custom loading
4. Keep a migration script for breaking changes
5. Version your model architectures

**Warning signs:**
- "Unexpected keys" or "Missing keys" in checkpoint loading errors
- Model loads but produces NaN losses (layer shape mismatch)
- Different number of parameters before vs. after refactoring

**Phase to address:**
Addressed in **restructuring phase** with checkpoint migration strategy. Must verify all existing checkpoints can be loaded after refactoring.

---

### Pitfall 3: Introducing Circular Imports

**What goes wrong:**
When reorganizing modules, circular import errors appear: "ImportError: cannot import name 'CBM' from partially initialized module".

**Why it happens:**
Moving classes between modules without understanding import dependencies. Common pattern: `A.py` imports `B.py`, `B.py` imports `A.py`, or module hierarchy creates cycles during reorganization.

**How to avoid:**
1. Use lazy imports inside functions instead of top-level imports when cycle risk exists
2. Create a shared `__init__.py` that imports only leaf modules
3. Move shared utilities to a neutral `utils/` or `common/` module
4. Use `TYPE_CHECKING` for type hints to break import cycles
5. Keep imports unidirectional (higher-level modules import lower-level, not vice versa)

**Warning signs:**
- ImportError when running any Python file in the project
- Tests fail to import
- IDE shows import errors even if code "works" (circular imports can appear to work in Python)

**Phase to address:**
Addressed in **restructuring phase** (Module Reorganization). Test imports after each module move.

---

### Pitfall 4: Breaking Config Contract

**What goes wrong:**
After refactoring, existing YAML config files fail with errors like "KeyError: 'jpgd_lambda'" or configs load but training behaves differently.

**Why it happens:**
- Renaming config keys that existing configs depend on
- Changing default values
- Removing config parameters that code expects
- Changing config structure (CONCERNS.md notes this: `jpgd_args: {jpgd_lambda: 1}` nested but code expects top-level)

**How to avoid:**
1. Never remove config keys without deprecation warning
2. Implement config migration: try new key, fall back to old key, warn about deprecation
3. Add config validation early in main.py to catch missing/invalid keys
4. Document all config keys and their purposes
5. Use config schemas (pydantic/dataclass) for validation

**Warning signs:**
- Training fails immediately with KeyError on config load
- Config loads but wrong values used (silent failure)
- Different behavior with same config before vs. after refactoring

**Phase to address:**
Addressed in **cleanup phase** (Phase 1: Import/Naming Cleanup). Add config validation schema early.

---

### Pitfall 5: Breaking Dataset/DataModule Contract

**What goes wrong:**
Refactoring dataset classes breaks training because `LightningDataModule` methods (`train_dataloader`, `val_dataloader`, `setup`) have specific contracts, or dataset return formats change.

**Why it happens:**
- Changing return format of `__getitem__` (e.g., changing from tuple to dict)
- Changing data augmentation pipeline without backward compatibility
- Renaming attributes that training code expects
- Changing concept mapping logic

**How to avoid:**
1. Keep `__getitem__` return format consistent (prefer dict for clarity)
2. If changing return format, update all consumers or create adapter
3. Document expected data formats at dataset boundaries
4. Test data pipeline independently from model

**Warning signs:**
- "RuntimeError: expected tensor, got list" during training
- Concept labels wrong (swapped/missing)
- DataLoader collation errors

**Phase to address:**
Addressed in **restructuring phase** when reorganizing dataset modules. Ensure data pipeline tests exist.

---

### Pitfall 6: Removing "Private" Methods That Are Used

**What goes wrong:**
Refactoring removes methods starting with `_` or `__` assuming they're unused, but they're actually called from other files or the main training loop.

**Why it happens:**
Python's convention is that `_private` methods are for internal use but not part of public API. However, in a codebase like this, internal methods may be called by main.py, test.py, or other model classes.

**How to avoid:**
1. Search ALL files for method calls before removing any method: `grep -r "method_name" --include="*.py"`
2. Check `main.py` and `test.py` for direct calls
3. Check parent classes (CBM is parent of VCBM, CEM)
4. Deprecate rather than remove: add deprecation warning, then remove in next version

**Warning signs:**
- AttributeError: "'CBM' object has no attribute '_some_method'"
- Methods called in main.py or test.py
- Methods used in sibling model classes (VCBM, CEM inherit from or use CBM)

**Phase to address:**
Addressed in **cleanup phase** - verify all method usage before removal.

---

### Pitfall 7: Introducing Silent Behavior Changes

**What goes wrong:**
Code refactoring changes behavior subtly: loss values differ, model converges differently, intervention produces different results - but no errors are raised.

**Why it happens:**
- Changing default arguments (especially mutable defaults)
- Changing order of operations
- Changing rounding or floating-point operations
- Changing data preprocessing
- Accidentally changing hyperparameter defaults

**How to avoid:**
1. Use explicit keyword arguments in function calls, avoid positional args
2. Never change default values without documenting
3. Run identical experiments before/after refactoring to verify results
4. Compare loss curves, not just final metrics
5. Save random seeds and verify reproducibility

**Warning signs:**
- Loss curves diverge from baseline
- Metrics differ by >1% from previous runs
- Non-deterministic results (run same config twice, get different outcomes)

**Phase to address:**
Addressed in **restructuring phase** - verify behavior consistency with regression tests.

---

### Pitfall 8: Breaking Test Pipeline

**What goes wrong:**
No tests exist (noted in CONCERNS.md: "No formal testing framework"), so refactoring breaks functionality without detection. Or adding tests后发现 they pass but functionality is wrong.

**Why it happens:**
- No baseline tests to compare against
- Tests written after refactoring reflect new (broken) behavior
- Test coverage gaps mean untested code breaks silently

**How to avoid:**
1. BEFORE refactoring, create minimal regression tests for core functionality:
   - Model instantiation
   - Single forward pass
   - Single training step
   - Config loading
   - Data loading
2. Run baseline tests before and after each refactoring step
3. Add new tests as new functionality is added, not after

**Warning signs:**
- No tests in repository
- Tests only test "happy path"
- Tests pass but training fails end-to-end
- No test for intervention logic, loss calculation, or attack methods

**Phase to address:**
Addressed in **testing phase** (Phase 3: Testing Framework). Tests must exist before major refactoring.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Copy-paste code between model variants | Faster initial implementation | Bug fixes need to be applied in multiple places; inconsistency drifts | Never - extract to base class or shared module |
| Hardcoded paths in config | Works immediately for one environment | Breaks in different setups; CONCERNS.md notes this | Only during initial prototyping |
| Dynamic module loading without validation | Flexible model/dataset selection | Silent failures if typo in config; crashes at runtime | Only with proper error messages and validation |
| Multiple backward passes in MTL | Simple implementation | Performance penalty; CONCERNS.md notes this | Only if gradient stability requires it |
| Iterate full dataset for imbalance weights | Simple weight calculation | Slow initialization; CONCERNS.md notes this | Only with caching mechanism |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| WandbLogger | Hardcoded offline=True, can't switch to online | Make offline configurable via config/env var |
| Checkpoint loading | Not handling missing keys when model changes | Use `strict=False` and log warnings for unexpected/missing keys |
| Config parsing | Using nested dict without schema validation | Use pydantic/dataclass for config validation with clear errors |
| Data symlinks | Hardcoding symlink path assumptions | Use environment variables or config for data paths |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Multiple backward passes in MTL | OOM on large models; slow training | Use gradient accumulation or single backward | At scale >10K parameters |
| Full dataset iteration for weights | Slow datamodule init; CONCERNS.md notes | Cache weights or compute once | Every run |
| Nested loops in test intervention | Test loop 11x slower than needed | Batch intervention computations | At evaluation time with many intervention levels |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Hardcoded project name in code | Not applicable for research | Use config |
| Data path in git history | Expose internal paths | Already in .gitignore, maintain this |
| No input validation on config | Runtime errors, confusing messages | Add pydantic validation schema |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No clear error messages on config failure | User doesn't know what's wrong | Add schema validation with helpful messages |
| Missing results directory crashes | Evaluation fails without clear fix | Create directory in code or document requirement |
| Wandb import fails silently | Logging doesn't work, no warning | Check logger init and warn if failed |

---

## "Looks Done But Isn't" Checklist

- [ ] **Checkpoint loading:** Often works in dev but fails in production — verify with actual saved checkpoint
- [ ] **Config migration:** Often done for new keys but old keys silently ignored — verify with old config files
- [ ] **Import cleanup:** Often removes imports from files but misses in `__init__.py` — verify all public APIs export correctly
- [ ] **Method removal:** Often removes "unused" methods but they're called via string — grep all files for method name
- [ ] **Test passing:** Often tests only run happy path — verify edge cases and error paths

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Breaking Lightning interface | HIGH | Revert to baseline, add minimal wrapper preserving original signature |
| Breaking checkpoint compatibility | MEDIUM | Use `strict=False` loading, log missing/unexpected keys, add migration |
| Circular imports | LOW | Identify cycle, move one import to lazy import inside function |
| Breaking config contract | LOW | Add fallback logic for old keys with deprecation warning |
| Silent behavior changes | HIGH | Revert, add baseline test, identify change, fix or document |

---

## Pitfall-to-Phase Mapping

How the planned phases should address these pitfalls:

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Lightning interface break | Phase 2: Module Reorganization | Run training for 1 epoch after any model refactor |
| Checkpoint compatibility | Phase 2: Module Reorganization | Load existing checkpoint after any model change |
| Circular imports | Phase 2: Module Reorganization | Test `python -c "import model; import dataset"` after moves |
| Config contract break | Phase 1: Import/Naming Cleanup | Add config schema validation; run with old configs |
| Dataset contract break | Phase 2: Module Reorganization | Run data pipeline test after dataset changes |
| Remove used methods | Phase 1: Import/Naming Cleanup | Grep all files before removing any method |
| Silent behavior changes | Phase 3: Testing Framework | Compare loss curves before/after refactoring |
| Test pipeline break | Phase 3: Testing Framework | Run full test suite after any change |

---

## Sources

- PyTorch Lightning Documentation: LightningModule interface requirements
- CONCERNS.md (local): Existing codebase issues to preserve during refactoring
- Python import system: Circular import patterns and solutions
- Common Python anti-patterns: Mutable default arguments, breaking API contracts
- Research codebase patterns: Based on typical CBM/VCBM/CEM model structures

---

*Pitfalls research for: Python Code Refactoring (PyTorch Lightning)*
*Researched: 2026-03-12*
