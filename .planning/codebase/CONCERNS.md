# Codebase Concerns

**Analysis Date:** 2026-03-12

## Tech Debt

**Dependency Mismatch (Critical):**
- Issue: `requirements.txt` specifies `swanlab==0.7.5` but the code imports and uses `WandbLogger` from `lightning.pytorch.loggers` in `main.py` and `test.py`
- Files: `requirements.txt`, `main.py`, `test.py`
- Impact: Runtime error when running training - "No module named 'wandb'" or similar import error
- Fix approach: Either install wandb package or replace all `WandbLogger` calls with appropriate logger for the installed package

**Hardcoded Project Name:**
- Issue: Project name "RAIDCXM" is hardcoded in multiple places
- Files: `main.py` (lines 61, 78)
- Impact: Not portable, requires code changes to use for different projects
- Fix approach: Move to configuration file or environment variable

**Hardcoded Paths:**
- Issue: Many paths are hardcoded (checkpoints/, results/, data/)
- Files: `main.py`, `test.py`, `utils.py`, `dataset/CUB.py`
- Impact: Not portable across different environments
- Fix approach: Use configuration or environment variables for paths

**Duplicate Import in CUB.py:**
- Issue: `import numpy as np` appears twice - once at line 6 and again at line 566
- Files: `dataset/CUB.py`
- Impact: Code clutter, potential confusion
- Fix approach: Remove duplicate import at line 566

## Known Bugs

**JPGD Lambda Parameter Mismatch:**
- Issue: In `config.yaml`, `jpgd_args: {jpgd_lambda: 1}` is nested, but in `model/CBM.py` line 62, the code expects `jpgd_lambda` as a top-level parameter (`self.hparams.jpgd_lambda`). Also, in the loss function at line 106, it directly uses `jpgd_lambda` from `self.hparams`.
- Files: `config.yaml`, `model/CBM.py`
- Trigger: Running training with JPGD attack enabled
- Workaround: Pass `jpgd_lambda` as a top-level parameter in config, not nested in `jpgd_args`

**Missing Results Directory:**
- Issue: Code writes to `results/result.csv` without creating the directory first
- Files: `main.py` (line 96-102)
- Trigger: Running evaluation without pre-existing results directory
- Workaround: Create `results/` directory manually before running

## Security Considerations

**Environment-Specific Data Symlink:**
- Issue: `data` directory is a symlink to `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/data/`
- Files: `data -> /mnt/shared-storage-gpfs2/...`
- Current mitigation: Directory is in .gitignore
- Recommendations: Document the data dependency clearly; consider using environment variables for data path

**No Input Validation:**
- Issue: Configuration parameters are used directly without validation
- Files: `main.py`, `test.py`
- Risk: Invalid config values could cause runtime errors or unexpected behavior
- Recommendations: Add config validation schema

## Performance Bottlenecks

**MTL Gradient Computation Inefficiency:**
- Problem: Multiple backward passes in MTL mode with full gradient computation
- Files: `model/CBM.py` (lines 252-256), `mtl.py` (lines 53-64)
- Cause: Computing gradients multiple times for multi-task learning
- Improvement path: Use gradient accumulation or single backward pass optimization

**Dataset Iteration for Imbalance Weights:**
- Problem: `cal_class_imbalance_weights` iterates through entire dataset to compute weights
- Files: `utils.py` (lines 63-82)
- Cause: Full dataset iteration on every datamodule initialization
- Improvement path: Cache imbalance weights or compute once and store

**Concept Intervene Loop in Test:**
- Problem: Test step has nested loops (11 intervention levels x 6 attack modes)
- Files: `model/CBM.py` (lines 381-406)
- Cause: Forward pass repeated many times per batch
- Improvement path: Batch intervene computations where possible

## Fragile Areas

**CBM.py - Large Class (406 lines):**
- Files: `model/CBM.py`
- Why fragile: Multiple responsibilities (training, validation, testing, intervention, adversarial attacks) in single class
- Safe modification: Refactor into smaller modules (CBMTrainer, CBMTester, CBMIntervener)
- Test coverage: Test coverage limited - only runs via end-to-end training

**CUB.py - Large Dataset Module (680 lines):**
- Files: `dataset/CUB.py`
- Why fragile: Contains 4 classes (SELECTED_CONCEPTS, CONCEPT_SEMANTICS, CUBDataSet, CustomCUBDataSet, CustomCUB) with 680 lines
- Safe modification: Split into separate files for each dataset class

**Dynamic Module Loading:**
- Files: `main.py` (lines 26-27, 43-44), `test.py` (lines 47-48)
- Why fragile: Uses `getattr(dataset, cfg["dataset"])` and `getattr(pl_model, cfg["model"])` without validation
- Safe modification: Add validation for dataset/model names before loading

## Scaling Limits

**Checkpoint Storage:**
- Current capacity: Single checkpoint per run (save_top_k=1)
- Limit: No way to resume from best checkpoint automatically without manual intervention
- Scaling path: Increase save_top_k or implement checkpoint cleanup strategy

**Batch Size Configuration:**
- Current capacity: Fixed batch size from config
- Limit: May not be optimal for different GPU memory sizes
- Scaling path: Add automatic batch size detection or learning rate scaling

## Dependencies at Risk

**autoattack Package:**
- Risk: External package `autoattack==0.1` with pinned old version
- Impact: Package compatibility issues with newer PyTorch versions
- Migration plan: Update to latest version or verify compatibility

**torchvision Model Weights:**
- Risk: Using `DEFAULT` weights may change behavior with torchvision updates
- Impact: Model weights could change, affecting reproducibility
- Migration plan: Pin specific weight versions for reproducibility

## Missing Critical Features

**Configuration Validation:**
- Problem: No schema validation for YAML config files
- Blocks: Easy debugging of config errors; early failure detection

**Unit Tests:**
- Problem: No unit tests for core components
- Blocks: Safe refactoring; regression detection

**Logging Configuration:**
- Problem: Hardcoded offline=True for WandbLogger
- Impact: Cannot easily switch to online logging without code changes

## Test Coverage Gaps

**Model Tests:**
- What's not tested: Individual model components (forward pass, loss calculation, intervention)
- Files: `model/CBM.py`, `model/VCBM.py`
- Risk: Bugs in model logic could go unnoticed until full training run
- Priority: High

**Utility Functions:**
- What's not tested: `yaml_merge`, `flatten_dict`, `build_name`, `initialize_weights`
- Files: `utils.py`
- Risk: Edge cases in utility functions could cause failures
- Priority: Medium

**Dataset Tests:**
- What's not tested: Data loading, concept mapping, imbalance weight calculation
- Files: `dataset/CUB.py`, `dataset/celeba.py`, `dataset/AwA.py`
- Risk: Data processing bugs could affect model training
- Priority: Medium

---

*Concerns audit: 2026-03-12*
