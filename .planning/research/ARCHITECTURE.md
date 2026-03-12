# Architecture Patterns for Code Refactoring

**Domain:** Python refactoring (PyTorch Lightning codebase)
**Researched:** 2026-03-12

## Refactoring Approach for RobustCBM

Based on the codebase analysis (STRUCTURE.md) and identified concerns (unused imports, hardcoded values, limited error handling, inconsistent naming, no testing framework), this document outlines a phased approach to organizing the refactoring.

## Recommended Refactoring Phases

### Phase 1: Foundation & Safety (Prerequisite)

**Goal:** Establish baseline and prevent regressions.

| Task | Description | Risk Level |
|------|-------------|-------------|
| Analyze codebase | Map all dependencies, identify duplication, catalog hardcoded values | Low |
| Add basic tests | Test critical paths: training loop, forward pass, dataloader | Medium |
| Version control | Ensure all changes are committed before starting | Low |

**Why first:** Provides safety net for subsequent changes. Without tests, any refactoring risks silent breakage.

### Phase 2: Low-Risk Cleanup

**Goal:** Remove obvious issues with minimal risk.

| Task | Description | Dependencies |
|------|-------------|--------------|
| Remove unused imports | Clean up imports identified in CONCERNS.md | Phase 1 |
| Remove dead code | Functions/variables not referenced anywhere | Phase 1 |
| Fix syntax warnings | Address all linter warnings | Phase 1 |

**Why early:** High confidence, low risk. Quick wins build momentum and reduce noise when doing deeper refactoring.

### Phase 3: Structural Refactoring (Core)

**Goal:** Reduce duplication and improve module organization.

| Task | Description | Dependencies |
|------|-------------|--------------|
| Extract common model logic | Shared code between CBM, VCBM, CEM into base class or mixins | Phase 2 |
| Reorganize utility modules | Move shared logic to appropriate modules (e.g., `utils.py` vs domain modules) | Phase 2 |
| Consolidate configuration | Extract hardcoded values to config/constants | Phase 2 |
| Standardize naming | Enforce consistent naming (e.g., dataset files are inconsistent: `CUB.py` vs `celeba.py`) | Phase 2 |

**Why mid-phase:** This is the highest-value refactoring but also highest risk. Benefits:
- CBM, VCBM, CEM likely share backbone loading, concept projection logic
- Hardcoded values scattered in models make experimentation harder
- Inconsistent naming causes confusion

**Lightning-specific note:** When extracting base classes for models:
- Keep `LightningModule` methods (`training_step`, `forward`, etc.) in leaf classes
- Extract shared neural network architecture to pure PyTorch base classes
- Use composition for training logic, inheritance for model architecture

### Phase 4: Quality Improvements

**Goal:** Improve robustness and maintainability.

| Task | Description | Dependencies |
|------|-------------|--------------|
| Add error handling | Input validation, meaningful error messages | Phase 3 |
| Document public APIs | Docstrings for model classes, DataModules, key functions | Phase 3 |
| Add type hints | Where feasible (especially for DataModule interfaces) | Phase 3 |
| Implement testing framework | Add pytest/pytest-cov setup | Phase 1 |

### Phase 5: Validation & Polish

**Goal:** Ensure refactoring preserved behavior.

| Task | Description | Dependencies |
|------|-------------|--------------|
| Run full test suite | Verify all existing experiments still work | All prior |
| Run existing experiments | Compare results before/after refactoring | All prior |
| Performance benchmark | Ensure no regression in training speed | All prior |

## Dependency Graph

```
Phase 1 (Foundation)
    │
    ├──► Phase 2 (Cleanup) ─────────────┐
    │                                    │
    │                                    ▼
    │                          Phase 3 (Structure)
    │                                    │
    │                                    ▼
    │                          Phase 4 (Quality)
    │                                    │
    │                                    ▼
    │                          Phase 5 (Validation)
    │
    └──► Phase 4 (Testing framework) ───┘
```

**Key constraint:** Phases must proceed in order. Testing framework can be added earlier but tests should be run after each phase.

## Lightning-Specific Considerations

### Model Refactoring

For PyTorch Lightning projects, maintain this separation:

| Layer | What to Refactor | What to Keep Separate |
|-------|-----------------|----------------------|
| Neural Network | Architecture into pure PyTorch classes | LightningModule wrapper |
| Training Logic | Shared training loops into base classes | Experiment-specific overrides |
| Data | Common dataset operations | Dataset-specific preprocessing |

**Recommended pattern:**
```
model/
├── CBM.py           # LightningModule
├── VCBM.py          # LightningModule
├── CEM.py           # LightningModule
├── layers.py        # Pure PyTorch: MLP, ConceptLayer, etc.
└── base.py          # Optional: BaseModel with shared logic
```

### DataModule Refactoring

LightningDataModules should be stateless for training. Refactoring should:
- Keep data loading in DataModule
- Keep concept mappings in DataModule attributes
- Extract preprocessing to separate utility functions

## Suggested Approach for This Project

Given the identified concerns in PROJECT.md:

1. **Start with Phase 1**: Add minimal smoke tests for training/evaluation
2. **Quick Phase 2**: Remove unused imports (already identified)
3. **Focus Phase 3 on extraction**: The model duplication is likely the biggest maintenance burden
4. **Phase 4 for error handling**: Research codebases need flexibility, but basic validation prevents debugging nightmare
5. **Critical**: Run full experiment suite before marking refactoring complete

## Sources

- Martin Fowler's Refactoring (refactoring.com) - General refactoring principles
- PyTorch Lightning documentation - Framework-specific patterns
- Project context: PROJECT.md, STRUCTURE.md, CONCERNS.md
