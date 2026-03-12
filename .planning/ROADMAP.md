# RobustCBM Refactoring Roadmap

## Phases

- [x] **Phase 1: Foundation** - Establish baseline and analyze codebase (completed 2026-03-12)
- [x] **Phase 2: Code Cleanup** - Remove unused imports, dead code, fix naming, extract constants (completed 2026-03-12)
- [ ] **Phase 3: Structural Refactoring** - Extract common model logic, reorganize modules, add type hints
- [ ] **Phase 4: Quality & Validation** - Add error handling, docstrings, logging, tests; verify functionality

## Phase Details

### Phase 1: Foundation
**Goal**: Establish baseline state and analyze codebase for refactoring targets

**Depends on**: Nothing (first phase)

**Requirements**: FOUND-01, FOUND-02

**Success Criteria** (what must be TRUE):
1. Current working state committed to git with descriptive message
2. Codebase analysis completed identifying refactoring targets (duplication, hardcoded values, naming issues)
3. Smoke tests exist for model loading, forward pass, and config loading

**Plans**: 1 plan
- [x] 01-PLAN.md — Testing infrastructure + codebase analysis

---

### Phase 2: Code Cleanup
**Goal**: Remove code noise and establish consistent coding standards

**Depends on**: Phase 1

**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04

**Success Criteria** (what must be TRUE):
1. No unused imports remain in any Python file (verified by Ruff F401)
2. No dead code or unreachable code paths exist (verified by Vulture)
3. All naming conventions consistent (snake_case for functions/variables, PascalCase for classes)
4. All magic numbers and hardcoded values extracted to named constants

**Plans**: 1 plan
- [x] 02-PLAN.md — Remove unused imports, dead code, fix naming, extract constants (COMPLETE)

---

### Phase 3: Structural Refactoring
**Goal**: Improve code architecture through extraction and reorganization

**Depends on**: Phase 2

**Requirements**: STRUCT-01, STRUCT-02, STRUCT-03

**Success Criteria** (what must be TRUE):
1. Common logic from CBM/VCBM/CEM models extracted into base classes or utilities
2. Module structure reorganized for better maintainability and logical grouping
3. Type hints added to key public functions and classes

**Plans**: 1 plan
- [ ] 03-01-PLAN.md — Extract common logic, add type hints, reorganize modules

---

### Phase 4: Quality & Validation
**Goal**: Add robustness and verify all functionality preserved

**Depends on**: Phase 3

**Requirements**: QUAL-01, QUAL-02, QUAL-03, QUAL-04, VALID-01, VALID-02, VALID-03, VALID-04

**Success Criteria** (what must be TRUE):
1. Error handling added to key functions with appropriate exceptions
2. Docstrings added to all public APIs explaining purpose, args, returns
3. Logging replaces print statements for debugging and monitoring
4. Comprehensive test coverage for core functionality
5. Training pipeline executes successfully after refactoring
6. Evaluation pipeline executes successfully after refactoring
7. Model checkpoint loading works (backward compatibility preserved)
8. Results before and after refactoring match (regression test passes)

**Plans**: TBD

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 1/1 | Complete    | 2026-03-12 |
| 2. Code Cleanup | 1/1 | Complete    | 2026-03-12 |
| 3. Structural Refactoring | 0/1 | Not started | - |
| 4. Quality & Validation | 0/1 | Not started | - |

---

## Notes

- **Granularity**: Coarse - 4 phases combining quality improvements with validation
- **Total v1 Requirements**: 17
- **Research already completed**: Yes (SUMMARY.md provides detailed guidance)
- **Key constraint**: Must preserve LightningModule interface signatures to avoid breaking checkpoint loading
