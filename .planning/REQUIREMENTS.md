# Requirements: RobustCBM Refactoring

**Defined:** 2026-03-12
**Core Value:** A research framework for studying adversarial robustness in Concept Bottleneck Models through concept intervention.

## v1 Requirements

Refactoring the existing RobustCBM codebase. Each maps to roadmap phases.

### Foundation

- [ ] **FOUND-01**: Establish baseline by committing current working state
- [ ] **FOUND-02**: Analyze existing codebase to identify refactoring targets

### Code Cleanup

- [ ] **CLEAN-01**: Remove unused imports across all Python files
- [ ] **CLEAN-02**: Remove dead code and unreachable code paths
- [ ] **CLEAN-03**: Fix inconsistent naming conventions
- [ ] **CLEAN-04**: Extract magic numbers and hardcoded values to constants

### Structural Refactoring

- [ ] **STRUCT-01**: Extract common logic from CBM/VCBM/CEM models into base classes
- [ ] **STRUCT-02**: Reorganize module structure for better maintainability
- [ ] **STRUCT-03**: Add type hints to key functions and classes

### Quality Improvements

- [ ] **QUAL-01**: Add error handling to key functions
- [ ] **QUAL-02**: Add docstrings to public APIs
- [ ] **QUAL-03**: Add logging for debugging and monitoring
- [ ] **QUAL-04**: Add comprehensive test coverage

### Validation

- [ ] **VALID-01**: Verify training still works after refactoring
- [ ] **VALID-02**: Verify evaluation still works after refactoring
- [ ] **VALID-03**: Verify model checkpoint loading still works
- [ ] **VALID-04**: Compare results before and after refactoring

## v2 Requirements

Deferred to future release.

### Testing Framework

- **TEST-01**: Add unit tests for utility functions
- **TEST-02**: Add integration tests for model training
- **TEST-03**: Add smoke tests for dataset loading

### Advanced Refactoring

- **ADV-01**: Refactor attack layer for better extensibility
- **ADV-02**: Add configuration validation
- **ADV-03**: Implement plugin system for new model variants

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Adding new model architectures | Beyond current scope |
| Adding new datasets | Beyond current scope |
| Implementing new attack methods | Beyond current scope |
| Building user-facing applications | Research framework, not product |
| Breaking API changes | Must maintain backward compatibility |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Pending |
| FOUND-02 | Phase 1 | Pending |
| CLEAN-01 | Phase 2 | Pending |
| CLEAN-02 | Phase 2 | Pending |
| CLEAN-03 | Phase 2 | Pending |
| CLEAN-04 | Phase 2 | Pending |
| STRUCT-01 | Phase 3 | Pending |
| STRUCT-02 | Phase 3 | Pending |
| STRUCT-03 | Phase 3 | Pending |
| QUAL-01 | Phase 4 | Pending |
| QUAL-02 | Phase 4 | Pending |
| QUAL-03 | Phase 4 | Pending |
| QUAL-04 | Phase 4 | Pending |
| VALID-01 | Phase 4 | Pending |
| VALID-02 | Phase 4 | Pending |
| VALID-03 | Phase 4 | Pending |
| VALID-04 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-03-12 after roadmap creation*
