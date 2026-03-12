# Feature Research: Python Code Refactoring Patterns

**Domain:** Python Code Refactoring (PyTorch Lightning context)
**Researched:** 2026-03-12
**Confidence:** MEDIUM

Note: Web search encountered issues; findings based on established software engineering principles and common Python/PyTorch patterns.

## Feature Landscape

### Table Stakes (Essential Refactoring)

Core refactoring activities that must be performed for any improvement effort. Without these, the codebase remains problematic.

| Pattern | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Remove unused imports | Standard practice, reduces confusion and load time | LOW | Use tools like `pyflakes` or IDE inspections |
| Remove dead code | Reduces maintenance burden and confusion | LOW | Unused functions, unreachable code branches |
| Fix naming conventions | Readability and maintainability | LOW | Align with CONVENTIONS.md snake_case/PascalCase |
| Extract duplicated logic | DRY principle violation causes bugs | MEDIUM | Models share training loops, loss computations |
| Add type hints where missing | Better IDE support, catches bugs early | MEDIUM | Already partially used per CONVENTIONS.md |
| Organize imports correctly | Standard Python style, faster resolution | LOW | Already in CONVENTIONS.md but not enforced |

### Good Practices (Advanced Refactoring)

Refactoring patterns that significantly improve code quality but require more effort and understanding.

| Pattern | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Extract base classes from similar models | Reduces duplication between CBM/VCBM/CEM | MEDIUM-HIGH | CBM is parent of VCBM, CEM, backbone |
| Use composition over inheritance | Flexible model building, easier testing | MEDIUM | Already has inheritance, could add composition |
| Extract utility functions | Improves testability, reusability | LOW-MEDIUM | utils.py, hsic.py, mtl.py already exist |
| Add error handling with specific exceptions | Better debugging, user feedback | MEDIUM | CONCERNS.md mentions limited error handling |
| Replace magic numbers with constants | Self-documenting code | LOW | CONCERNS.md mentions hardcoded values |
| Extract configuration to config files | Separation of concerns, easier changes | LOW | Already uses YAML, ensure all params externalized |
| Add docstrings to public APIs | Better maintainability, helps users | MEDIUM | CONVENTIONS.md notes rare docstrings |
| Replace print with logging | Configurable output, better levels | LOW | CONCERNS.md notes print statements |
| Add input validation | Prevents silent failures | MEDIUM | Validate shapes, types, value ranges |

### Anti-Features (Avoid)

Refactoring activities that seem beneficial but can cause problems or are not worth the effort.

| Pattern | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Over-abstraction | "Clean code" obsession | Creates indirection, harder to navigate | Keep related logic together |
| Premature optimization | Performance concerns | Wastes time, may harm readability | Profile first, optimize critical paths |
| Refactor everything at once | Desire for perfection | High risk, hard to test, team blocked | Refactor incrementally, one pattern at a time |
| Add interfaces everywhere | "Java style" in Python | Python duck typing makes this unnecessary | Use protocols if truly needed |
| Enforce all linters | Code quality obsession | Conflicting rules, slows development | Selectively enable rules, project-specific |
| Replace all dicts with dataclasses | "Modern Python" | Not always better, breaks dict APIs | Use where appropriate |
| Remove all `**kwargs` | Perceived as untyped | Sometimes kwargs are appropriate for extensibility | Keep where extension needed |

## Refactoring Dependencies

```
[Remove unused imports]
    └──required──> [Fix naming conventions]

[Extract duplicated logic]
    └──requires──> [Identify duplication first]
                       └──requires──> [Remove dead code]

[Add type hints]
    └──enhances──> [Add error handling]

[Extract base classes]
    └──requires──> [Understand inheritance hierarchy]
                       └──requires──> [CBM, VCBM, CEM, backbone relationships]

[Replace print with logging]
    └──conflicts──> [Need logging framework decision first]
```

### Dependency Notes

- **Remove unused imports before any other refactoring:** Reduces noise, makes patterns easier to see
- **Identify duplication before extracting:** Can't extract what isn't recognized
- **Understand inheritance hierarchy before extracting base classes:** Must know what CBM/VCBM/CEM share
- **Logging framework decision conflicts with immediate print replacement:** Need to decide on logging first

## MVP Definition

### Must Do First (Phase 1)

Essential refactoring that enables subsequent work and has low risk.

- [ ] Remove unused imports — quick win, clears noise
- [ ] Remove dead code — quick win, reduces confusion
- [ ] Fix naming convention violations — aligns with CONVENTIONS.md
- [ ] Replace magic numbers with named constants — self-documenting

### Should Do Second (Phase 2)

Refactoring that improves maintainability and reduces duplication.

- [ ] Extract duplicated logic between models — DRY principle
- [ ] Add type hints to public functions — better DX
- [ ] Add error handling for common failure modes — robustness
- [ ] Organize/reorganize module structure — logical layout

### Consider Later (Phase 3)

Advanced refactoring for long-term code health.

- [ ] Extract base classes from similar models — if composition doesn't work
- [ ] Replace print statements with logging — configurable output
- [ ] Add comprehensive docstrings — if API stability needed
- [ ] Add input validation layer — if robustness is priority

## Prioritization Matrix

| Pattern | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Remove unused imports | LOW (internal) | LOW | P1 |
| Remove dead code | LOW (internal) | LOW | P1 |
| Fix naming conventions | MEDIUM (DX) | LOW | P1 |
| Replace magic numbers | MEDIUM (maintainability) | LOW | P1 |
| Extract duplicated logic | HIGH (maintainability) | MEDIUM | P1 |
| Add type hints | MEDIUM (DX) | MEDIUM | P2 |
| Add error handling | MEDIUM (robustness) | MEDIUM | P2 |
| Extract base classes | HIGH (architecture) | HIGH | P2 |
| Replace print with logging | LOW (internal) | MEDIUM | P3 |
| Add comprehensive docstrings | MEDIUM (usability) | HIGH | P3 |

## PyTorch Lightning-Specific Considerations

The codebase uses Lightning extensively. Refactoring should preserve:

| Area | Current Pattern | Refactoring Consideration |
|------|-----------------|---------------------------|
| Training loop | Lightning automatic | Don't break `trainer.fit()` contract |
| DataModule | `LightningDataModule` | Keep interface for data loading |
| Hyperparameters | `save_hyperparameters()` | Preserve for checkpoint compatibility |
| Callbacks | Currently unused | Could add for logging, early stopping |
| Lightning hooks | `training_step`, `configure_optimizers` | Don't break hook signatures |

## Sources

- Martin Fowler's Refactoring Catalog (general patterns)
- PEP 8 Style Guide (naming, imports)
- CONVENTIONS.md (project-specific patterns)
- PROJECT.md (known issues: unused imports, hardcoded values, naming inconsistencies)

---
*Feature research for: Python Code Refactoring*
*Researched: 2026-03-12*
