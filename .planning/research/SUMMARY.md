# Project Research Summary

**Project:** RobustCBM Refactoring
**Domain:** Python Code Refactoring (PyTorch Lightning)
**Researched:** 2026-03-12
**Confidence:** HIGH

## Executive Summary

This is a Python code refactoring project for a PyTorch Lightning-based concept bottleneck model (CBM) codebase. Research indicates the recommended approach combines automated tooling (Ruff for linting/formatting, Pyright for type checking) with a phased refactoring strategy that prioritizes safety through testing before structural changes.

The key insight from research is that refactoring a Lightning codebase carries unique risks: breaking the `LightningModule` interface contract, breaking checkpoint compatibility, and introducing silent behavior changes are the most critical pitfalls. The recommended approach is a 5-phase plan starting with foundation (tests + baseline), progressing through cleanup, structural refactoring, quality improvements, and ending with validation.

**Key risks and mitigations:**
- **Breaking Lightning interface** — Preserve method signatures for `training_step`, `validation_step`, `forward`, `configure_optimizers`
- **Checkpoint compatibility** — Use `strict=False` loading and implement migration for layer renames
- **Silent behavior changes** — Run regression tests comparing loss curves before/after each phase

## Key Findings

### Recommended Stack

**Core technologies:**
- **Ruff** — Primary linter/formatter, 10-100x faster than alternatives, replaces Flake8/isort/autoflake
- **Pyright** — Static type checking, Microsoft-backed, excellent for PyTorch/Lightning stubs
- **Vulture** — Dead code detection (unused functions/variables)
- **Rope** — AST-based refactoring library for rename/move operations
- **Bowler** — Safe batch refactoring for interface changes

Configuration should target Python 3.12 with line-length 100. Ruff handles import sorting and unused import removal; Pyright provides type checking. Rope and Bowler are for manual refactoring of renaming and batch changes.

### Expected Features

**Must have (table stakes):**
- Remove unused imports — quick win, clears noise for deeper analysis
- Remove dead code — reduces confusion and maintenance burden
- Fix naming convention violations — aligns with CONVENTIONS.md
- Replace magic numbers with named constants — self-documenting code
- Extract duplicated logic between CBM/VCBM/CEM models — DRY principle

**Should have (competitive):**
- Extract base classes from similar models — reduces duplication between CBM/VCBM/CEM
- Add type hints to public functions — better IDE support and bug detection
- Add error handling for common failure modes — robustness
- Organize/reorganize module structure — logical layout

**Defer (v2+):**
- Comprehensive docstrings — only if API stability is required
- Replace print statements with logging — configurable output (low user value)
- Full input validation layer — only if robustness is priority

### Architecture Approach

Research recommends a 5-phase approach organized by risk level:

1. **Foundation & Safety** — Analyze codebase, add basic tests, commit baseline
2. **Low-Risk Cleanup** — Remove unused imports, dead code, fix warnings
3. **Structural Refactoring** — Extract common model logic, reorganize modules, consolidate config
4. **Quality Improvements** — Add error handling, docstrings, type hints, testing framework
5. **Validation & Polish** — Run test suite, compare results, benchmark performance

The key architectural constraint is that **LightningModule hooks must preserve their signatures**. Extract shared logic to pure PyTorch base classes or utility functions, keeping Lightning-specific methods in leaf classes.

### Critical Pitfalls

1. **Breaking LightningModule Interface Contract** — Refactoring removes or changes signatures of `training_step`, `validation_step`, `forward`, `configure_optimizers`. Prevention: Never change hook signatures; use `self.hparams` or instance variables for new parameters.

2. **Breaking Checkpoint Compatibility** — Renaming layers or changing architecture causes checkpoint loading failures. Prevention: Document all attribute names before refactoring; use `strict=False` with migration logic.

3. **Circular Imports** — Moving classes between modules creates import cycles. Prevention: Use lazy imports inside functions; keep imports unidirectional; use `TYPE_CHECKING` for type hints.

4. **Breaking Config Contract** — Renaming config keys breaks existing YAML configs. Prevention: Implement fallback logic for old keys with deprecation warnings; add config schema validation.

5. **Silent Behavior Changes** — Refactoring changes loss values or convergence behavior without errors. Prevention: Run identical experiments before/after; compare loss curves; use explicit keyword arguments.

6. **Breaking Test Pipeline** — No existing tests means refactoring breaks functionality undetected. Prevention: Create regression tests for core functionality BEFORE refactoring.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Foundation & Safety
**Rationale:** Must establish baseline and safety net before any refactoring. No tests exist currently (CONCERNS.md notes this), so this is critical.
**Delivers:** Baseline linter output (JSON), committed pre-refactoring state, minimal smoke tests for model loading, forward pass, config loading
**Addresses:** Creates foundation for all subsequent phases
**Avoids:** Pitfall 8 (Breaking Test Pipeline) — tests must exist before structural changes

### Phase 2: Low-Risk Cleanup
**Rationale:** High confidence, low risk changes clear noise and build momentum. Dependencies from Phase 1.
**Delivers:** Clean imports (F401), no dead code, linter warnings resolved
**Addresses:** Features: Remove unused imports, Remove dead code, Fix naming conventions, Replace magic numbers
**Uses:** Ruff (auto-fix for imports, naming), Vulture (dead code detection)
**Avoids:** Pitfall 6 (Removing Used Methods) — grep all files before removal

### Phase 3: Structural Refactoring
**Rationale:** Highest value but highest risk. Must have tests and clean codebase first. This is where CBM/VCBM/CEM extraction happens.
**Delivers:** Base classes for common model logic, reorganized modules, consolidated config
**Addresses:** Features: Extract duplicated logic, Extract base classes, Organize module structure
**Uses:** Rope (rename refactoring), Bowler (batch interface changes)
**Avoids:** Pitfall 1 (Lightning Interface), Pitfall 2 (Checkpoints), Pitfall 3 (Circular Imports), Pitfall 7 (Silent Behavior)

### Phase 4: Quality Improvements
**Rationale:** Adds robustness after structure is stable.
**Delivers:** Error handling, docstrings, type hints, full pytest setup
**Addresses:** Features: Add type hints, Add error handling, Add comprehensive docstrings
**Avoids:** Pitfall 4 (Config Contract) — add validation here

### Phase 5: Validation & Polish
**Rationale:** Final verification that refactoring preserved behavior.
**Delivers:** Passing test suite, benchmark results, verified checkpoint loading
**Verifies:** All previous phases, runs existing experiments to compare results

### Phase Ordering Rationale

- **Why 1-2 before 3:** Structural refactoring is high-risk; tests and clean codebase make changes safer
- **Why extraction after cleanup:** Can't identify duplication accurately with noisy imports and dead code
- **Why testing before quality:** Type hints and error handling have less risk than structural changes
- **Why validation last:** Must verify behavior preservation before declaring success

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (Structural Refactoring):** Complex — need detailed analysis of CBM/VCBM/CEM inheritance to know what to extract. Recommend `/gsd:research-phase` for specific extraction patterns.
- **Phase 4 (Quality Improvements):** May need research on best practices for error handling in research codebases

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** Standard testing setup patterns
- **Phase 2 (Cleanup):** Ruff/Vulture are well-documented, standard patterns
- **Phase 5 (Validation):** Standard regression testing patterns

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Ruff and Pyright are standard in 2025-2026 Python ecosystem; tools well-documented |
| Features | HIGH | Based on established Python refactoring principles; matches typical research codebase patterns |
| Architecture | HIGH | 5-phase approach is standard for large refactoring; Lightning-specific patterns well-documented |
| Pitfalls | HIGH | Comprehensive list with clear prevention strategies; well-aligned with Lightning constraints |

**Overall confidence:** HIGH

### Gaps to Address

- **CBM/VCBM/CEM inheritance details:** Need to examine actual code to determine what logic to extract to base classes. Recommend doing this analysis during Phase 1.
- **Checkpoint inventory:** How many existing checkpoints need migration? Need to audit before Phase 3.
- **Config structure:** CONCERNS.md notes `jpgd_args: {jpgd_lambda: 1}` nested issue. Need to validate exact config format during Phase 1.

## Sources

### Primary (HIGH confidence)
- Ruff Documentation (astral.sh) — Primary linter/formatter
- Pyright Documentation (microsoft.github.io) — Type checker
- PyTorch Lightning Documentation — LightningModule interface requirements

### Secondary (HIGH confidence)
- Martin Fowler's Refactoring Catalog — General refactoring principles
- Rope Documentation — AST-based refactoring library
- Vulture Documentation — Dead code detection

### Tertiary (HIGH confidence)
- Project context: PROJECT.md, STRUCTURE.md, CONCERNS.md, CONVENTIONS.md — Local codebase analysis

---

*Research completed: 2026-03-12*
*Ready for roadmap: yes*
