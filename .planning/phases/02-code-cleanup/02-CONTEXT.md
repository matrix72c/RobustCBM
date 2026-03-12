# Phase 2: Code Cleanup - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove code noise and establish consistent coding standards:
- Fix unused imports (Ruff F401)
- Remove dead code (Vulture)
- Fix naming inconsistencies
- Extract magic numbers to constants

</domain>

<decisions>
## Implementation Decisions

### Unused Code (Ruff)
- Manual review for each of the 21 Ruff issues
- Fix issues one by one, commit after review

### Dead Code (Vulture)
- Remove all 5 dead code items automatically
- Use vulture --remove if safe

### Magic Numbers
- Extract from model/ and dataset/ directories only
- Create constants file or extract to module-level

### Naming Conventions
- Standard Python: snake_case for functions/variables, PascalCase for classes

</decisions>

<specifics>
## Specific Ideas

- Phase 1 found 21 Ruff issues (unused imports F401, variables F841)
- Phase 1 found 5 Vulture dead code items
- Focus on model/ and dataset/ directories for constants extraction

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-code-cleanup*
*Context gathered: 2026-03-12*
