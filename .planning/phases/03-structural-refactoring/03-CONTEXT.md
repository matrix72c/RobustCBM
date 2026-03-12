# Phase 3: Structural Refactoring - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Extract common logic from CBM/VCBM/CEM models into reusable base classes. Reorganize module structure for better maintainability. Add type hints to core public interfaces. Must preserve LightningModule interface for checkpoint compatibility.

</domain>

<decisions>
## Implementation Decisions

### Common Logic Extraction
- **Approach:** Extend CBM — VCBM and CEM inherit from CBM, override specific methods
- **calc_loss:** VCBM/CEM override calc_loss method for custom loss computation
- **Optimizer:** Keep optimizer and scheduler configuration in base CBM class
- **Refactoring:** Incremental — clean up CBM first, then refactor VCBM/CEM to use new patterns

### Module Reorganization
- **Model structure:** By model — model/cbm/, model/vcbm/, model/cem/ directories
- **Model contents:** Each directory contains just the model class (single responsibility)
- **Dataset structure:** Keep flat — dataset/AwA.py, CUB.py, CelebA.py in dataset/ root

### Type Hints
- **Scope:** Public APIs only — __init__ parameters, forward() returns, training_step outputs
- **Focus:** Core interfaces — not comprehensive, prioritize key methods
- **Typing approach:** Python types (list, dict) — simple, no dependencies

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- CBM class: LightningModule with training/validation loops, optimizer config
- MLP class: Simple feedforward network in CBM.py
- backbone.py: ResNet, VGG backbones

### Established Patterns
- Inheritance: VCBM(CBM), CEM(CBM) — both extend base CBM
- Override pattern: VCBM overrides forward + calc_loss, CEM overrides __init__ + forward
- Lightning interface: training_step, validation_step, test_step, configure_optimizers

### Integration Points
- model/__init__.py: Exports CBM, VCBM, CEM classes
- dataset/__init__.py: Exports AwA, CUB, CelebA datamodules
- main.py, test.py: Import from model/ and dataset/

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-structural-refactoring*
*Context gathered: 2026-03-12*
