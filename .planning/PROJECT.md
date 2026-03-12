# RobustCBM

## What This Is

A PyTorch Lightning-based Concept Bottleneck Model (CBM) framework for adversarial robustness in computer vision. Supports multiple datasets (CUB, CelebA, AwA), multiple model variants (CBM, VCBM, CEM), adversarial training (PGD-based attacks), and concept intervention at test time.

## Core Value

A research framework for studying adversarial robustness in Concept Bottleneck Models through concept intervention.

## Requirements

### Validated

- ✓ CBM model with hybrid/fuzzy/relu/bool concept modes — existing
- ✓ VCBM (Virtual Concept Bottleneck Model) — existing
- ✓ CEM (Concept Embedding Model) — existing
- ✓ Backbone-only model variant — existing
- ✓ Multi-attack training: LPGD, CPGD, JPGD, APGD — existing
- ✓ AutoAttack evaluation — existing
- ✓ Concept intervention at test time — existing
- ✓ Multi-dataset support: CUB, CelebA, AwA — existing
- ✓ HSIC regularization for VCBM — existing
- ✓ Multi-task learning modes — existing

### Active

- [ ] Code cleanup: clean up imports, remove dead code, improve naming
- [ ] Reduce duplication: extract common logic between models
- [ ] Restructure: reorganize modules and files

### Out of Scope

- Adding new model architectures beyond CBM/VCBM/CEM
- Adding new datasets beyond CUB, CelebA, AwA
- Implementing new attack methods
- Building user-facing applications

## Context

This is an existing research codebase for adversarial robustness in Concept Bottleneck Models. The codebase map shows:
- Model layer: CBM, VCBM, CEM, backbone
- Data layer: CUB, CelebA, AwA datasets
- Attack layer: PGD-based attacks
- Training/Entry: main.py, test.py
- Utility: utils.py, hsic.py, mtl.py

Known areas for improvement from CONCERNS.md:
- Some unused imports
- Hardcoded values scattered in code
- Limited error handling
- Inconsistent naming conventions
- No formal testing framework

## Constraints

- **Tech Stack**: Python 3.12, PyTorch, Lightning - Must maintain compatibility
- **Dependencies**: Must work with existing dataset formats and model architectures
- **Functionality**: Must preserve all existing training and evaluation capabilities

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use Lightning for training | Existing pattern, works well | — Pending |
| Concept intervention at test time | Core research capability | — Pending |

---
*Last updated: 2026-03-12 after project initialization*
