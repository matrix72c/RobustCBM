---
phase: 02-code-cleanup
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dataset/AwA.py
  - dataset/CUB.py
  - dataset/__init__.py
  - model/__init__.py
  - model/CBM.py
autonomous: true
requirements:
  - CLEAN-01
  - CLEAN-02
  - CLEAN-03
  - CLEAN-04

must_haves:
  truths:
    - No unused imports remain in Python files
    - No dead code or unreachable code paths exist
    - All naming conventions consistent (snake_case, PascalCase)
    - Magic numbers extracted to named constants
  artifacts:
    - path: "dataset/AwA.py"
      provides: "Fixed unused import"
    - path: "dataset/CUB.py"
      provides: "Fixed unused imports"
    - path: "dataset/__init__.py"
      provides: "Removed unused exports"
    - path: "model/__init__.py"
      provides: "Removed unused exports"
    - path: "model/CBM.py"
      provides: "Removed unused variables"
  key_links:
    - from: "dataset/__init__.py"
      to: "dataset/CUB.py, dataset/AwA.py"
      via: "import statements"
    - from: "model/__init__.py"
      to: "model/CBM.py"
      via: "import statements"
---

<objective>
Remove code noise: unused imports, dead code, and establish consistent coding standards.
</objective>

<context>
@.planning/phases/02-code-cleanup/02-CONTEXT.md

Ruff issues (21 total):
- dataset/AwA.py: unused itertools
- dataset/CUB.py: unused itertools, unused os
- dataset/__init__.py: unused CUB.CUB, CUB.CustomCUB, AwA.AwA, celeba.celeba
- model/__init__.py: unused CBM.CBM, CEM.CEM, VCBM.VCBM, backbone.backbone

Vulture dead code (5 items):
- dataset/AwA.py: unused itertools (duplicate)
- dataset/CUB.py: unused itertools (duplicate)
- model/CBM.py: unused batch_idx (lines 277, 300, 343)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Remove unused imports (CLEAN-01)</name>
  <files>dataset/AwA.py, dataset/CUB.py, dataset/__init__.py, model/__init__.py</files>
  <action>
Fix all Ruff F401 unused import issues:
- dataset/AwA.py: Remove `itertools` import
- dataset/CUB.py: Remove `itertools` and `os` imports
- dataset/__init__.py: Remove unused class imports (CUB.CUB, CustomCUB, AwA.AwA, celeba.celeba)
- model/__init__.py: Remove unused class imports (CBM.CBM, CEM.CEM, VCBM.VCBM, backbone.backbone)

For __init__.py files, verify whether these classes are used elsewhere before removing. If they are public API exports, add to __all__ instead.
  </action>
  <verify>
ruff check --select F401 --fix
  </verify>
  <done>All Ruff F401 unused import errors resolved</done>
</task>

<task type="auto">
  <name>Task 2: Remove dead code (CLEAN-02)</name>
  <files>model/CBM.py</files>
  <action>
Remove unused variables identified by Vulture:
- Line 277: Remove unused `batch_idx` variable
- Line 300: Remove unused `batch_idx` variable
- Line 343: Remove unused `batch_idx` variable

These appear in training hooks - prefix with underscore if needed for signature compatibility.
  </action>
  <verify>
vulture model/CBM.py --min-confidence 90
  </verify>
  <done>No dead code remains in model/CBM.py</done>
</task>

<task type="auto">
  <name>Task 3: Verify naming conventions (CLEAN-03)</name>
  <files>model/, dataset/</files>
  <action>
Check for naming convention violations:
- snake_case for functions/variables
- PascalCase for classes

Run: ruff check --select N, C4
Fix any violations found in model/ and dataset/ directories.
  </action>
  <verify>
ruff check --select N,C4 model/ dataset/
  </verify>
  <done>All naming conventions consistent</done>
</task>

<task type="auto">
  <name>Task 4: Extract magic numbers (CLEAN-04)</name>
  <files>model/, dataset/</files>
  <action>
Search for magic numbers (hardcoded integers/strings) in model/ and dataset/:
- Look for values like 0.001, 1e-3, 32, 64, 100, etc.
- Extract to module-level constants with descriptive names

Focus on model/ and dataset/ directories as specified in CONTEXT.md.
  </action>
  <verify>
Manual review + grep for common magic numbers pattern
  </verify>
  <done>Reusable constants created for significant magic numbers</done>
</task>

</tasks>

<verification>
ruff check . --select F401,N,C4 && vulture model/ dataset/ --min-confidence 90
</verification>

<success_criteria>
- No Ruff F401 unused import errors
- No Vulture dead code warnings
- All naming conventions consistent
- Magic numbers extracted to constants
</success_criteria>

<output>
After completion, create .planning/phases/02-code-cleanup/02-01-SUMMARY.md
</output>
