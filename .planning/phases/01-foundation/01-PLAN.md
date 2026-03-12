---
phase: 01-foundation
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - requirements.txt
  - tests/smoke/test_model.py
  - tests/smoke/test_config.py
  - tests/conftest.py
  - tests/__init__.py
  - tests/smoke/__init__.py
autonomous: true
requirements:
  - FOUND-01
  - FOUND-02
user_setup: []
must_haves:
  truths:
    - "Testing infrastructure exists (pytest can run)"
    - "Model classes can be imported from model/ directory"
    - "Config file can be loaded with yaml.safe_load"
    - "Code analysis tools run without errors (Ruff + Vulture complete)"
  artifacts:
    - path: "requirements.txt"
      provides: "Test dependencies (pytest, ruff, vulture)"
      contains: "pytest"
    - path: "tests/smoke/test_model.py"
      provides: "Smoke tests for model imports"
      min_tests: 3
    - path: "tests/smoke/test_config.py"
      provides: "Smoke tests for config loading"
      min_tests: 2
    - path: "tests/conftest.py"
      provides: "Shared pytest fixtures"
      contains: "fixture"
  key_links:
    - from: "tests/smoke/test_model.py"
      to: "model"
      via: "import"
      pattern: "from model import"
    - from: "tests/smoke/test_config.py"
      to: "config.yaml"
      via: "yaml.safe_load"
      pattern: "yaml.safe_load"
---

<objective>
Establish baseline state and analyze codebase for refactoring targets.

Purpose: Create testing infrastructure and identify refactoring targets (unused imports, dead code, naming issues, hardcoded values) in model/ and dataset/ directories.

Output: Smoke tests for model loading/forward pass/config loading, plus Ruff + Vulture analysis report.
</objective>

<execution_context>
@/home/jincheng1/.claude/get-shit-done/workflows/execute-plan.md
@/home/jincheng1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/01-foundation/01-CONTEXT.md
@.planning/phases/01-foundation/01-RESEARCH.md
@.planning/phases/01-foundation/01-VALIDATION.md
@.planning/codebase/STRUCTURE.md

# Key Interfaces

From model/__init__.py (expected exports):
- CBM, VCBM, CEM, Backbone classes from model module
- LightningModule subclasses

From dataset/__init__.py (expected exports):
- CUB, CelebA, AwA dataset classes
- LightningDataModule subclasses

From config.yaml:
- dict structure with model, dataset, optimizer sections
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add testing dependencies to requirements.txt</name>
  <files>requirements.txt</files>
  <action>
    Add pytest, ruff, and vulture to requirements.txt:
    - pytest>=8.0.0
    - ruff>=0.9.0
    - vulture>=2.0.0

    Append these lines to the existing requirements.txt file.
  </action>
  <verify>
    <automated>grep -q "pytest" requirements.txt && grep -q "ruff" requirements.txt && grep -q "vulture" requirements.txt</automated>
  </verify>
  <done>requirements.txt contains pytest, ruff, vulture with version specifiers</done>
</task>

<task type="auto">
  <name>Task 2: Create smoke test infrastructure</name>
  <files>tests/__init__.py, tests/smoke/__init__.py, tests/conftest.py, tests/smoke/test_model.py, tests/smoke/test_config.py</files>
  <action>
    Create the following test files:

    1. tests/__init__.py - Empty file
    2. tests/smoke/__init__.py - Empty file
    3. tests/conftest.py - Shared pytest fixtures:
       - Fixture that loads config.yaml (if exists, else skip)
       - Fixture providing minimal model config dict
    4. tests/smoke/test_model.py - Model smoke tests:
       - test_cbm_imports: Verify CBM can be imported from model
       - test_vcbm_imports: Verify VCBM can be imported
       - test_cem_imports: Verify CEM can be imported
       - test_dataset_imports: Verify dataset classes can be imported
    5. tests/smoke/test_config.py - Config smoke tests:
       - test_config_yaml_loads: Verify config.yaml can be loaded with yaml.safe_load
       - test_config_has_required_keys: Verify config has 'model' or 'models' key

    Keep tests simple - use pytest.skip for tests requiring full model initialization.
  </action>
  <verify>
    <automated>python -m pytest tests/smoke/ --collect-only 2>/dev/null | grep -c "test_"</automated>
  </verify>
  <done>5+ smoke tests collected, test files created in tests/smoke/</done>
</task>

<task type="auto">
  <name>Task 3: Run codebase analysis with Ruff and Vulture</name>
  <files>N/A - analysis only</files>
  <action>
    Run analysis tools on model/ and dataset/ directories:

    1. Run Ruff to find unused imports (F401):
       ruff check model/ dataset/ --select F401 --output-format=json > .planning/phases/01-foundation/ruff-unused-imports.json || true

    2. Run Ruff for general linting:
       ruff check model/ dataset/ --output-format=json > .planning/phases/01-foundation/ruff-general.json || true

    3. Run Vulture to find dead code:
       vulture model/ dataset/ --min-confidence=80 --json > .planning/phases/01-foundation/vulture-dead-code.json || true

    Save outputs as JSON files for structured analysis. Use || true to allow commands to continue even if issues found.
  </action>
  <verify>
    <automated>ls -la .planning/phases/01-foundation/ruff-*.json .planning/phases/01-foundation/vulture-*.json 2>/dev/null | wc -l</automated>
  </verify>
  <done>Analysis JSON files created in phase directory (ruff-unused-imports.json, ruff-general.json, vulture-dead-code.json)</done>
</task>

</tasks>

<verification>
1. Run smoke tests: `python -m pytest tests/smoke/ -v --tb=short`
   - Expected: All tests collected, skipped tests marked appropriately
2. Verify analysis tools ran: Check JSON output files exist
3. Quick check: Import test - `python -c "from model import CBM; print('CBM imported')"`
</verification>

<success_criteria>
- Smoke tests exist for model loading and config loading
- Ruff analysis completed, unused imports identified
- Vulture analysis completed, dead code identified
- All test dependencies added to requirements.txt
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation/01-SUMMARY.md`
</output>
