# Python Refactoring Tools & Best Practices

**Project:** RobustCBM Refactoring
**Researched:** 2026-03-12
**Confidence:** HIGH

## Executive Summary

Python refactoring requires a combination of linters, formatters, static analyzers, and refactoring-specific tools. For a PyTorch Lightning codebase like RobustCBM, the recommended approach combines **Ruff** (fast linting/formatter), **Pyright** (type checking), **AST-based refactoring tools** (Rope, Bowler), and manual code review patterns.

---

## Recommended Tool Stack

### 1. Linting & Formatting

| Tool | Purpose | Version | Why Use |
|------|---------|---------|---------|
| **Ruff** | Linting + Formatting | Latest (0.9+) | 10-100x faster than alternatives, single tool for both linting and formatting, drop-in replacement for Flake8/isort/ autoflake |
| **Pyright** | Static type checking | Latest | Microsoft-backed, fast, excellent for PyTorch/Lightning type stubs |

**Installation:**
```bash
# Core
pip install ruff pyright

# Optional: for type stubs
pip install torch-typer  # If using CLI argument typing
```

**Configuration (pyproject.toml):**
```toml
[tool.ruff]
target-version = "py312"
line-length = 100
select = [
    "E",     # pycodestyle errors
    "W",     # pycodestyle warnings
    "F",     # pyflakes
    "I",     # isort
    "UP",    # pyupgrade
    "B",     # flake8-bugbear
    "C4",    # flake8-comprehensions
    "ASYNC", # flake8-async
]
ignore = [
    "E501",  # line too long (handled by formatter)
    "B008",  # do not perform function calls in argument defaults
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "basic"
reportMissingTypeStubs = false
```

**Usage:**
```bash
# Lint and fix
ruff check . --fix
ruff check . --fix --select I  # isort only

# Format
ruff format .

# Type check
pyright .
```

### 2. Dead Code Detection

| Tool | Purpose | When to Use |
|------|---------|--------------|
| **Vulture** | Find unused code | After initial cleanup, before major refactors |
| **Perflint** | Find performance issues | For optimization phase |

```bash
pip install vulture

# Find unused code
vulture . --min-confidence 80
```

### 3. AST-Based Refactoring Tools

| Tool | Purpose | Best For |
|------|---------|----------|
| **Rope** | Refactoring library | Rename refactors, extract methods, move functions |
| **Bowler** | Safe refactoring | Batch refactors, interface changes |
| **RedBaron** | AST manipulation | Complex code transformations |

```bash
pip install rope bowler

# Rope example: Rename function
python -c "
import rope.base.project
import rope.refactor.rename

project = rope.base.project.Project('.')
resource = project.get_resource('model/CBM.py')
renamer = rope.refactor.rename.Rename(project, resource, 'old_name')
changes = renamer.get_changes('new_name')
changes.apply()
project.close()
"

# Bowler example: Change all function signatures
python -c "
import bowler
bowler.Main().main([
    '--write',
    'model/',
    '--recursive',
    '-f', 'function_name',  # pattern
])
```

### 4. Import Management

| Tool | Purpose | Why |
|------|---------|-----|
| **reorder-python-imports** | Sort imports | Consistent ordering |
| ** autoflake** | Remove unused imports | Handled by Ruff now |

**Use Ruff instead** - it handles both:
```bash
ruff check . --select I --fix  # isort
ruff check . --select F401 --fix  # remove unused imports
```

### 5. Complexity Analysis

| Tool | Purpose | Metric |
|------|---------|--------|
| **Radon** | Cyclomatic complexity | Function complexity scores |
| **Wily** | Track complexity over time | Historical analysis |

```bash
pip install radon

# Measure complexity
radon cc -s model/  # cyclomatic complexity
radon mi -s model/  # maintainability index
```

---

## Best Practices for Refactoring

### Phase 1: Preparation

1. **Ensure tests exist** (or create minimal smoke tests)
   ```bash
   # Create test that verifies model loads
   pytest -xvs -k "test_model_loads"  # placeholder
   ```

2. **Commit current state**
   ```bash
   git add -A
   git commit -m "refactor: pre-refactoring checkpoint"
   ```

3. **Run baseline linter**
   ```bash
   ruff check . --output-format=json > baseline.json
   ```

### Phase 2: Incremental Refactoring

**Order of operations (recommended):**

1. **Formatting** - Ruff format (non-breaking)
   ```bash
   ruff format .
   ```

2. **Import cleanup** - Remove unused, sort imports
   ```bash
   ruff check . --select F401,I --fix
   ```

3. **Simple fixes** - Rename, extract (use Rope or manual)
   ```bash
   ruff check . --select N --fix  # name errors
   ruff check . --select UP --fix  # upgrade syntax
   ```

4. **Complex refactors** - Extract base classes, restructure
   - Use manual changes with AST guidance
   - Test after each logical unit

### Phase 3: Validation

1. **Run full linter**
   ```bash
   ruff check .
   pyright .
   ```

2. **Type checking**
   - Add type annotations where practical
   - Use `cast()` for complex PyTorch operations

3. **Functional verification**
   ```bash
   python main.py --config config.yaml  # quick training check
   python test.py --config test_config.yaml  # full test
   ```

---

## What NOT to Use and Why

| Tool | Why Avoid | Alternative |
|------|-----------|-------------|
| **Pyflakes** | Covered by Ruff | Use Ruff |
| **Flake8** | Slower, multiple plugins needed | Use Ruff |
| **isort** | Covered by Ruff | Use Ruff |
| **autopep8** | Formatting only, slower | Use Ruff |
| **pylint** | Slow, verbose | Use Ruff + Pyright |
| **_oldflake8 plugins_** | Many merged into Ruff | Use Ruff |

---

## RobustCBM-Specific Recommendations

### For this codebase, recommended workflow:

```bash
# 1. Initial assessment
ruff check . --output-format=json | jq '.[] | select(.fix | not) | .message' | sort | uniq -c | sort -rn

# 2. Quick wins (safe, automated)
ruff format .
ruff check . --fix --select F401,I,UP

# 3. Manual refactoring (requires review)
# - Extract base class for CBM/VCBM/CEM common logic
# - Move utility functions to proper modules
# - Add type hints to public APIs

# 4. Verify functionality
python main.py --config config.yaml
python test.py --config test_config.yaml
```

### Code areas to target (based on PROJECT.md):

| Issue | Tool | Approach |
|-------|------|----------|
| Unused imports | Ruff | `ruff check --select F401 --fix` |
| Hardcoded values | Manual | Search for magic numbers, extract to constants |
| Inconsistent naming | Ruff + manual | `N` rules + Rope rename |
| Dead code | Vulture | `vulture . --min-confidence 80` |
| Limited error handling | Manual | Add try/except with descriptive messages |

---

## Anti-Patterns to Avoid

1. **Don't refactor everything at once** - Incremental changes reduce risk
2. **Don't skip testing** - Run functional tests after each logical refactor
3. **Don't ignore linter warnings** - They indicate technical debt
4. **Don't over-abstract** - PyTorch Lightning already provides abstraction
5. **Don't remove type ignores without verification** - Some may be intentional

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Tools | HIGH | Ruff and Pyright are standard in 2025-2026 Python ecosystem |
| Best Practices | HIGH | Based on established Python patterns |
| Specific to PyTorch/Lightning | MEDIUM | General tools apply; framework-specific patterns are known |
| Tool recommendations | HIGH | Ruff has largely consolidated the linting space |

---

## Sources

- [Ruff Documentation](https://docs.astral.sh/ruff/) - Primary linter/formatter
- [Pyright Documentation](https://microsoft.github.io/pyright/) - Type checker
- [Rope Documentation](https://github.com/python-rope/rope) - Refactoring library
- [Vulture](https://github.com/jendrikseipp/vulture) - Dead code detection
- [Radon](https://radon.readthedocs.io/) - Complexity metrics
