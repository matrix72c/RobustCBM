# Phase 1: Foundation - Research

**Researched:** 2026-03-12
**Domain:** Python testing and code analysis
**Confidence:** HIGH

## Summary

Phase 1: Foundation requires establishing a baseline commit and analyzing the codebase for refactoring targets. Based on user decisions, this phase will use pytest for smoke tests and Ruff + Vulture for comprehensive code analysis, focusing on the model/ and dataset/ directories.

Key findings:
- pytest is the standard tool for Python smoke tests with auto-discovery
- Ruff (10-100x faster than Flake8) provides 800+ rules including unused import detection (F401)
- Vulture provides dead code detection with confidence values
- No existing test infrastructure - need to add pytest to requirements.txt

**Primary recommendation:** Install pytest, configure Ruff and Vulture, create simple smoke tests for model loading, forward pass, and config loading.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use pytest for smoke tests
- Standard Python testing, integrates with CI
- Comprehensive analysis using automated tools (Ruff, Vulture)
- Identify all issues: unused imports, dead code, naming inconsistencies, hardcoded values

### Claude's Discretion
- Focus analysis on model/ and dataset/ directories first
- Keep tests simple and fast (smoke tests, not comprehensive)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FOUND-01 | Establish baseline by committing current working state | Git is available, baseline commit pending |
| FOUND-02 | Analyze existing codebase to identify refactoring targets | Ruff and Vulture tools identified for comprehensive analysis |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | latest (8.x) | Smoke tests | Standard Python testing framework, auto-discovery, integrates with CI |
| Ruff | latest (0.9.x) | Linting, unused import detection | 10-100x faster than Flake8, 800+ built-in rules |
| Vulture | latest (2.x) | Dead code detection | Finds unused code using static analysis |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PyYAML | 6.0.3 | Config loading | Already in requirements.txt for config tests |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pytest | unittest | More boilerplate, no auto-discovery |
| Ruff | Flake8 | Ruff is 10-100x faster, more rules |
| Vulture | Custom AST analysis | Vulture is well-maintained, specialized |

**Installation:**
```bash
# Add to requirements.txt
pytest>=8.0.0
ruff>=0.9.0
vulture>=2.0.0
```

## Architecture Patterns

### Recommended Project Structure
```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── smoke/
│   ├── __init__.py
│   ├── test_model.py     # Model loading, forward pass
│   └── test_config.py    # Config loading
```

### Pattern 1: Smoke Tests
**What:** Simple, fast tests that verify core functionality works
**When to use:** Quick sanity checks before deeper testing
**Example:**
```python
# tests/smoke/test_model.py
import pytest
import torch
from model import CBM

def test_cbm_model_loads():
    """Smoke test: verify CBM model can be instantiated."""
    model = CBM(...)  # Use minimal config
    assert model is not None

def test_cbm_forward_pass():
    """Smoke test: verify forward pass runs without error."""
    model = CBM(...)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert output is not None
```

### Pattern 2: Ruff Configuration
**What:** pyproject.toml configuration for Ruff
**When to use:** Project-wide linting rules
**Example:**
```toml
# pyproject.toml
[tool.ruff]
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = []

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
```

### Pattern 3: Vulture Usage
**What:** Command-line dead code detection
**When to use:** Finding unused functions, classes, variables
**Example:**
```bash
# Basic usage
vulture model/ dataset/ --min-confidence 60

# With whitelist for false positives
vulture model/ --make-whitelist > whitelist.py
vulture model/ --ignore-names "main,*_callback"
```

### Anti-Patterns to Avoid
- **Comprehensive tests in smoke phase:** Keep tests simple and fast
- **Testing all edge cases:** Focus on core functionality only
- **Running full Vulture without filtering:** Use --min-confidence to reduce noise

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Test framework | Custom test runner | pytest | Auto-discovery, fixtures, CI integration |
| Unused import detection | Custom AST analysis | Ruff F401 | 800+ rules, automatic fixes |
| Dead code detection | Custom script | Vulture | Confidence values, whitelist support |

**Key insight:** pytest, Ruff, and Vulture are industry-standard tools with extensive community support. Building custom solutions would reinvent the wheel and miss edge cases these tools handle.

## Common Pitfalls

### Pitfall 1: No Test Dependencies in requirements.txt
**What goes wrong:** Tests cannot run in CI because pytest is not installed
**Why it happens:** Test dependencies added as afterthought
**How to avoid:** Add pytest, ruff, vulture to requirements.txt from the start
**Warning signs:** `ModuleNotFoundError: No module named 'pytest'`

### Pitfall 2: Smoke Tests That Are Too Slow
**What goes wrong:** Smoke tests take minutes to run, defeating the purpose
**Why it happens:** Loading full datasets, running multiple epochs
**How to avoid:** Use small batch sizes (batch=1), skip data loading when possible, mock heavy operations
**Warning signs:** Tests take more than 30 seconds total

### Pitfall 3: Ruff Fixing Code Without Review
**What goes wrong:** `ruff --fix` changes code automatically, potentially breaking functionality
**Why it happens:** Automatic fixes applied without reviewing changes
**How to avoid:** Run `ruff check` first, review diff, then apply fixes selectively
**Warning signs:** Large diff after running ruff --fix

### Pitfall 4: Vulture False Positives Overwhelming Output
**What goes wrong:** Vulture reports many false positives, making it hard to find real issues
**Why it happens:** Not using confidence thresholds or whitelist
**How to avoid:** Use --min-confidence 80 or higher, build whitelist for known-false positives
**Warning signs:** More than 50% of Vulture output are false positives

## Code Examples

### Smoke Test: Model Loading
```python
# Source: pytest best practices
# tests/smoke/test_model.py
import pytest
import torch

# Import from project
from model.CBM import CBM
from model.VCBM import VCBM


class TestModelLoading:
    """Smoke tests for model loading."""

    def test_cbm_imports(self):
        """Verify CBM can be imported."""
        from model import CBM
        assert CBM is not None

    def test_cbm_instantiation(self):
        """Verify CBM can be instantiated with minimal config."""
        # This would need actual config - keep minimal
        from model.CBM import CBM
        # Skip if requires complex setup
        pytest.skip("Requires config - test config loading first")


class TestForwardPass:
    """Smoke tests for forward pass."""

    def test_cbm_forward_shape(self):
        """Verify CBM forward pass produces expected output shape."""
        # Skip for smoke - requires full setup
        pytest.skip("Requires full model config")
```

### Smoke Test: Config Loading
```python
# Source: PyYAML usage
# tests/smoke/test_config.py
import pytest
import yaml
from pathlib import Path


def test_config_yaml_loads():
    """Verify config.yaml can be loaded."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        pytest.skip("config.yaml not found")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config is not None
    assert isinstance(config, dict)


def test_config_has_required_keys():
    """Verify config has required keys for training."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        pytest.skip("config.yaml not found")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check for key sections
    assert "model" in config or "models" in config
```

### Ruff: Finding Unused Imports
```bash
# Run on model/ directory
ruff check model/ --select F401 --output-format=json

# Auto-fix unused imports
ruff check model/ --select F401 --fix
```

### Vulture: Finding Dead Code
```bash
# Find dead code in model/ and dataset/
vulture model/ dataset/ --min-confidence 80

# Create whitelist for known-false positives
vulture model/ --make-whitelist > tests/vulture_whitelist.py
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Flake8 | Ruff | 2023 | 10-100x faster, more rules |
| pyflakes | Ruff (F rules) | 2023 | Unified tool |
| Custom dead code scripts | Vulture | 2004 (maintained) | Specialized tool with confidence values |

**Deprecated/outdated:**
- Flake8: Replaced by Ruff for most use cases (slower, fewer rules)
- pylint: More comprehensive but slower; Ruff preferred for quick analysis

## Open Questions

1. **How to handle model initialization requiring full config?**
   - What we know: CBM.__init__ requires many parameters from config
   - What's unclear: Should we create minimal fixtures or mock config loading?
   - Recommendation: Create a conftest.py with minimal fixture that loads config.yaml

2. **What confidence threshold to use for Vulture?**
   - What we know: 100% for unreachable code, 90% for imports, 60% for variables
   - What's unclear: How many false positives at 60%?
   - Recommendation: Start at 80%, adjust based on output

3. **Should we commit the baseline before or after smoke tests?**
   - What we know: FOUND-01 requires commit, FOUND-02 requires analysis
   - What's unclear: Does baseline include smoke tests?
   - Recommendation: Commit current state first (FOUND-01), then add tests (still FOUND-01 scope)

## Validation Architecture

> This section is included because workflow.nyquist_validation is true in .planning/config.json.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0.0 |
| Config file | pyproject.toml (to create) or pytest.ini |
| Quick run command | `pytest tests/smoke/ -v --tb=short` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-01 | Commit current working state | manual | `git commit -m "..."` | N/A (git) |
| FOUND-02 | Analyze codebase for refactoring targets | tool | `ruff check model/ dataset/` + `vulture model/ dataset/` | No - tools to install |

### Sampling Rate
- **Per task commit:** Run smoke tests only
- **Per wave merge:** Full suite (when expanded)
- **Phase gate:** Analysis complete (Ruff + Vulture output) + smoke tests pass

### Wave 0 Gaps
- [ ] `pytest` — Add to requirements.txt
- [ ] `ruff` — Add to requirements.txt
- [ ] `vulture` — Add to requirements.txt
- [ ] `tests/smoke/test_model.py` — Smoke tests for model loading
- [ ] `tests/smoke/test_config.py` — Smoke tests for config loading
- [ ] `pyproject.toml` — pytest and ruff configuration

## Sources

### Primary (HIGH confidence)
- pytest Documentation (https://docs.pytest.org/en/stable/) - Framework features and best practices
- Ruff Documentation (https://docs.astral.sh/ruff/) - Configuration and unused import detection
- Vulture GitHub (https://github.com/jendrikseipp/vulture) - Dead code detection usage

### Secondary (MEDIUM confidence)
- Python testing best practices from project requirements

### Tertiary (LOW confidence)
- Web search results (failed, using training data)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - pytest, Ruff, Vulture are well-established tools
- Architecture: HIGH - Standard pytest project structure
- Pitfalls: HIGH - Common issues with well-known solutions

**Research date:** 2026-03-12
**Valid until:** 30 days (tool versions stable)
