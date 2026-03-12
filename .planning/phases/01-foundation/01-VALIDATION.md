---
phase: 01
slug: foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none — Wave 0 creates |
| **Quick run command** | `python -m pytest tests/smoke/ -v` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/smoke/ -v`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | FOUND-01 | smoke | `python -c "import torch; from model import CBM"` | ⬜ pending |
| 01-01-02 | 01 | 1 | FOUND-01 | smoke | `python -c "import dataset; print(dataset.__file__)"` | ⬜ pending |
| 01-02-01 | 01 | 2 | FOUND-02 | analysis | `ruff check model/ dataset/ --select=F401` | ⬜ pending |
| 01-02-02 | 01 | 2 | FOUND-02 | analysis | `vulture model/ dataset/ --min-confidence=80` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/smoke/` — smoke test stubs for model loading, forward pass, config loading
- [ ] `tests/conftest.py` — shared fixtures for config loading
- [ ] Add pytest to requirements.txt

---

## Success Criteria

1. **Baseline commit exists** — git shows refactoring commits separate from docs
2. **Smoke tests pass** — model loading, forward pass, config loading all work
3. **Analysis tools run** — Ruff and Vulture produce output without errors
4. **Analysis report generated** — list of refactoring targets identified
