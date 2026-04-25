# Repository Guidelines

## Project Structure & Module Organization

This Python research codebase trains and evaluates robust concept-based models through `main.py`.

- `model/`: Concept model implementations (`CBM.py`, `CEM.py`, `VCBM.py`) plus shared backbones.
- `dataset/`: Lightning data modules for `CUB` and `AwA`; add new datasets here and export them from `dataset/__init__.py`.
- `attacks/`: Adversarial attack utilities, including PGD.
- `hsic.py`, `utils.py`: Shared losses, config merging, naming, and helpers.
- `data/`: Symlinked dataset root. Do not commit datasets, checkpoints, logs, or W&B outputs.

## Build, Test, and Development Commands

- `uv sync`: Install dependencies from `pyproject.toml` and `uv.lock`.
- `python main.py --config tmp.yaml --task_id 0`: Run one experiment entry from a YAML list config.
- `python main.py --config cfg.yaml --task_id 1`: Run entry index `1`.

Configs must be YAML lists. Each item is merged with defaults defined in model and dataset classes. The current CLI defaults to `task_id=0`; invoke task IDs separately for multiple entries unless the CLI is changed.

## Coding Style & Naming Conventions

Use Python 3.12 style with 4-space indentation. Prefer explicit validation and clear errors, matching `main.py`. Model and dataset classes use `PascalCase` and must match config values exactly, for example `model: VCBM` or `dataset: CUB`. Functions, variables, and config keys use `snake_case`. Group imports as standard library, third-party, then local.

No formatter or linter is configured in `pyproject.toml`; keep edits consistent with existing files and avoid unrelated formatting churn.

## Testing Guidelines

There is no formal test suite in the current tree. For model, dataset, or attack changes, validate with a small YAML config and run one training/evaluation pass:

```bash
python main.py --config tmp.yaml --task_id 0
```

For checkpoint-loading changes, verify both the `.ckpt` and matching `.yaml` under `checkpoints/`. For dataset changes, confirm the `data/` layout described in `README.md`.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit-style messages such as `feat(results): ...`, `refactor(dataset): ...`, `docs: ...`, and `chore: ...`. Keep each commit focused.

Pull requests should include a summary, validation config, dataset assumptions, and notes about generated artifacts. Do not commit large data files, checkpoints, logs, or local notebook outputs unless required.

## Configuration & Security Tips

Weights & Biases logging is enabled through `WandbLogger`; configure credentials outside the repository. Keep machine-specific paths in local YAML overrides, not committed defaults. Treat `tmp.yaml` as local scratch unless the configuration is intentionally reusable.

## Agent-Specific Instructions

When interacting with the repository owner, use Chinese by default.
