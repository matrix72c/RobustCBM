# Codebase Structure

**Analysis Date:** 2026-03-12

## Directory Layout

```
RobustCBM/
├── main.py                  # Training entry point
├── test.py                  # Testing entry point
├── config.yaml             # Default configuration
├── exps.yaml               # Experiment configurations
├── requirements.txt         # Python dependencies
├── utils.py                 # Utility functions
├── hsic.py                 # HSIC/CKA computation
├── mtl.py                  # Multi-task learning gradients
├── dataset/                # Dataset implementations
│   ├── __init__.py
│   ├── celeba.py           # CelebA dataset
│   ├── CUB.py              # CUB Birds dataset
│   └── AwA.py              # Animals with Attributes dataset
├── model/                  # Model implementations
│   ├── __init__.py
│   ├── CBM.py              # Concept Bottleneck Model
│   ├── VCBM.py             # Variational CBM
│   ├── CEM.py              # Concept Embedding Model
│   └── backbone.py          # Backbone-only model
├── attacks/                # Adversarial attack implementations
│   ├── __init__.py
│   ├── attack.py           # Base attack class
│   └── pgd.py             # PGD attack
├── checkpoints/            # Saved model checkpoints
└── data -> /mnt/...        # Symlink to data directory
```

## Directory Purposes

**Root Level:**
- Purpose: Entry points and core algorithms
- Contains: `main.py`, `test.py`, `utils.py`, `hsic.py`, `mtl.py`, config files

**Dataset Directory:**
- Purpose: Dataset implementations extending LightningDataModule
- Contains: Dataset classes for CelebA, CUB (Caltech-UCSD Birds), AwA (Animals with Attributes)
- Key files: `dataset/__init__.py`, `dataset/CUB.py`, `dataset/celeba.py`, `dataset/AwA.py`

**Model Directory:**
- Purpose: Neural network model implementations
- Contains: CBM variants (CBM, VCBM, CEM, backbone)
- Key files: `model/__init__.py`, `model/CBM.py`, `model/VCBM.py`, `model/CEM.py`, `model/backbone.py`

**Attacks Directory:**
- Purpose: Adversarial attack implementations
- Contains: Base Attack class, PGD attack
- Key files: `attacks/__init__.py`, `attacks/attack.py`, `attacks/pgd.py`

**Checkpoints Directory:**
- Purpose: Saved model weights and configs
- Contains: `.ckpt` model files, `.yaml` config files, test configs

## Key File Locations

**Entry Points:**
- `/home/jincheng1/RobustCBM/main.py`: Training pipeline - loads config, creates datamodule/model, runs trainer.fit() then evaluate()
- `/home/jincheng1/RobustCBM/test.py`: Testing pipeline - loads checkpoint, runs trainer.test() with multiple attack modes

**Configuration:**
- `/home/jincheng1/RobustCBM/config.yaml`: Default training config (model, dataset, optimizer, attack params)
- `/home/jincheng1/RobustCBM/exps.yaml`: Batch experiment configurations
- `/home/jincheng1/RobustCBM/checkpoints/celeba_test.yaml`: Test configs for CelebA
- `/home/jincheng1/RobustCBM/checkpoints/cub_test.yaml`: Test configs for CUB

**Core Logic:**
- `/home/jincheng1/RobustCBM/model/CBM.py`: Main CBM model implementation (~400 lines)
- `/home/jincheng1/RobustCBM/model/VCBM.py`: Variational CBM with VIB
- `/home/jincheng1/RobustCBM/model/CEM.py`: Concept Embedding Model
- `/home/jincheng1/RobustCBM/dataset/CUB.py`: CUB dataset with concept grouping
- `/home/jincheng1/RobustCBM/utils.py`: Utility functions (build_base, initialize_weights, cal_class_imbalance_weights)

**Testing:**
- `/home/jincheng1/RobustCBM/attacks/pgd.py`: PGD adversarial attack

## Naming Conventions

**Files:**
- CamelCase for dataset files: `celeba.py`, `CUB.py`, `AwA.py`
- CamelCase for model files: `CBM.py`, `VCBM.py`, `CEM.py`
- Snake_case for utilities: `utils.py`, `hsic.py`, `mtl.py`
- Lowercase for attack files: `attack.py`, `pgd.py`

**Classes:**
- CamelCase: `CBM`, `VCBM`, `CEM`, `celeba`, `CUB`, `AwA`, `PGD`, `Attack`
- Internal MLP: `class MLP` in `model/CBM.py`

**Functions:**
- Snake_case: `build_base()`, `initialize_weights()`, `cal_class_imbalance_weights()`, `build_name()`, `nhsic()`, `mtl()`

**Variables:**
- Snake_case: `concept_pred`, `label_pred`, `img`, `concept`, `label`
- Lowercase with underscore: `num_concepts`, `num_classes`, `batch_size`

## Where to Add New Code

**New Model Variant:**
- Primary code: `model/` directory - create new file (e.g., `model/NewModel.py`)
- Must subclass `L.LightningModule` and implement `forward()`, `training_step()`, `validation_step()`, `test_step()`, `configure_optimizers()`
- Export in `model/__init__.py`

**New Dataset:**
- Primary code: `dataset/` directory - create new file (e.g., `dataset/NewDataset.py`)
- Must subclass `L.LightningDataModule` and implement `train_dataloader()`, `val_dataloader()`, `test_dataloader()`
- Set attributes: `num_classes`, `num_concepts`, `concept_group_map`, `group_concept_map`, `max_intervene_budget`, `imbalance_weights`
- Export in `dataset/__init__.py`

**New Attack:**
- Primary code: `attacks/` directory - create new file (e.g., `attacks/new_attack.py`)
- Subclass `Attack` base class, implement `attack(model, x, y)` method

**New Utility:**
- Shared helpers: `utils.py` if general-purpose, or create new module if specific to a component

**New Experiment Config:**
- Add to `exps.yaml` or create new YAML file in root/checkpoints directory

## Special Directories

**Checkpoints:**
- Purpose: Model weights and configs saved during training
- Generated: Yes (during training)
- Committed: No (in .gitignore)

**Data:**
- Purpose: Symlink to external data storage
- Location: `/home/jincheng1/RobustCBM/data` -> `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/data/`
- Generated: No (external)
- Committed: No (symlink in .gitignore)

**Results:**
- Purpose: Test results CSV (referenced in code, created at runtime)
- Generated: Yes (during evaluation)
- Committed: No

---

*Structure analysis: 2026-03-12*
