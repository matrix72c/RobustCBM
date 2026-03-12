# Architecture

**Analysis Date:** 2026-03-12

## Pattern Overview

**Overall:** PyTorch Lightning-based Concept Bottleneck Model (CBM) Framework

**Key Characteristics:**
- Uses Lightning for training loop abstraction and distributed training support
- Implements Concept Bottleneck Models (CBM) where images are first mapped to concept predictions, then labels are predicted from concepts
- Supports multiple CBM variants (CBM, VCBM, CEM) with configurable concept-to-label mapping
- Integrates adversarial attack training and evaluation (PGD-based attacks, AutoAttack)
- Implements concept intervention at test time for robustness analysis

## Layers

**Model Layer:**
- Purpose: Neural network models for concept prediction and label classification
- Location: `model/`
- Contains: `CBM.py`, `VCBM.py`, `CEM.py`, `backbone.py`
- Depends on: PyTorch, torchvision (backbones), lightning
- Used by: Training loop in `main.py`

**Data Layer:**
- Purpose: Dataset loading, preprocessing, and concept management
- Location: `dataset/`
- Contains: `celeba.py`, `CUB.py`, `AwA.py`
- Depends on: PyTorch datasets, torchvision transforms, PIL
- Used by: Lightning trainer via LightningDataModule

**Attack Layer:**
- Purpose: Adversarial attack implementations for training and evaluation
- Location: `attacks/`
- Contains: `attack.py` (base class), `pgd.py` (PGD attack)
- Depends on: PyTorch
- Used by: Model training (`CBM.py`) and testing steps

**Training/Entry Layer:**
- Purpose: Main training pipeline, configuration, checkpoint management
- Location: `main.py`, `test.py`
- Depends on: lightning.pytorch, model, dataset
- Invokes: Model training, evaluation, checkpoint loading

**Utility Layer:**
- Purpose: Shared helper functions and algorithms
- Location: `utils.py`, `hsic.py`, `mtl.py`
- Contains: Weight initialization, model building, HSIC computation, multi-task learning gradients
- Used by: All layers

## Data Flow

**Training Flow:**
1. `main.py` loads config from YAML (`--config` argument)
2. Creates dataset via `dataset.{DatasetName}(**cfg)` - e.g., `dataset.CUB(**cfg)`
3. Creates model via `pl_model.{ModelName}(dm=dm, **cfg)` - e.g., `model.CBM(dm=dm, **cfg)`
4. Lightning Trainer runs `model.training_step()`:
   - Batch: `(img, label, concepts)` from dataloader
   - Forward: `base(img)` -> concept predictions -> `classifier(concepts)` -> label predictions
   - Loss: combines label loss (cross-entropy) + concept loss (binary cross-entropy)
5. After training, saves checkpoint to `checkpoints/{name}.ckpt` and config to `checkpoints/{name}.yaml`

**Testing Flow:**
1. `main.py` or `test.py` loads checkpoint via `ModelClass.load_from_checkpoint()`
2. Runs `trainer.test()` which calls `model.test_step()`
3. For each attack mode (Std, LPGD, CPGD, JPGD, APGD, AA):
   - Generates adversarial examples (if not Std)
   - Runs forward pass
   - Computes accuracy metrics
4. Optionally runs intervention experiments at different budgets (0-100%)

**Concept Intervention:**
1. During test, for each image and intervention budget:
   - Compare predicted concepts to ground truth
   - Identify groups where prediction differs from ground truth
   - Override concept predictions with either positive or negative concept logits
   - Re-run classification with intervened concepts

## Key Abstractions

**LightningDataModule (Dataset):**
- Purpose: Encapsulates dataset, transforms, concept metadata
- Examples: `dataset/CUB.py:CUB`, `dataset/celeba.py:celeba`, `dataset/AwA.py:AwA`
- Pattern: Subclass of `L.LightningDataModule` with `train_dataloader()`, `val_dataloader()`, `test_dataloader()` methods
- Key attributes: `num_classes`, `num_concepts`, `concept_group_map`, `group_concept_map`, `imbalance_weights`

**LightningModule (Model):**
- Purpose: Encapsulates model architecture and training logic
- Examples: `model/CBM.py:CBM`, `model/VCBM.py:VCBM`, `model/CEM.py:CEM`
- Pattern: Subclass of `L.LightningModule` with `forward()`, `training_step()`, `validation_step()`, `test_step()`, `configure_optimizers()`
- Returns: `{"label": label_pred, "concept": concept_pred}` from forward

**Attack (Base Class):**
- Purpose: Abstract interface for adversarial attacks
- Examples: `attacks/pgd.py:PGD`
- Pattern: Subclass with `attack(model, x, y)` method that returns adversarial examples

## Entry Points

**Training:**
- Location: `main.py`
- Triggers: `python main.py --config config.yaml`
- Responsibilities: Parse config, create datamodule and model, set up trainer with callbacks (ModelCheckpoint, EarlyStopping) and logger (WandbLogger), run training, evaluate on test set, save results to CSV

**Testing:**
- Location: `test.py`
- Triggers: `python test.py --config test_config.yaml`
- Responsibilities: Load model from checkpoint, run test suite with multiple attack modes, optionally run intervention experiments

**Model Building:**
- Location: `utils.py:build_base()`
- Triggers: Called by model `__init__`
- Responsibilities: Create torchvision backbone (resnet50, vit, vgg16, inceptionv3) with modified final FC layer for concept prediction

## Error Handling

**Strategy:** Python exceptions with descriptive error messages

**Patterns:**
- `FileNotFoundError`: Checkpoint/config not found in `load_checkpoint()`
- `NotImplementedError`: Unknown attack mode in `generate_adv()`
- `ValueError`: Unknown base model in `build_base()`, unknown MTL mode in `mtl()`
- Config validation: Type checking in dataset/model initialization

## Cross-Cutting Concerns

**Logging:** WandbLogger (Weights & Biases) - configured in `main.py` and `test.py`

**Validation:** Imbalance weights for weighted BCE loss, configured per dataset

**Authentication:** Not applicable (no user auth)

**Checkpointing:** ModelCheckpoint callback saves best model by validation accuracy

**Randomness:** seed_everything() from Lightning sets all random seeds

---

*Architecture analysis: 2026-03-12*
