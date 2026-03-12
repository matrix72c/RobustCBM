# Coding Conventions

**Analysis Date:** 2026-03-12

## Language

**Primary:**
- Python 3.12

**Key Frameworks:**
- PyTorch 2.9.1 - Deep learning tensor operations
- Lightning 2.6.0 - Training loop abstraction
- Torchvision 0.24.1 - Vision models and transforms
- Torchmetrics 1.8.2 - Evaluation metrics

## Naming Patterns

**Files:**
- snake_case: `main.py`, `utils.py`, `test.py`
- Module directories use lowercase: `model/`, `dataset/`, `attacks/`

**Classes:**
- PascalCase: `CBM`, `VCBM`, `CUB`, `PGD`, `backbone`
- Dataset classes follow pattern: `{DatasetName}DataSet`, `{DatasetName}(LightningDataModule)`

**Functions:**
- snake_case: `build_name()`, `initialize_weights()`, `calc_info_loss()`
- Verb-noun pattern: `build_base()`, `cal_class_imbalance_weights()`, `flatten_dict()`

**Variables:**
- snake_case: `ckpt_path`, `run_id`, `num_classes`, `concept_weight`
- Private variables: underscore prefix (rarely used)

**Type Hints:**
- Used in function signatures:
  ```python
  def build_base(base, out_size, use_pretrained=True):
  def cal_class_imbalance_weights(dataset: torch.utils.data.Dataset):
  def modify_fc(model, base, out_size):
  ```

## Code Style

**Formatting:**
- 4 spaces indentation
- Line length: Not explicitly configured (no formatter found)
- No pre-commit hooks configured

**Linting:**
- Not detected

**Imports:**
- Standard library first
- Third-party packages (torch, lightning, numpy, etc.)
- Local modules last

Order in `model/CBM.py`:
```python
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy
from attacks import PGD
from autoattack import AutoAttack
from hsic import nhsic, standardize
from mtl import mtl
from utils import build_base, cls_wrapper, initialize_weights, suppress_stdout
```

## Function Design

**Size:**
- Medium-sized functions (20-100 lines)
- Complex logic in separate functions within same file

**Parameters:**
- Many parameters use `**kwargs` for extensibility:
  ```python
  class CBM(L.LightningModule):
      def __init__(
          self,
          dm: L.LightningDataModule,
          base: str = "resnet50",
          use_pretrained: bool = True,
          concept_weight: float = 1.0,
          optimizer: str = "SGD",
          optimizer_args: dict = {"lr": 0.1, "momentum": 0.9, "weight_decay": 4e-5},
          ...
          **kwargs,
      ):
  ```

**Return Values:**
- Multiple return patterns:
  - Single values: `return len(self.data)`
  - Tuples: `return {"label": label_pred, "concept": concept_pred}`
  - Dictionaries (loss dicts): `{"Label Loss": label_loss, "Loss": loss}`

## Class Design

**Inheritance:**
- Heavy use of inheritance:
  - `CBM(L.LightningModule)` - Base model class
  - `VCBM(CBM)` - Inherits from CBM
  - `CEM(CBM)` - Inherits from CBM
  - `backbone(CBM)` - Inherits from CBM
  - `PGD(Attack)` - Inherits from Attack base class

**Initialization:**
- `super().__init__()` called in all classes
- `save_hyperparameters(ignore="dm")` used in Lightning modules

## Error Handling

**Patterns:**
- Basic error checking with informative messages:
  ```python
  if not os.path.exists(ckpt_path) or not os.path.exists(cfg_path):
      raise FileNotFoundError(f"Checkpoint or config file not found for ckpt: {ckpt}")
  ```

- Value validation:
  ```python
  if base == "resnet50":
      ...
  else:
      raise ValueError("Unknown base model")
  ```

- NotImplementedError for unimplemented features:
  ```python
  def attack(self, model, x, y):
      raise NotImplementedError
  ```

**Logging:**
- Print statements for progress: `print(f"Run ID: {run_id}, Run name: {name}")`
- No logging framework detected (standard print)

## Comments

**When to Comment:**
- Minimal comments in code
- Some descriptive comments for complex logic:
  ```python
  # Generate a mapping containing all concept groups in CUB generated
  # using a simple prefix tree
  ```
- Chinese comments in utility functions:
  ```python
  def parse_value(value):
      """
      解析 CSV 文件中的值，可以解析数字与列表
      """
  ```

**Docstrings:**
- Rarely used in implementation files
- Used in `test.py` for main functions

## Configuration

**Approach:**
- YAML-based configuration files: `config.yaml`
- Python dictionaries for parameters: `optimizer_args`, `scheduler_args`
- Hydra-style config merging via `yaml_merge()`

## Module Design

**Exports:**
- `__init__.py` files expose classes: `from .CBM import CBM`
- Direct imports in main scripts: `import dataset`, `import model as pl_model`
- Dynamic class loading: `getattr(dataset, cfg["dataset"])`, `getattr(torch.optim, self.hparams.optimizer)`

**Barrel Files:**
- Simple `__init__.py` in each module directory
- No explicit re-exports or complex barrel patterns

---

*Convention analysis: 2026-03-12*
