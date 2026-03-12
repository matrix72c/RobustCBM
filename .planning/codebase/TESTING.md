# Testing Patterns

**Analysis Date:** 2026-03-12

## Test Framework

**Status:** No formal test framework configured

**What Exists:**
- No `pytest`, `unittest`, or `pytest.ini` configuration
- No test directory or test discovery patterns
- Testing is performed via Lightning's built-in testing infrastructure

**Evaluation Approach:**
- `main.py` - Training and evaluation entry point
- `test.py` - Standalone script for testing trained checkpoints
- Both use Lightning's `Trainer.test()` method

## Lightning Testing

**Test Step Implementation:**

The codebase uses Lightning's `test_step()` method in models. Example from `model/CBM.py`:

```python
def test_step(self, batch, batch_idx):
    img, label, concepts = batch

    if self.hparams.get("ignore_adv", False):
        modes = ["Std"]
    else:
        modes = ["Std", "LPGD", "CPGD", "JPGD", "APGD", "AA"]

    for mode in modes:
        if self.hparams.model == "backbone" and (mode == "CPGD" or mode == "JPGD"):
            continue
        if mode == "Std":
            x = img
        else:
            x = self.generate_adv(img, label, concepts, mode)
        pred = self(x)
        label_pred, concept_pred = pred["label"], pred["concept"]
        semantic_concept_pred = concept_pred[:, : self.num_concepts]
        acc_metric = getattr(self, f"{mode}_acc")
        acc_metric(label_pred, label)
        self.log(
            f"{mode} Acc",
            acc_metric,
            on_step=False,
            on_epoch=True,
        )
```

**Test Entry Points:**

```python
# main.py - Combined train/eval
def evaluate(model, dm, cfg):
    logger = WandbLogger(...)
    trainer = Trainer(logger=logger, inference_mode=False)
    res = trainer.test(model, dm)

# test.py - Standalone testing
def test_model(config):
    trainer = Trainer(
        logger=False,
        enable_progress_bar=True,
        inference_mode=False,
    )
    test_results = trainer.test(model, dm)
```

## Test File Organization

**Location:**
- Not applicable - no dedicated test directory

**Alternative Testing:**
- `test.py` - Located at `/home/jincheng1/RobustCBM/test.py`
- Contains `test_model()` and `batch_test()` functions
- Focuses on loading checkpoints and running evaluations

## Test Structure

**Suite Organization:**

Not applicable for unit tests. However, Lightning uses:
- `training_step()` - Training loop
- `validation_step()` - Validation during training
- `test_step()` - Test-time evaluation

## Validation Approach

**Validation Step Pattern:**

```python
def validation_step(self, batch, batch_idx):
    img, label, concepts = batch
    if self.train_mode != "Std":
        img = self.generate_adv(img, label, concepts, self.train_mode)
    pred = self(img)
    losses = self.calc_loss({"label": label, "concept": concepts}, pred)
    label_pred, concept_pred = pred["label"], pred["concept"]
    semantic_concept_pred = concept_pred[:, : self.num_concepts]
    self.concept_acc(semantic_concept_pred, concepts)
    self.acc(label_pred, label)
    for name, val in losses.items():
        self.log(f"{name}", val, on_step=False, on_epoch=True)
```

## Metrics

**Using TorchMetrics:**

```python
self.concept_acc = Accuracy(task="multilabel", num_labels=num_concepts)
self.acc = Accuracy(task="multiclass", num_classes=num_classes)
```

Dynamic metric creation:
```python
for s in ["Std", "LPGD", "CPGD", "JPGD", "APGD", "AA"]:
    setattr(
        self,
        f"{s}_acc",
        Accuracy(task="multiclass", num_classes=num_classes),
    )
```

## Mocking

**Framework:** None detected

**Approach:**
- No mocking framework (no unittest.mock, pytest-mock, etc.)
- Real model/data used in all testing
- GPU/CPU device handling via `.to(self.device)`

## Fixtures and Factories

**Test Data:**
- Dataset classes serve as data fixtures: `CUB`, `AwA`, `celeba`
- Configuration-driven: `config.yaml` specifies dataset and model parameters

## Test Coverage

**Requirements:** None enforced

**Current Coverage:**
- No coverage tracking
- Manual evaluation via wandb logs and CSV results

## Test Types

**Unit Tests:**
- Not present in the codebase

**Integration Tests:**
- Done via Lightning's Trainer with real data loaders
- `test_step()` evaluates full model pipeline

**E2E Tests:**
- `main.py` - Full training pipeline with `train()` + `evaluate()`
- `test.py` - Checkpoint loading and evaluation

## Running Tests

**Train/Eval Pipeline:**
```bash
python main.py --config config.yaml
```

**Standalone Testing:**
```bash
python test.py --config config.yaml
```

**With Specific Task:**
```bash
python main.py --config config.yaml --task_id 0
python test.py --config config.yaml --task_id 0
```

## Common Patterns

**Test Configuration:**
```python
# Load checkpoint
ckpt_path = "checkpoints/" + config["ckpt"] + ".ckpt"
cfg_path = "checkpoints/" + config["ckpt"] + ".yaml"

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

dm = getattr(dataset, cfg["dataset"])(**cfg)
model = getattr(pl_model, cfg["model"]).load_from_checkpoint(ckpt_path, dm=dm, **cfg)
```

**Adversarial Evaluation:**
```python
def generate_adv(self, img, label, concepts, atk):
    if atk == "JPGD":
        adv_img = self.jpgd(self, img, {"label": label, "concept": concepts})
    elif atk == "AA":
        adv_img = self.aa.run_standard_evaluation(img, label, bs=img.shape[0])
    ...
```

---

*Testing analysis: 2026-03-12*
