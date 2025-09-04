# RAIDCXM (Under Review)

Code for our double‑blind submission on securing Concept-based Models (CxMs) under adversarial threat while preserving human intervention effectiveness.

## Motivation & Background
Concept-based Models insert a human-understandable concept layer between a pretrained backbone and the classifier, enabling explanation and *human intervention* (manual concept correction). 
Prior work emphasized concept prediction robustness but rarely measured the end-to-end robustness and how interventions interact with adversarial attacks. We evaluate both pre‑intervention and post‑intervention adversarial robustness.

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Prepare datasets (see Dataset Preparation).
3. Create override YAML list (e.g., `tmp.yaml`).
4. Run an experiment:
```bash
python main.py --config tmp.yaml --task_id 0
```

5. Omit `--task_id` to run all entries.

## Configuration System
Defaults reside in `config.yaml`. Override file must be a YAML list; each element merges with defaults (recursive) via `utils.yaml_merge`. The merged config and checkpoint are stored under `checkpoints/`; run logs go to `logs/<run_name>/<run_id>/` with deterministic `run_name` + hashed `run_id`.

## Running Experiments
Example `cfg.yaml`:
```yaml
- model: CBM
  dataset: CUB
  train_mode: Std
# below is config of RAIDCXM
- model: VCBM
  dataset: CUB
  train_mode: JPGD
  res_dim: 32
  hsic_weight: 0.0001
```
Run the second entry:
```bash
python main.py --config cfg.yaml --task_id 1
```

## Dataset Preparation
Expected structure under `data/`:
```
CUB_200_2011/
  images/ ...
  attributes/
  train.pkl / val.pkl / test.pkl
Animals_with_Attributes2/
  JPEGImages/ ...
  classes.txt
  predicates.txt
  predicate-matrix-binary.txt
```

Ensure `data_path` points to the parent directory. To add a dataset, implement a loader class under `dataset/`.
For CUB, `*.pkl` files can be found in CBM's repository [here](https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683).
