# External Integrations

**Analysis Date:** 2026-03-12

## APIs & External Services

**Experiment Tracking:**
- Weights & Biases (WandbLogger) - Experiment logging and visualization
  - Integration: `lightning.pytorch.loggers.WandbLogger`
  - Project name: "RAIDCXM"
  - Mode: Offline (training runs logged locally without network connection)
  - Location: `/home/jincheng1/RobustCBM/wandb/` (ignored in git)

**Adversarial Attacks:**
- AutoAttack - Robustness evaluation
  - Package: `autoattack==0.1`
  - Used in: `model/CBM.py` for model robustness testing
  - Config: `aa_args` in config.yaml (default epsilon: 0.0156862745)

## Data Storage

**Datasets:**
- CUB-200-2011 (Caltech-UCSD Birds)
  - Data path: `./data/CUB_200_2011/` (symlinked to external storage)
  - External storage: `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/data/`
  - Loaded via: `dataset/CUB.py`

- AwA (Animals with Attributes)
  - Loaded via: `dataset/AwA.py`

- CelebA
  - Loaded via: `dataset/celeba.py`

**File Storage:**
- Local filesystem for checkpoints
  - Path: `checkpoints/` directory
  - Format: PyTorch checkpoint (.ckpt)
  - YAML config saved alongside checkpoints

- Local filesystem for results
  - Path: `results/result.csv`
  - Format: CSV for aggregated test results

**Caching:**
- Image caching via PIL
- PyTorch DataLoader with multiprocessing (num_workers=12-24)

## Authentication & Identity

**Experiment Tracking:**
- No authentication required (offline mode)
- wandb disabled in .gitignore

## Monitoring & Observability

**Training Logs:**
- Weights & Biases (offline mode)
  - Logs: Metrics, training progress, configuration
  - Disabled for local-only runs

**Test Results:**
- CSV logging in `results/result.csv`
  - Tracks accuracy metrics across different attack modes

## CI/CD & Deployment

**Hosting:**
- Not applicable (research codebase)
- No cloud deployment

**CI Pipeline:**
- None configured

## Environment Configuration

**Required env vars:**
- None explicitly required (offline mode)
- CUDA_VISIBLE_DEVICES (implicit for GPU training)

**Secrets location:**
- No secrets stored in codebase
- .env files ignored in .gitignore

**Data paths:**
- Data directory symlinked to external storage: `data -> /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/data/`

## Webhooks & Callbacks

**Training Callbacks:**
- ModelCheckpoint (saves best model based on accuracy)
- EarlyStopping (monitors learning rate, patience=1000)

**Experiment Tracking:**
- Lightning callbacks for metric logging

## Pretrained Models

**Computer Vision:**
- torchvision.models - Backbone networks
  - resnet50 (default)
  - vit_b_16 (Vision Transformer)
  - vgg16
  - inception_v3
- Weights: DEFAULT (ImageNet pretrained)
- Loaded via: `utils.build_base()` function

---

*Integration audit: 2026-03-12*
