# Technology Stack

**Analysis Date:** 2026-03-12

## Languages

**Primary:**
- Python 3.12 - Main development language for all code

## Runtime

**Environment:**
- Python 3.12
- Virtual environment: `.venv/` directory with venv

**Package Manager:**
- pip (via requirements.txt)
- Lockfile: Not present (requirements.txt only)

## Frameworks

**Core:**
- PyTorch 2.9.1 - Deep learning framework for model training and inference
- torchvision 0.24.1 - Computer vision utilities and pretrained models
- torchaudio 2.9.1 - Audio processing (installed as PyTorch dependency)

**Training:**
- Lightning 2.6.0 - PyTorch Lightning framework for training orchestration

**Testing:**
- torchmetrics 1.8.2 - Metrics computation for model evaluation

**Data Processing:**
- pandas 2.3.3 - Data manipulation and CSV result handling
- numpy 2.4.0 - Numerical operations
- Pillow 12.0.0 - Image loading and processing

**Adversarial Robustness:**
- autoattack 0.1 - Adversarial attack framework for evaluation

## Key Dependencies

**Deep Learning:**
- torch 2.9.1 - Core PyTorch library
- torchvision 0.24.1 - Vision models and transforms
- lightning 2.6.0 - Training framework

**Model Components:**
- scikit_learn 1.8.0 - Machine learning utilities (not directly imported, potential use for metrics)
- torchmetrics 1.8.2 - Accuracy and other metrics

**Data Processing:**
- pandas 2.3.3 - CSV results logging
- numpy 2.4.0 - Array operations
- Pillow 12.0.0 - Image loading

**Configuration & Utilities:**
- PyYAML 6.0.3 - YAML config file parsing
- autoattack 0.1 - Adversarial attacks

## Configuration

**Environment:**
- Python version specified in `.python-version` (3.12)
- Dependencies in `requirements.txt`
- Configuration via YAML files (`config.yaml`, `exps.yaml`, `tmp.yaml`)

**Build:**
- No build system (pure Python project)
- Virtual environment in `.venv/`

## Platform Requirements

**Development:**
- Python 3.12
- CUDA-capable GPU (for PyTorch training with GPU acceleration)
- Linux environment (tested on Linux 5.10.25-nvidia-gpu)

**Production:**
- PyTorch-compatible GPU environment
- Data stored in external path: `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/data/`
- Results and checkpoints stored locally in `checkpoints/` and `results/` directories

---

*Stack analysis: 2026-03-12*
