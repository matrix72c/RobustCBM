import argparse
import hashlib
import logging
import os

import torch
import yaml
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer

import dataset
import model as pl_model
from utils import build_name

logger = logging.getLogger(__name__)

def load_checkpoint(ckpt):
    """Load a model from a checkpoint file.

    Args:
        ckpt: Checkpoint name (without extension).

    Returns:
        Tuple of (model, datamodule, config).

    Raises:
        FileNotFoundError: If checkpoint or config file not found.
        ValueError: If required config keys are missing or invalid.
        RuntimeError: If model or dataset class cannot be loaded.
    """
    ckpt_path = "checkpoints/" + ckpt + ".ckpt"
    cfg_path = "checkpoints/" + ckpt + ".yaml"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Validate required config keys
    required_keys = ["dataset", "model"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required config key: '{key}' in {cfg_path}")

    # Validate dataset class exists
    if not hasattr(dataset, cfg["dataset"]):
        raise RuntimeError(
            f"Dataset class '{cfg['dataset']}' not found in dataset module. "
            f"Available: {', '.join(dataset.__all__)}"
        )

    # Validate model class exists
    if not hasattr(pl_model, cfg["model"]):
        raise RuntimeError(
            f"Model class '{cfg['model']}' not found in model module. "
            f"Available: {', '.join([n for n in dir(pl_model) if not n.startswith('_')])}"
        )

    # Initialize datamodule with error handling
    try:
        dm = getattr(dataset, cfg["dataset"])(**cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize dataset '{cfg['dataset']}': {e}")

    # Validate datamodule has required attributes
    required_attrs = ["num_classes", "num_concepts"]
    for attr in required_attrs:
        if not hasattr(dm, attr):
            raise RuntimeError(
                f"DataModule '{cfg['dataset']}' missing required attribute '{attr}'"
            )

    # Load model from checkpoint
    try:
        model = getattr(pl_model, cfg["model"]).load_from_checkpoint(ckpt_path, dm=dm, **cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    return model, dm, cfg

def train(cfg):
    """Train a model with the given configuration.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Tuple of (model, datamodule, config).

    Raises:
        ValueError: If required config keys are missing or invalid.
        RuntimeError: If model or dataset initialization fails.
    """
    if cfg.get("ckpt", None) is not None:
        return load_checkpoint(cfg["ckpt"])

    # Validate required config keys
    required_keys = ["dataset", "model"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required config key: '{key}'")

    # Validate dataset class exists
    if not hasattr(dataset, cfg["dataset"]):
        raise RuntimeError(
            f"Dataset class '{cfg['dataset']}' not found in dataset module. "
            f"Available: {', '.join(dataset.__all__)}"
        )

    # Validate model class exists
    if not hasattr(pl_model, cfg["model"]):
        raise RuntimeError(
            f"Model class '{cfg['model']}' not found in model module. "
            f"Available: {', '.join([n for n in dir(pl_model) if not n.startswith('_')])}"
        )

    name = build_name(cfg)
    run_id = hashlib.md5(name.encode()).hexdigest()[:8]
    cfg["run_name"] = name
    cfg["run_id"] = run_id
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.get("seed", 42))
    yaml.dump(cfg, open(f"checkpoints/{name}.yaml", "w"))
    logger.info(f"Run ID: {run_id}, Run name: {name}")

    # Initialize datamodule with error handling
    try:
        dm = getattr(dataset, cfg["dataset"])(**cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize dataset '{cfg['dataset']}': {e}")

    # Validate datamodule has required attributes
    required_attrs = ["num_classes", "num_concepts"]
    for attr in required_attrs:
        if not hasattr(dm, attr):
            raise RuntimeError(
                f"DataModule '{cfg['dataset']}' missing required attribute '{attr}'"
            )

    # Initialize model with error handling
    try:
        model = getattr(pl_model, cfg["model"])(dm=dm, **cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model '{cfg['model']}': {e}")

    checkpoint_callback = ModelCheckpoint(
        monitor="acc",
        dirpath="checkpoints/",
        filename=name,
        save_top_k=1,
        mode="max",
        enable_version_counter=False,
        save_weights_only=True,
        every_n_epochs=1,
    )
    early_stopping = EarlyStopping(
        monitor="lr", mode="min", patience=1000, stopping_threshold=1e-4
    )
    callbacks = [checkpoint_callback, early_stopping]
    wandb_logger = WandbLogger(
        project="RAIDCXM",
        name=name,
    )
    trainer = Trainer(
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.get("epochs", -1),
        inference_mode=False,
    )
    trainer.fit(model, dm)
    model = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, dm=dm, **cfg)
    return model, dm, cfg

def evaluate(model, dm, cfg):
    wandb_logger = WandbLogger(
        project="RAIDCXM",
        name=cfg.get("run_name", "unknown"),
        id=cfg.get("run_id", None),
        resume="allow"
    )
    trainer = Trainer(logger=wandb_logger, inference_mode=False)
    trainer.test(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=0)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        c = yaml.safe_load(f)

    if args.task_id is not None:
        if args.task_id < len(c):
            model, dm, cfg = train(c[args.task_id])
            evaluate(model, dm, cfg)
    else:
        for i, config in enumerate(c):
            model, dm, cfg = train(config)
            evaluate(model, dm, cfg)
