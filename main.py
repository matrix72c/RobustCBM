import os
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch import seed_everything
import torch
import wandb
import yaml, argparse
import model as pl_model
import dataset
from utils import build_name, yaml_merge
import hashlib


def exp(config):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = yaml_merge(cfg, config)
    name = build_name(config)
    run_id = hashlib.md5(name.encode()).hexdigest()[:8]
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.get("seed", 42))

    dm = getattr(dataset, cfg["dataset"])(**cfg)
    model = getattr(pl_model, cfg["model"])(dm=dm, **cfg)

    logger = WandbLogger(
        project="RAID",
        name=name,
        id=run_id,
        resume="allow",
        config=cfg,
        group=cfg.get("group", None),
    )
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
    trainer = Trainer(
        log_every_n_steps=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.get("epochs", None),
        inference_mode=False,
    )
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint {ckpt_path} not found")
        print("Load from checkpoint: ", ckpt_path)
    else:
        trainer.fit(model, dm)
        ckpt_path = trainer.checkpoint_callback.best_model_path

    model = model.__class__.load_from_checkpoint(ckpt_path, dm=dm, **cfg)

    csv_logger = CSVLogger(save_dir="saved/", name=name)
    trainer = Trainer(
        log_every_n_steps=1,
        logger=[csv_logger, logger],
        inference_mode=False,
    )
    trainer.test(model, dm)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        c = yaml.safe_load(f)

    if isinstance(c, list):
        for config in c:
            exp(config)
    elif isinstance(c, dict):
        exp(c)
