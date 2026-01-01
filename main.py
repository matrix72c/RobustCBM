import os
import pandas as pd
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch import seed_everything
import torch
import yaml, argparse
import model as pl_model
import dataset
from utils import build_name, yaml_merge
import hashlib

def load_checkpoint(ckpt):
    ckpt_path = "checkpoints/" + ckpt + ".ckpt"
    cfg_path = "checkpoints/" + ckpt + ".yaml"

    if not os.path.exists(ckpt_path) or not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Checkpoint or config file not found for ckpt: {ckpt}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    dm = getattr(dataset, cfg["dataset"])(**cfg)
    model = getattr(pl_model, cfg["model"]).load_from_checkpoint(ckpt_path, dm=dm, **cfg)
    return model, dm, cfg

def train(cfg):
    if cfg.get("ckpt", None) is not None:
        return load_checkpoint(cfg["ckpt"])

    name = build_name(cfg)
    run_id = hashlib.md5(name.encode()).hexdigest()[:8]
    cfg["run_name"] = name
    cfg["run_id"] = run_id
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.get("seed", 42))
    yaml.dump(cfg, open(f"checkpoints/{name}.yaml", "w"))
    print(f"Run ID: {run_id}, Run name: {name}")

    dm = getattr(dataset, cfg["dataset"])(**cfg)
    model = getattr(pl_model, cfg["model"])(dm=dm, **cfg)

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
        logger=None,
        callbacks=callbacks,
        max_epochs=cfg.get("epochs", None),
        inference_mode=False,
    )
    trainer.fit(model, dm)
    model = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, dm=dm, **cfg)
    return model, dm, cfg

def evaluate(model, dm, cfg):
    trainer = Trainer(logger=None, inference_mode=False)
    res = trainer.test(model, dm)
    if res:
        result = res[0]
        sanitized = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                sanitized[k] = v.item()
            else:
                sanitized[k] = v
        sanitized["name"] = cfg.get("run_name", "unknown")
        sanitized["run_id"] = cfg.get("run_id", "unknown")

        row = pd.DataFrame([sanitized])
        results_path = "results/result.csv"
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            df = pd.concat([df, row], ignore_index=True)
        else:
            df = row
        df.to_csv(results_path, index=False)


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
