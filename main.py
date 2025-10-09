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


def build(config):
    if config.get("ckpt", None) is not None:
        ckpt_path = "checkpoints/" + config["ckpt"] + ".ckpt"
        cfg_path = "checkpoints/" + config["ckpt"] + ".yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        dm = getattr(dataset, cfg["dataset"])(**cfg)
        model = getattr(pl_model, cfg["model"]).load_from_checkpoint(ckpt_path, dm=dm, **cfg)
        return model, dm, cfg

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = yaml_merge(cfg, config)
    name = build_name(config)
    run_id = hashlib.md5(name.encode()).hexdigest()[:8]
    cfg["run_name"] = name
    cfg["run_id"] = run_id
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.get("seed", 42))
    yaml.dump(cfg, open(f"checkpoints/{name}.yaml", "w"))
    print(f"Run ID: {run_id}, Run name: {name}")

    dm = getattr(dataset, cfg["dataset"])(**cfg)
    model = getattr(pl_model, cfg["model"])(dm=dm, **cfg)

    logger = TensorBoardLogger(
        save_dir="logs",
        name=name,
        version=run_id,
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
    trainer.fit(model, dm)
    model = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, dm=dm, **cfg)

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
        results_path = os.path.join(os.getcwd(), "result.csv")
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            df = pd.concat([df, row], ignore_index=True)
        else:
            df = row
        df.to_csv(results_path, index=False)
    return model, dm, cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=0)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        c = yaml.safe_load(f)

    if args.task_id is not None:
        if args.task_id < len(c):
            build(c[args.task_id])
    else:
        for i, config in enumerate(c):
            build(config)
