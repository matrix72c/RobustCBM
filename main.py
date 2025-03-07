import os
import tempfile
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch import seed_everything
import torch
import wandb
import yaml, argparse
from utils import get_oss
import model as pl_model
import dataset
from attacks import *


def exp(config):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg.update(config)
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg["seed"])
    dm = getattr(dataset, cfg["dataset"])(**cfg)
    model = getattr(pl_model, cfg["model"])(dm=dm, **cfg)
    if config.get("experiment_name", None) is not None:
        name = config["experiment_name"]
    else:
        d = sorted(config.items(), key=lambda x: x[0])
        # pop dict
        for k, v in d:
            if isinstance(v, dict):
                d.remove((k, v))
                d.extend(v.items())
        name = "_".join([f"{v}" if isinstance(v, str) else f"{k}-{v}" for k, v in d])
        name = name.lower()

    logger = WandbLogger(
        name=name,
        project="CBM",
        config=cfg,
        tags=[
            cfg["model"],
            cfg["dataset"],
            cfg["adv_mode"],
            cfg["base"],
        ],
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
        accelerator="gpu",
        devices=cfg["gpus"],
        strategy="ddp_find_unused_parameters_true",
        log_every_n_steps=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg["epochs"],
        inference_mode=False,
    )
    bucket = get_oss()
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            if os.path.exists(ckpt_path):
                fp = ckpt_path
            elif bucket.object_exists(ckpt_path):
                fp = os.path.join(tmpdir, os.path.basename(ckpt_path))
                bucket.get_object_to_file(ckpt_path, fp)
                print(f"Download {ckpt_path} to {fp}")
            else:
                raise ValueError(f"Checkpoint {ckpt_path} not found")
            model = model.__class__.load_from_checkpoint(fp, dm=dm, **cfg)
            print("Load from checkpoint: ", ckpt_path)
    else:
        trainer.fit(model, dm)
        ckpt_path = "checkpoints/" + name + ".ckpt"
        best = trainer.checkpoint_callback.best_model_path
        bucket.put_object_from_file(ckpt_path, best)
        print(f"Upload {best} to {ckpt_path}")
        model = model.__class__.load_from_checkpoint(best, dm=dm, **cfg)

    trainer = Trainer(
        devices=1,
        num_nodes=1,
        accelerator="gpu",
        logger=logger,
        inference_mode=False,
    )
    model.eval_stage = "robust"
    if model.adv_mode == "std":
        eps = [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
    else:
        eps = list(range(5))
    for i in eps:
        if i > 0:
            model.eval_atk = PGD(eps=i / 255)
            model.adv_mode = "adv"
        else:
            model.adv_mode = "std"
        trainer.test(model, datamodule=dm)

    model.eval_stage = "intervene"
    if cfg["model"] != "backbone":
        for eps in [0, 4]:
            for i in range(cfg["max_intervene_budget"] + 1):
                model.intervene_budget = i
                if eps > 0:
                    model.eval_atk = PGD(eps=eps / 255)
                    model.adv_mode = "adv"
                else:
                    model.adv_mode = "std"
                trainer.test(model, datamodule=dm)

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
