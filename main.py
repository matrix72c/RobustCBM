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
import sys


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
        strategy=(
            "ddp_find_unused_parameters_true" if cfg["model"] == "backbone" else "ddp"
        ),
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
        if trainer.is_global_zero:
            ckpt_path = "checkpoints/" + name + ".ckpt"
            best = trainer.checkpoint_callback.best_model_path
            model = model.__class__.load_from_checkpoint(best, dm=dm, **cfg)
            try:
                bucket.put_object_from_file(ckpt_path, best)
                print(f"Upload {best} to {ckpt_path}")
            except Exception as e:
                print(f"Upload {best} to {ckpt_path} failed: {e}")
                wandb.run.alert(
                    f"Upload {best} to {ckpt_path} failed: {e}",
                    alert_level="error",
                )

        torch.distributed.destroy_process_group()
        if not trainer.is_global_zero:
            sys.exit(0)

        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("NODE_RANK", None)
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)

    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=logger,
        inference_mode=False,
    )
    dm = getattr(dataset, cfg["dataset"])(**cfg)
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
