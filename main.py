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
from attacks import AutoAttack


def train(config):
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
        name = "_".join(["{}".format(t[1]) for t in d])
        name = name.lower()
    wandb.run.name = name
    wandb.run.tags = [
        cfg["model"],
        cfg["dataset"],
        cfg["adv_mode"],
        cfg["base"],
    ]
    wandb.config.update(cfg)

    logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(
        monitor="acc",
        dirpath="checkpoints/",
        filename=wandb.run.name,
        save_top_k=1,
        mode="max",
        enable_version_counter=False,
        save_weights_only=True,
        every_n_epochs=20,
    )
    early_stopping = EarlyStopping(
        monitor="lr", mode="min", patience=1000, stopping_threshold=1e-4
    )
    callbacks = [checkpoint_callback, early_stopping]
    trainer = Trainer(
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg["epochs"],
        inference_mode=False,
    )
    bucket = get_oss()
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None and bucket.object_exists(ckpt_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, os.path.basename(ckpt_path))
            print(f"Downloading {ckpt_path} to {fp}")
            bucket.get_object_to_file(ckpt_path, fp)
            model = model.__class__.load_from_checkpoint(fp, dm=dm, **cfg)
        print("Load from checkpoint: ", ckpt_path)
    else:
        trainer.fit(model, dm)
        ckpt_path = "checkpoints/" + wandb.run.name + ".ckpt"
        bucket.put_object_from_file(ckpt_path, ckpt_path)
        model = model.__class__.load_from_checkpoint(ckpt_path, dm=dm, **cfg)
    return trainer, model, dm


def exp(config):
    trainer, model, dm = train(config)
    if model.adv_mode == "std":
        eps = [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
    else:
        eps = list(range(5))
    accs, acc5s, acc10s, asrs, asr5s, asr10s = [], [], [], [], [], []
    for i in eps:
        if i > 0:
            model.eval_atk = AutoAttack(eps=i / 255)
            model.adv_mode = "adv"
        else:
            model.adv_mode = "std"
        ret = trainer.test(model, datamodule=dm)[0]
        acc, acc5, acc10 = ret["acc"], ret["acc5"], ret["acc10"]
        accs.append(acc), acc5s.append(acc5), acc10s.append(acc10)
        if i == 0:
            ca, ca5, ca10 = acc, acc5, acc10
            asr, asr5, asr10 = 0, 0, 0
        else:
            asr = (ca - acc) / ca
            asr5 = (ca5 - acc5) / ca5
            asr10 = (ca10 - acc10) / ca10
        asrs.append(asr), asr5s.append(asr5), asr10s.append(asr10)

    wandb.run.summary["eps"] = eps
    wandb.run.summary["Acc@1"] = accs
    wandb.run.summary["Acc@5"] = acc5s
    wandb.run.summary["Acc@10"] = acc10s
    wandb.run.summary["ASR@1"] = asrs
    wandb.run.summary["ASR@5"] = asr5s
    wandb.run.summary["ASR@10"] = asr10s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        c = yaml.safe_load(f)
    wandb.init(project="RobustCBM")
    exp(c)
