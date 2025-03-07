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
import math
from utils import get_oss
import model as pl_model
import dataset
from attacks import *


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
                d.extend(v.items())
        name = "_".join([f"{v}" if isinstance(v, str) else f"{k}-{v}" for k, v in d])
        name = name.lower()

    config = cfg
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
        strategy="ddp",
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
        ckpt_path = "checkpoints/" + wandb.run.name + ".ckpt"
        best = trainer.checkpoint_callback.best_model_path
        bucket.put_object_from_file(ckpt_path, best)
        print(f"Upload {best} to {ckpt_path}")
        model = model.__class__.load_from_checkpoint(best, dm=dm, **cfg)
    return trainer, model, dm


def exp(config):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    trainer, model, dm = train(config)
    if model.adv_mode == "std":
        eps = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    else:
        eps = list(range(5))
    accs, acc5s, acc10s, asrs, asr5s, asr10s = [], [], [], [], [], []
    concept_accs, concept_asrs = [], []
    for i in eps:
        if i > 0:
            model.eval_atk = PGD(eps=i / 255)
            model.adv_mode = "adv"
        else:
            model.adv_mode = "std"
        ret = trainer.test(model, datamodule=dm)[0]
        acc, acc5, acc10 = ret["acc"], ret["acc5"], ret["acc10"]
        accs.append(acc), acc5s.append(acc5), acc10s.append(acc10)
        concept_acc = ret["concept_acc"]
        if i == 0:
            ca, ca5, ca10 = acc, acc5, acc10
            clean_concept_acc = concept_acc
            asr, asr5, asr10 = 0, 0, 0
            concept_asr = 0
        else:
            asr = (ca - acc) / ca
            asr5 = (ca5 - acc5) / ca5
            asr10 = (ca10 - acc10) / ca10
            concept_asr = (clean_concept_acc - concept_acc) / clean_concept_acc
        asrs.append(asr), asr5s.append(asr5), asr10s.append(asr10)
        concept_accs.append(concept_acc), concept_asrs.append(concept_asr)
        wandb.log(
            {
                "Acc@1": acc,
                "Acc@5": acc5,
                "Acc@10": acc10,
                "Concept Acc@1": concept_acc,
                "ASR@1": asr,
                "ASR@5": asr5,
                "ASR@10": asr10,
                "Concept ASR@1": concept_asr,
                "eps": i if (i >= 1 or i == 0) else int(math.log10(i) - math.log10(min(x for x in eps if x > 0)) + 1),
            },
        )

    wandb.run.summary["eps"] = eps
    wandb.run.summary["Acc@1"] = accs
    wandb.run.summary["Acc@5"] = acc5s
    wandb.run.summary["Acc@10"] = acc10s
    wandb.run.summary["ASR@1"] = asrs
    wandb.run.summary["ASR@5"] = asr5s
    wandb.run.summary["ASR@10"] = asr10s
    wandb.run.summary["Concept Acc@1"] = concept_accs
    wandb.run.summary["Concept ASR@1"] = concept_asrs

    if cfg["model"] != "backbone":
        for eps in [0, 4]:
            for i in range(cfg["max_intervene_budget"] + 1):
                model.intervene_budget = i
                if eps > 0:
                    model.eval_atk = PGD(eps=eps / 255)
                    model.adv_mode = "adv"
                else:
                    model.adv_mode = "std"
                ret = trainer.test(model, datamodule=dm)[0]
                acc = ret["acc"]
                concept_acc = ret["concept_acc"]
                if eps == 0 and i == 0:
                    ca = acc
                    clean_concept_acc = concept_acc

                if eps == 0:
                    wandb.log(
                        {
                            "Clean Acc under Intervene": acc,
                            "Clean Concept Acc under Intervene": concept_acc,
                            "Intervene Budget": i,
                        },
                    )
                else:
                    wandb.log(
                        {
                            "Robust Acc under Intervene": acc,
                            "Robust Concept Acc under Intervene": concept_acc,
                            "ASR under Intervene": (ca - acc) / ca,
                            "Concept ASR under Intervene": (clean_concept_acc - concept_acc)
                            / clean_concept_acc,
                            "Intervene Budget": i,
                        },
                    )
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
