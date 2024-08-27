import os
import sys
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
import pandas as pd
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import wandb


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--run_name", default="run")
        parser.add_argument("--adv_hparams", default=None)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli = MyLightningCLI(save_config_callback=None, run=False)

    ckpt_path = "checkpoints/" + cli.config.run_name + ".ckpt"

    if not os.path.exists(ckpt_path):
        # Std training
        logger = WandbLogger(name=cli.config.run_name, project="RobustCBM")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            dirpath="checkpoints/",
            filename=cli.config.run_name,
            save_top_k=1,
            mode="max",
            enable_version_counter=False,
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=15, mode="min")
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks = [checkpoint_callback, early_stopping, lr_monitor]
        trainer = Trainer(
            log_every_n_steps=10,
            logger=logger,
            callbacks=callbacks,
            max_epochs=-1,
            precision="16-mixed",
        )
        trainer.fit(cli.model, cli.datamodule)
        wandb.finish()

    # Construct the model
    model = cli.model.__class__.load_from_checkpoint(
        ckpt_path, **cli.config.adv_hparams if cli.config.adv_hparams else {}
    )
    dm = cli.datamodule
    logger = WandbLogger(name="Adv_" + cli.config.run_name, project="RobustCBM")
    checkpoint_callback = ModelCheckpoint(
        monitor="adv_val_acc",
        dirpath="checkpoints/",
        filename="Adv_" + cli.config.run_name,
        save_top_k=1,
        mode="max",
        enable_version_counter=False,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=200, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, early_stopping, lr_monitor]
    trainer = Trainer(
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        max_epochs=-1,
        precision="16-mixed",
    )

    # Evaluate standard model
    model.adv_mode = False
    ret = trainer.test(model, dm)
    std_acc, std_concept_acc = (
        ret[0]["test_acc"],
        ret[0]["test_concept_acc"],
    )

    # Adversarial training
    model.adv_mode = True
    trainer.fit(model, dm)

    # Evaluate robust model
    ret = trainer.test(model, dm, ckpt_path="best")
    adv_acc, adv_concept_acc = (
        ret[0]["test_acc"],
        ret[0]["test_concept_acc"],
    )
    adv_pgd_acc, adv_pgd_concept_acc = (
        ret[0]["adv_test_acc"],
        ret[0]["adv_test_concept_acc"],
    )

    model.eval_atk = model.auto_atk
    ret = trainer.test(model, dm, ckpt_path="best")
    adv_aa_acc, adv_aa_concept_acc = (
        ret[0]["adv_test_acc"],
        ret[0]["adv_test_concept_acc"],
    )

    new_row = pd.DataFrame([{
        "run_name": cli.config.run_name,
        "std_acc": std_acc,
        "std_concept_acc": std_concept_acc,
        "adv_acc": adv_acc,
        "adv_concept_acc": adv_concept_acc,
        "adv_pgd_acc": adv_pgd_acc,
        "adv_pgd_concept_acc": adv_pgd_concept_acc,
        "adv_aa_acc": adv_aa_acc,
        "adv_aa_concept_acc": adv_aa_concept_acc,
    }])
    tb = wandb.Table(dataframe=new_row)
    wandb.log({"Results": tb})

    df = pd.read_csv("result.csv")
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("result.csv", index=False)
