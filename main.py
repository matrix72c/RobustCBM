import os
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
import pandas as pd
import torch
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--run_name", default="run")
        parser.add_argument("--adv_hparams", default=None)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli = MyLightningCLI(save_config_callback=None, run=False)

    ckpt_path = "saved_models/" + cli.config.run_name + ".ckpt"

    if not os.path.exists(ckpt_path):
        # Std training
        logger = AimLogger(run_name=cli.config.run_name)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            dirpath="saved_models/",
            filename=cli.config.run_name,
            save_top_k=1,
            mode="max",
            save_weights_only=True,
            enable_version_counter=False,
        )
        early_stopping = EarlyStopping(
            monitor="val_acc", patience=10, mode="max", min_delta=0.001
        )
        callbacks = [checkpoint_callback, early_stopping]
        trainer = Trainer(
            max_steps=1000,
            log_every_n_steps=1,
            logger=logger,
            callbacks=callbacks,
        )
        trainer.fit(cli.model, cli.datamodule)

    # Construct the model
    model = cli.model.__class__.load_from_checkpoint(
        ckpt_path, **cli.config.adv_hparams
    )
    dm = cli.datamodule
    logger = AimLogger(run_name="Adv_" + cli.config.run_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
        filename="Adv_" + cli.config.run_name,
        save_top_k=1,
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_acc", patience=20, mode="max", min_delta=0.001
    )
    callbacks = [checkpoint_callback, early_stopping]
    trainer = Trainer(
        max_steps=1000,
        log_every_n_steps=1,
        logger=logger,
        callbacks=callbacks,
    )

    # Evaluate standard model
    model.adv_training = False
    ret = trainer.test(model, dm)
    std_acc, std_concept_acc = (
        ret[0]["test_acc"],
        ret[0]["test_concept_acc"],
    )

    # Adversarial training
    model.adv_training = True
    trainer.fit(model, dm)

    # Evaluate robust model
    model.adv_training = False
    ret = trainer.test(model, dm, ckpt_path="best")
    adv_acc, adv_concept_acc = (
        ret[0]["test_acc"],
        ret[0]["test_concept_acc"],
    )

    # Adversarial attacks
    model.adv_training = True

    model.eval_atk = "PGD"
    ret = trainer.test(model, dm, ckpt_path="best")
    adv_pgd_acc, adv_pgd_concept_acc = (
        ret[0]["test_acc"],
        ret[0]["test_concept_acc"],
    )

    model.eval_atk = "AA"
    ret = trainer.test(model, dm, ckpt_path="best")
    adv_aa_acc, adv_aa_concept_acc = ret[0]["test_acc"], ret[0]["test_concept_acc"]

    df = pd.read_csv("result.csv")
    new_row = {
        "run_name": cli.config.run_name,
        "std_acc": std_acc,
        "std_concept_acc": std_concept_acc,
        "adv_acc": adv_acc,
        "adv_concept_acc": adv_concept_acc,
        "adv_pgd_acc": adv_pgd_acc,
        "adv_pgd_concept_acc": adv_pgd_concept_acc,
        "adv_aa_acc": adv_aa_acc,
        "adv_aa_concept_acc": adv_aa_concept_acc,
    }
    new_row = new_row | model.hparams
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("result.csv", index=False)
