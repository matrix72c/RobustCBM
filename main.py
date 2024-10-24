import os
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import wandb
from attacks import PGD


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--run_name", default="run")
        parser.add_argument("--patience", default=30)


def train(model, dm, cli):
    logger = WandbLogger(name=cli.config.run_name, project="RobustCBM")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
        filename=cli.config.run_name,
        save_top_k=1,
        mode="max",
        enable_version_counter=False,
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=cli.config.patience, mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, early_stopping, lr_monitor]
    trainer = Trainer(
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        max_epochs=-1,
        precision="16-mixed",
    )

    # Adversarial training
    model.adv_mode = True
    trainer.fit(model, dm)


def eval(model):
    model.adv_mode = False
    trainer = Trainer(precision="16-mixed")

    wandb.init(project="RobustCBM", name="Eval_" + cli.config.run_name)
    for j in [1, 2, 4, 10]:
        for i in range(11):
            if i > 0:
                model.eval_atk = PGD(model, steps=10, eps=i / 255.0)
                model.adv_mode = True
            else:
                model.adv_mode = False
            acc = trainer.test(model, datamodule=dm)[0]["acc"]
            wandb.log({"PGD" + str(j): acc, "eps": i})
    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli = MyLightningCLI(save_config_callback=None, run=False)
    model = cli.model
    dm = cli.datamodule
    ckpt_path = "checkpoints/" + cli.config.run_name + ".ckpt"
    if os.path.exists(ckpt_path):
        print("Loading checkpoint:", ckpt_path)
    else:
        train(model, dm, cli)
        wandb.finish()

    # Evaluate robust model
    model = model.__class__.load_from_checkpoint(ckpt_path)
    eval(model)
