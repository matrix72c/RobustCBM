from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from lightning.pytorch.loggers import WandbLogger


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--run_name", default="run")
        parser.add_argument("--adv_hparams", default=None)


torch.set_float32_matmul_precision("high")
cli = MyLightningCLI(save_config_callback=None, run=False)


def train():
    wandb.init(project="CBM-sweeps")
    logger = WandbLogger(project="CBM-sweeps")
    early_stopping = EarlyStopping(monitor="val_loss", patience=15, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
        filename=cli.config.run_name,
        save_top_k=1,
        mode="max",
    )
    trainer = Trainer(
        log_every_n_steps=10,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        max_epochs=-1,
        precision="16-mixed",
    )

    for k, v in wandb.config.items():
        cli.config["model"]["init_args"][k] = v

    cli.instantiate_classes()
    model = cli.model
    dm = cli.datamodule
    trainer.fit(model, dm)

sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "base": {"values": ["resnet50", "inceptionv3"]},
        "use_pretrained": {"values": [True, False]},
        "concept_weight": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 10,
        },
        "optimizer": {"values": ["Adam", "SGD"]},
        "lr": {
            "values": [1e-1, 1e-2, 1e-3],
        },
        "classifier": {"values": ["FC", "MLP"]},
        "scheduler_patience": {"values": [3, 5, 10]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="CBM-sweeps")
wandb.agent(sweep_id, train)
