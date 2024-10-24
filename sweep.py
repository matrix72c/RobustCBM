from lightning.pytorch.trainer import Trainer
import torch
import wandb
from main import MyLightningCLI, train


def sweep_train():
    wandb.init(project="SweepCBM")

    for k, v in wandb.config.items():
        cli.config["model"]["init_args"][k] = v
    cli.config["run_name"] = "Sweep_" + cli.config["run_name"]
    cli.instantiate_classes()
    train(cli.model, cli.datamodule, cli)


torch.set_float32_matmul_precision("high")
cli = MyLightningCLI(save_config_callback=None, run=False)
sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "optimizer": {"values": ["Adam", "SGD"]},
        "lr": {
            "values": [1e-1, 1e-2, 1e-3],
        },
        "step_size": {"values": [10, 20, 30]},
        "vib_lambda": {"values": [1, 0.1, 0.01]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="SweepCBM")
wandb.agent(sweep_id, sweep_train)
