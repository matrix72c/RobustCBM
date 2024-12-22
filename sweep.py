import torch
import wandb
from main import MyLightningCLI, exp
from utils import get_args

def sweep_exp():
    wandb.run.tags = [cli.model.__class__.__name__, cli.datamodule.__class__.__name__]
    for k, v in wandb.config.items():
        if k == "patience":
            cli.config["patience"] = v
            continue
        cli.config["model"]["init_args"][k] = v
        if k == "num_concepts":
            cli.config["data"]["init_args"]["num_concepts"] = v
    cli.instantiate_classes()
    args = get_args(cli.config.as_dict())
    args["model"] = cli.model.__class__.__name__
    args["dataset"] = cli.datamodule.__class__.__name__
    wandb.config.update(args)
    exp(cli.model, cli.datamodule, args)

torch.set_float32_matmul_precision("high")
cli = MyLightningCLI(save_config_callback=None, run=False)

sweep_id = cli.config["sweep_id"]
wandb.agent(sweep_id, function=sweep_exp, entity="matrix72c-jesse", project="RobustCBM")