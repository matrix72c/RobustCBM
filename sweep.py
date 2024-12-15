import torch
import wandb
import yaml
from main import MyLightningCLI, exp
from utils import get_args

def sweep_exp():
    wandb.init(project="SweepCBM")
    for k, v in wandb.config.items():
        cli.config["model"]["init_args"][k] = v
        if k == "num_concepts":
            cli.config["data"]["init_args"]["num_concepts"] = v
    cli.instantiate_classes()
    args = get_args(cli.config.as_dict())
    wandb.config.update(args)
    exp(cli.model, cli.datamodule, args)

torch.set_float32_matmul_precision("high")
cli = MyLightningCLI(save_config_callback=None, run=False)

if cli.config["sweep_id"] is not None:
    sweep_id = cli.config["sweep_id"]
else:
    model = cli.model.__class__.__name__
    dm = cli.datamodule.__class__.__name__
    with open(f"config/sweep/{dm}/{model}.yaml") as f:
        sweep_cfg = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_cfg, project="SweepCBM")

wandb.agent(sweep_id, function=sweep_exp, entity="matrix72c-jesse", project="SweepCBM")