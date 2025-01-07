import argparse, yaml
import wandb
from main import exp


def sweep_exp():
    wandb.init(project="RobustCBM")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    for k, v in wandb.config.items():
        config[k] = v
    exp(config)


parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", type=str, required=True)
args = parser.parse_args()
wandb.agent(
    args.sweep_id, function=sweep_exp, entity="matrix72c-jesse", project="RobustCBM"
)
