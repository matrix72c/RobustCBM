import argparse, yaml
import wandb
from main import build


def sweep_exp():
    wandb.init(project="CBM")
    c = wandb.config.as_dict()
    build(c)


parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", type=str, required=True)
args = parser.parse_args()
wandb.agent(
    args.sweep_id, function=sweep_exp, entity="matrix72c-jesse", project="CBM"
)
