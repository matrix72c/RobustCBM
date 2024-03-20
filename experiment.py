import yaml
from train import train

f = open("experiment.yaml", "r", encoding="utf-8")
exps = yaml.load(f.read(), Loader=yaml.FullLoader)
f = open("train.yaml", "r", encoding="utf-8")
conf = yaml.load(f.read(), Loader=yaml.FullLoader)
for exp in exps:
    conf["trainer"] = exp["trainer"]
    if "use_adv" in exp:
        for use_adv in exp["use_adv"]:
            conf["use_adv"] = use_adv
            train(conf)
    if "use_noise" in exp:
        for use_noise in exp["use_noise"]:
            conf["use_noise"] = use_noise
            train(conf)