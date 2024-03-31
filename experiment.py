import yaml
from train import train

f = open("experiment.yaml", "r", encoding="utf-8")
exps = yaml.load(f.read(), Loader=yaml.FullLoader)
f = open("train.yaml", "r", encoding="utf-8")
template = yaml.load(f.read(), Loader=yaml.FullLoader)
for exp in exps:
    conf = template.copy()
    for k, v in exp.items():
        conf[k] = v
    if len(conf["use_adv"]) > 0:
        conf["batch_size"] /= 2
    train(conf)