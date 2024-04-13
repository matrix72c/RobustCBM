import yaml
from itertools import product

yaml_content = """
dataset: ["CUB", "AWA"]
model: ["CBM", "PBM"]
mode: ["Joint", "Sequential"]
use_pretrained: [False, True, "pretrained_models/resnet_train_acc_0.85_test_acc_0.74.pth"]
use_adv: ["image2label", "concept2label", "image2concept", "image2concept&concept2label"]
"""

data = yaml.safe_load(yaml_content)

keys = data.keys()

lists = [value for value in data.values()]

combinations = list(product(*lists))

dict_combinations = []
for combo in combinations:
    combo_dict = {key: value for key, value in zip(keys, combo)}
    dict_combinations.append(combo_dict)

for combo in dict_combinations:
    print(combo)
