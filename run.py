import yaml
import copy
import os

concept_weights = [10, 1, 0.1, 0.01]
run_names = ["1e1", "1e0", "1e-1", "1e-2"]

with open('config/exp1/CUB/CBM.yaml', 'r') as file:
    config = yaml.safe_load(file)

for concept_weight, run_name in zip(concept_weights, run_names):
    new_config = copy.deepcopy(config)
    
    new_config['model']['init_args']['concept_weight'] = concept_weight
    new_config['run_name'] = f"exp1_CUB_CBM_{run_name}"
    
    new_config_filename = f"config/exp1/CUB/{new_config['run_name']}.yaml"
    with open(new_config_filename, 'w') as new_file:
        yaml.dump(new_config, new_file)

    command = f"python main.py --config {new_config_filename}"
    print(f"Running: {command}")
    os.system(command)

with open('config/exp1/CUB/CEM.yaml', 'r') as file:
    config = yaml.safe_load(file)

for concept_weight, run_name in zip(concept_weights, run_names):
    new_config = copy.deepcopy(config)
    
    new_config['model']['init_args']['concept_weight'] = concept_weight
    new_config['run_name'] = f"exp1_CUB_CEM_{run_name}"
    
    new_config_filename = f"config/exp1/CUB/{new_config['run_name']}.yaml"
    with open(new_config_filename, 'w') as new_file:
        yaml.dump(new_config, new_file)

    command = f"python main.py --config {new_config_filename}"
    print(f"Running: {command}")
    os.system(command)

os.system("/usr/bin/shutdown")