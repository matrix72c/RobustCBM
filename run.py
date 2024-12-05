import yaml
import copy
import os

cfg_path = "config/exp1/CUB/CBM.yaml"
concept_weights = [10, 1, 0.1, 0.01]
concept_weights_names = ["1e1", "1e0", "1e-1", "1e-2"]

with open(cfg_path, 'r') as file:
    config = yaml.safe_load(file)

for concept_weight, concept_weights_name in zip(concept_weights, concept_weights_names):
    new_config = copy.deepcopy(config)
    
    new_config['model']['init_args']['concept_weight'] = concept_weight
    new_config['run_name'] = new_config['run_name'] + "_" + concept_weights_name
    
    new_config_filename = cfg_path.replace(".yaml", f"_{concept_weights_name}.yaml")
    with open(new_config_filename, 'w') as new_file:
        yaml.dump(new_config, new_file)

    command = f"python main.py --config {new_config_filename}"
    print(f"Running: {command}")
    os.system(command)

os.system("/usr/bin/shutdown")