import yaml
import copy
import os

cfg_path = "config/exp1/CUB/nonCBM.yaml"
num_concepts = [32, 64, 112, 256, 512]
num_concepts_names = ["32", "64", "align", "256", "512"]

with open(cfg_path, 'r') as file:
    config = yaml.safe_load(file)

for num_concept, num_concepts_name in zip(num_concepts, num_concepts_names):
    new_config = copy.deepcopy(config)
    
    new_config['model']['init_args']['num_concepts'] = num_concept
    new_config['data']['init_args']['num_concepts'] = num_concept
    new_config['run_name'] = new_config['run_name'] + "_" + num_concepts_name
    
    new_config_filename = cfg_path.replace(".yaml", f"_{num_concepts_name}.yaml")
    with open(new_config_filename, 'w') as new_file:
        yaml.dump(new_config, new_file)

    command = f"python main.py --config {new_config_filename}"
    print(f"Running: {command}")
    os.system(command)

os.system("/usr/bin/shutdown")