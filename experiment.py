from itertools import product
import yaml

# Assuming the train function is defined in train.py and is importable
from train import train

yaml_content = """
dataset: ["CUB", "AwA"]
model: ["CBM", "PBM"]
mode: ["Joint", "Sequential"]
use_pretrained: [False, True, "pretrained_models/resnet_train_acc_0.85_test_acc_0.74.pth"]
use_adv_joint: ["", "image2label"]
use_adv_sequential: ["", "concept2label", "image2concept", "image2concept&concept2label"]
"""

data = yaml.safe_load(yaml_content)

# Load the template configuration
with open("train.yaml", "r", encoding="utf-8") as f:
    template = yaml.safe_load(f)

def create_combinations(mode):
    # Use a condition to choose the correct 'use_adv' list based on the 'mode'
    use_adv_options = data['use_adv_joint'] if mode == 'Joint' else data['use_adv_sequential']
    
    # Prepare keys for product, excluding 'mode' and 'use_adv' keys
    product_keys = [key for key in data if key not in ['mode', 'use_adv_joint', 'use_adv_sequential']]
    
    # Prepare values for product, excluding 'use_adv' options
    product_values = [data[key] for key in product_keys]
    
    # Generate the combinations excluding 'use_adv'
    base_combinations = list(product(*product_values))
    
    # Generate the full combinations including 'use_adv'
    combinations = [
        {**dict(zip(product_keys, base_combo)), 'mode': mode, 'use_adv': use_adv}
        for base_combo in base_combinations for use_adv in use_adv_options
    ]
    
    return combinations

# Train models for each mode with the respective 'use_adv' settings
for mode in data['mode']:
    combinations = create_combinations(mode)
    for combo in combinations:
        conf = template.copy()
        conf.update(combo)
        
        try:
            train(conf)
        except Exception as e:
            print(f"Training failed for configuration {combo} with error: {e}")
