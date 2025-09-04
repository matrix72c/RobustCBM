import os
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch import seed_everything
import torch
import yaml, argparse
import model as pl_model
import dataset
from utils import build_name, yaml_merge
import hashlib


def test_model(config):
    """
    Test function that loads a model from checkpoint and performs testing.
    Similar to build() function but focused only on testing existing checkpoints.
    """
    if config.get("ckpt", None) is None:
        raise ValueError("ckpt must be provided for testing")
    
    ckpt_path = "checkpoints/" + config["ckpt"] + ".ckpt"
    cfg_path = "checkpoints/" + config["ckpt"] + ".yaml"
    
    # Check if checkpoint files exist
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    print(f"Loading checkpoint: {config['ckpt']}")
    
    # Load config from checkpoint
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Merge any additional config parameters
    cfg = yaml_merge(cfg, config)
    
    # Set up environment
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.get("seed", 42))
    
    # Load dataset and model
    dm = getattr(dataset, cfg["dataset"])(**cfg)
    model = getattr(pl_model, cfg["model"]).load_from_checkpoint(ckpt_path, dm=dm, **cfg)
    
    # Create trainer for testing
    trainer = Trainer(
        logger=False,  # No logging needed for testing
        enable_progress_bar=True,
        inference_mode=False,
    )
    
    print(f"Testing model: {cfg.get('model', 'Unknown')} on dataset: {cfg.get('dataset', 'Unknown')}")
    
    # Run test
    test_results = trainer.test(model, dm)
    
    print(f"Test completed for {config['ckpt']}")
    print("Test Results:")
    for result in test_results:
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    return model, dm, cfg, test_results


def batch_test(configs):
    """
    Run tests on multiple configurations sequentially.
    """
    all_results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Testing configuration {i+1}/{len(configs)}")
        print(f"{'='*60}")
        
        try:
            model, dm, cfg, test_results = test_model(config)
            all_results.append({
                'config_index': i,
                'ckpt': config.get('ckpt', 'Unknown'),
                'model': cfg.get('model', 'Unknown'),
                'dataset': cfg.get('dataset', 'Unknown'),
                'test_results': test_results,
                'status': 'success'
            })
        except Exception as e:
            print(f"Error testing configuration {i}: {str(e)}")
            all_results.append({
                'config_index': i,
                'ckpt': config.get('ckpt', 'Unknown'),
                'error': str(e),
                'status': 'failed'
            })
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = [r for r in all_results if r['status'] == 'success']
    failed_tests = [r for r in all_results if r['status'] == 'failed']
    
    print(f"Total configurations: {len(configs)}")
    print(f"Successful tests: {len(successful_tests)}")
    print(f"Failed tests: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for result in failed_tests:
            print(f"  - Config {result['config_index']}: {result['ckpt']} - {result['error']}")
    
    if successful_tests:
        print("\nSuccessful tests:")
        for result in successful_tests:
            print(f"  - Config {result['config_index']}: {result['ckpt']} ({result['model']} on {result['dataset']})")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained models from checkpoints")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to YAML config file containing test configurations")
    parser.add_argument("--task_id", type=int, default=None,
                       help="Specific task ID to test (if not provided, all tasks will be tested)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)
    
    # Handle single config vs list of configs
    if not isinstance(configs, list):
        configs = [configs]
    
    # Filter by task_id if specified
    if args.task_id is not None:
        if args.task_id < len(configs):
            print(f"Testing only task {args.task_id}")
            test_model(configs[args.task_id])
        else:
            print(f"Error: task_id {args.task_id} is out of range (max: {len(configs)-1})")
    else:
        # Test all configurations
        print(f"Testing {len(configs)} configurations")
        batch_test(configs)
