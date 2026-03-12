"""Shared pytest fixtures for RobustCBM tests."""
import os
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def config_path():
    """Return path to config.yaml if it exists."""
    config_file = Path(__file__).parent.parent / "config.yaml"
    if not config_file.exists():
        pytest.skip("config.yaml not found")
    return config_file


@pytest.fixture
def config_data(config_path):
    """Load and return config.yaml contents."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def minimal_model_config():
    """Provide a minimal model config dict for testing."""
    return {
        "model": {
            "name": "CBM",
            "backbone": "resnet18",
            "num_classes": 10,
        }
    }
