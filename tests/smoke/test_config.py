"""Smoke tests for config loading."""
import pytest


def test_config_yaml_loads(config_data):
    """Verify config.yaml can be loaded with yaml.safe_load."""
    assert config_data is not None
    assert isinstance(config_data, dict)


def test_config_has_required_keys(config_data):
    """Verify config has 'model' or 'models' key."""
    has_model_key = "model" in config_data or "models" in config_data
    assert has_model_key, "Config must have 'model' or 'models' key"
