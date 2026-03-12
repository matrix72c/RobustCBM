"""Smoke tests for model imports."""
import pytest


def test_cbm_imports():
    """Verify CBM can be imported from model."""
    from model import CBM
    assert CBM is not None


def test_vcbm_imports():
    """Verify VCBM can be imported."""
    from model import VCBM
    assert VCBM is not None


def test_cem_imports():
    """Verify CEM can be imported."""
    from model import CEM
    assert CEM is not None


def test_backbone_imports():
    """Verify backbone can be imported."""
    from model import backbone
    assert backbone is not None


def test_dataset_imports():
    """Verify dataset classes can be imported."""
    from dataset import CUB, AwA, celeba
    assert CUB is not None
    assert AwA is not None
    assert celeba is not None
