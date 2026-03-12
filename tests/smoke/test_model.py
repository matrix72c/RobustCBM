"""Smoke tests for model imports and functionality."""
import pytest
import torch
from unittest.mock import MagicMock


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


def test_model_instantiation():
    """Verify CBM model can be instantiated with mock DataModule."""
    from model import CBM

    # Create a mock DataModule with required attributes
    mock_dm = MagicMock()
    mock_dm.num_classes = 10
    mock_dm.num_concepts = 20
    mock_dm.concept_names = [f"concept_{i}" for i in range(20)]

    # Minimal config for model instantiation
    cfg = {
        "model": "CBM",
        "base": "resnet50",
        "use_pretrained": True,
        "concept_weight": 1,
        "optimizer": "SGD",
        "optimizer_args": {"lr": 0.1, "momentum": 0.9},
        "cbm_mode": "hybrid",
    }

    # Instantiate model
    model = CBM(dm=mock_dm, **cfg)
    assert model is not None
    assert hasattr(model, "forward")
    assert hasattr(model, "training_step")


def test_model_forward_pass():
    """Verify forward pass runs with dummy input."""
    from model import CBM

    # Create mock DataModule
    mock_dm = MagicMock()
    mock_dm.num_classes = 10
    mock_dm.num_concepts = 20
    mock_dm.concept_names = [f"concept_{i}" for i in range(20)]

    # Minimal config
    cfg = {
        "model": "CBM",
        "base": "resnet50",
        "use_pretrained": False,  # Faster test without pretrained
        "concept_weight": 1,
        "optimizer": "SGD",
        "optimizer_args": {"lr": 0.1},
        "cbm_mode": "hybrid",
    }

    model = CBM(dm=mock_dm, **cfg)
    model.eval()  # Set to eval mode for inference

    # Create dummy input (batch_size=2, channels=3, height=224, width=224)
    dummy_input = torch.randn(2, 3, 224, 224)

    # Run forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify output structure
    assert output is not None
    if isinstance(output, tuple):
        # CBM returns (class_logits, concept_probs)
        class_logits, concept_probs = output
        assert class_logits.shape[0] == 2, "Batch size should match"
        assert class_logits.shape[1] == 10, "Num classes should match"
        assert concept_probs.shape[0] == 2, "Batch size should match"
        assert concept_probs.shape[1] == 20, "Num concepts should match"


def test_model_calc_loss():
    """Verify loss computation works."""
    from model import CBM

    # Create mock DataModule
    mock_dm = MagicMock()
    mock_dm.num_classes = 10
    mock_dm.num_concepts = 20
    mock_dm.concept_names = [f"concept_{i}" for i in range(20)]

    cfg = {
        "model": "CBM",
        "base": "resnet50",
        "use_pretrained": False,
        "concept_weight": 1,
        "optimizer": "SGD",
        "optimizer_args": {"lr": 0.1},
        "cbm_mode": "hybrid",
        "train_mode": "Std",
        "mtl_mode": "normal",
        "weighted_bce": False,
        "hsic_weight": 0,
        "res_dim": 0,
        "ignore_intervenes": False,
    }

    model = CBM(dm=mock_dm, **cfg)
    model.eval()  # Set to eval mode

    # Create dummy batch: (images, labels, concepts)
    images = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 10, (2,))
    concepts = torch.randint(0, 2, (2, 20)).float()
    batch = (images, labels, concepts)
    batch_idx = 0

    # Run training step to get loss
    loss = model.training_step(batch, batch_idx)

    # Verify loss is a tensor
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0, "Loss should be scalar"


def test_checkpoint_loading_imports():
    """Verify checkpoint loading function can be imported."""
    from main import load_checkpoint
    assert load_checkpoint is not None
