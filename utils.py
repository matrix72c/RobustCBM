from contextlib import contextmanager
from torch.nn.modules.batchnorm import _BatchNorm
import torch

@contextmanager
def eval_context(net: torch.nn.Module):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        if istrain:
            net.eval()
        yield net
    finally:
        if istrain:
            net.train()


@contextmanager
def train_context(net: torch.nn.Module):
    """Temporarily switch to training mode."""
    istrain = net.training
    try:
        if not istrain:
            net.train()
        yield net
    finally:
        if not istrain:
            net.eval()


@contextmanager
def batchnorm_no_update_context(net: torch.nn.Module):
    """Temporarily disable batchnorm update."""
    istrain = net.training
    try:
        if istrain:
            for module in net.modules():
                if isinstance(module, _BatchNorm):
                    module.track_running_stats = False
        yield net
    finally:
        if istrain:
            for module in net.modules():
                if isinstance(module, _BatchNorm):
                    module.track_running_stats = True
