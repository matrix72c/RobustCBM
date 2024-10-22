from contextlib import contextmanager
from torch.nn.modules.batchnorm import _BatchNorm
import torch
import torch.nn as nn

def initialize_weights(module: nn.Module):
    """Initialize the weights of a module."""
    if isinstance(module, nn.Sequential):
        for m in module:
            initialize_weights(m)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


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
