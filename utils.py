from contextlib import contextmanager
from torch.nn.modules.batchnorm import _BatchNorm
import torch

def cal_class_imbalance_weights(data):
    n = len(data)
    n_attr = len(data[0]["attribute_label"])
    n_ones = torch.zeros(n_attr, dtype=torch.float)
    total = [n] * n_attr
    for d in data:
        attr = d["attribute_label"]
        for i in range(n_attr):
            n_ones[i] += attr[i]
    imbalance_ratio = []
    for i in range(n_attr):
        imbalance_ratio.append(total[i] / n_ones[i] - 1)
    return imbalance_ratio



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
