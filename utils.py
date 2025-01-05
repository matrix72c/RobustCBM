from contextlib import contextmanager
import hashlib
import json
import math
import os
from torch.nn.modules.batchnorm import _BatchNorm
import torch
import torch.nn as nn
from lightning.pytorch.plugins.io import CheckpointIO
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import tempfile


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


def cal_class_imbalance_weights(dataset: torch.utils.data.Dataset):
    """Calculate the class imbalance weights."""
    n = len(dataset)

    _, _, first_attr_label = dataset[0]
    n_attr = first_attr_label.numel()

    n_ones = torch.zeros(n_attr, dtype=torch.float)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, num_workers=24, shuffle=False
    )
    for batch in dataloader:
        _, _, attr_labels = batch
        n_ones += torch.sum(attr_labels, dim=0)
    imbalance_ratio = []
    for count in n_ones:
        imbalance_ratio.append(n / count.item() - 1)

    return torch.tensor(imbalance_ratio)


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


def calc_info_loss(mu, var):
    var = torch.clamp(var, min=1e-8)  # avoid var -> 0
    info_loss = -0.5 * torch.mean(1 + var.log() - mu.pow(2) - var) / math.log(2)
    return info_loss


def get_md5(obj):
    args_str = json.dumps(obj, sort_keys=True)
    return hashlib.md5(args_str.encode()).hexdigest()


class OssCheckpointIO(CheckpointIO):
    def __init__(self, bucket: oss2.Bucket):
        super().__init__()
        self.bucket = bucket

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        key = os.path.relpath(path, os.getcwd())
        with tempfile.TemporaryFile() as f:
            torch.save(checkpoint, f)
            f.seek(0)
            self.bucket.put_object(key, f)

    def load_checkpoint(self, path, map_location=None):
        key = os.path.relpath(path, os.getcwd())
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, os.path.basename(key))
            self.bucket.get_object_to_file(key, fp)
            with open(fp, "rb") as f:
                ckpt = torch.load(f, map_location=map_location)
        return ckpt

    def remove_checkpoint(self, path):
        path = os.path.relpath(path, os.getcwd())
        self.bucket.delete_object(path)


def get_oss():
    bucket_name, endpoint, region = (
        os.environ["OSS_BUCKET"],
        os.environ["OSS_ENDPOINT"],
        os.environ["OSS_REGION"],
    )
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
    return bucket
