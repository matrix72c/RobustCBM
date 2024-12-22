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

def calc_info_loss(mu, var):
    var = torch.clamp(var, min=1e-8)  # avoid var -> 0
    info_loss = -0.5 * torch.mean(1 + var.log() - mu.pow(2) - var) / math.log(2)
    return info_loss

def get_args(cfg):
    args = {}
    cfg.pop("trainer", None)
    cfg.pop("config", None)
    for k, v in cfg.items():
        if v is None or v is False:
            continue
        if k in {"sweep_id", "train"}:
            continue
        if isinstance(v, dict):
            args.update({f"{kk}": vv for kk, vv in v["init_args"].items() if (vv is not None and vv is not False)})
        else:
            args[k] = v
    return args



def get_md5(obj):
    args_str = json.dumps(obj, sort_keys=True)
    return hashlib.md5(args_str.encode()).hexdigest()

class OssCheckpointIO(CheckpointIO):
    def __init__(self, bucket):
        super().__init__()
        self.bucket = bucket

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        path = os.path.relpath(path, os.getcwd())
        with open(path, "wb") as f:
            torch.save(checkpoint, f)
        self.bucket.put_object_from_file(path, path)

    def load_checkpoint(self, path, map_location=None):
        path = os.path.relpath(path, os.getcwd())
        if not os.path.exists(path):
            self.bucket.get_object_to_file(path, path)
        with open(path, "rb") as f:
            ckpt = torch.load(f, map_location=map_location)
        return ckpt

    def remove_checkpoint(self, path):
        path = os.path.relpath(path, os.getcwd())
        self.bucket.delete_object(path)

def get_oss():
    bucket_name, endpoint, region = os.environ['OSS_BUCKET'], os.environ['OSS_ENDPOINT'], os.environ['OSS_REGION']
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
    return bucket