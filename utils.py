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
from osstorchconnector import OssCheckpoint
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
    cfg.pop("trainer")
    cfg.pop("config")
    for k, v in cfg.items():
        if v is None or v is False:
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
    def __init__(self, bucket, connector, name):
        super().__init__()
        self.bucket = bucket
        self.connector = connector
        self.name = name

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        path = os.path.relpath(path, os.getcwd())
        uri = f"oss://{self.name}/{path}"
        with self.connector.writer(uri) as writer:
            torch.save(checkpoint, writer)

    def load_checkpoint(self, path, map_location=None):
        path = os.path.relpath(path, os.getcwd())
        uri = f"oss://{self.name}/{path}"
        with self.connector.reader(uri) as reader:
            ckpt = torch.load(reader, map_location=map_location)
        return ckpt

    def remove_checkpoint(self, path):
        path = os.path.relpath(path, os.getcwd())
        self.bucket.delete_object(path)

def get_oss():
    bucket_name, endpoint, region = os.environ['OSS_BUCKET'], os.environ['OSS_ENDPOINT'], os.environ['OSS_REGION']
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
    connector = OssCheckpoint(endpoint, os.environ["OSS_CRED_PATH"], os.environ["OSS_CONFIG_PATH"])
    return bucket, connector