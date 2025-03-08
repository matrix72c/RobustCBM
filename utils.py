import math
import os
import torch
import torch.nn as nn
from lightning.pytorch.plugins.io import CheckpointIO
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import tempfile
import torch.nn.functional as F
import torchvision


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
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)


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


def calc_info_loss(mu, var):
    var = torch.clamp(var, min=1e-8)  # avoid var -> 0
    info_loss = -0.5 * torch.mean(1 + var.log() - mu.pow(2) - var) / math.log(2)
    return info_loss


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


def modify_fc(model, base, out_size):
    if base == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, out_size).apply(initialize_weights)
    elif base == "vit":
        model.heads.head = nn.Linear(model.heads.head.in_features, out_size).apply(
            initialize_weights
        )
    elif base == "vgg16":
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, out_size
        ).apply(initialize_weights)
    elif base == "inceptionv3":
        model.fc = nn.Linear(model.fc.in_features, out_size).apply(initialize_weights)


def contrastive_loss(z, z_q, concepts, margin=1.0, lambda_neg=1.0):
    B, N, E = z.size()
    z_flat = z.view(B * N, E)
    zq_flat = z_q.view(B * N, E)
    c_flat = concepts.view(-1)
    dist_matrix = torch.cdist(z_flat, zq_flat, p=2)
    c_i = c_flat.unsqueeze(1)
    c_j = c_flat.unsqueeze(0)
    pos_mask = (c_i == 1) & (c_j == 1)
    eye_mask = torch.eye(B * N, device=z.device).bool()
    pos_mask = pos_mask & (~eye_mask)
    neg_mask = ((c_i == 1) & (c_j == 0)) | ((c_i == 0) & (c_j == 1))

    pos_loss = dist_matrix[pos_mask].pow(2).mean()
    neg_loss = F.relu(margin - dist_matrix[neg_mask]).pow(2).mean()
    return pos_loss + lambda_neg * neg_loss


def build_base(base, out_size, use_pretrained=True):
    if base == "resnet50":
        model = torchvision.models.resnet50(
            weights=(
                torchvision.models.ResNet50_Weights.DEFAULT if use_pretrained else None
            ),
        )
    elif base == "vit":
        model = torchvision.models.vit_b_16(
            weights=(
                torchvision.models.ViT_B_16_Weights.DEFAULT if use_pretrained else None
            ),
        )
    elif base == "vgg16":
        model = torchvision.models.vgg16(
            weights=(
                torchvision.models.VGG16_Weights.DEFAULT if use_pretrained else None
            ),
        )
    elif base == "inceptionv3":
        model = torchvision.models.inception_v3(
            weights=(
                torchvision.models.Inception_V3_Weights.DEFAULT
                if use_pretrained
                else None
            ),
        )
    else:
        raise ValueError("Unknown base model")
    modify_fc(model, base, out_size)
    return model


class cls_wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        o = self.model(*args, **kwargs)
        return o[0]

def calc_spectral_norm(model: nn.Linear):
    mat = model.weight.data
    u, s, v = torch.svd(mat)
    return s.max().item()