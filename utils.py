import ast
import math
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision


def yaml_merge(default, update):
    merged = default.copy()
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = yaml_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and v:
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def build_name(config):
    if config.get("run_name", None) is not None:
        return config["run_name"]
    d = flatten_dict(config)
    d = sorted(d.items(), key=lambda x: x[0])
    name = "_".join([f"{v}" if isinstance(v, str) else f"{k}-{v}" for k, v in d])
    name = name.lower()
    return name


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


def suppress_stdout(func):
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, "w")
        result = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return result

    return wrapper


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
    def __init__(self, model, index=0):
        super().__init__()
        self.model = model
        self.index = index

    def forward(self, *args, **kwargs):
        # Some attacks (e.g., AutoAttack Square) can end up querying the model with
        # an empty batch. TorchVision ViT does not support batch size 0 in attention.
        # Returning an empty output tensor keeps the attack code well-defined.
        if args and isinstance(args[0], torch.Tensor) and args[0].shape[0] == 0:
            x = args[0]
            if self.index == 0 and hasattr(self.model, "num_classes"):
                return x.new_empty((0, int(self.model.num_classes)))
            if self.index == 1 and hasattr(self.model, "num_concepts"):
                res_dim = getattr(getattr(self.model, "hparams", None), "res_dim", 0)
                return x.new_empty((0, int(self.model.num_concepts) + int(res_dim)))
            # Fallback: preserve batch dimension; last dim is unknown.
            return x.new_empty((0, 0))
        o = self.model(*args, **kwargs)
        return o[self.index]


def parse_value(value):
    """
    解析 CSV 文件中的值，可以解析数字与列表
    """
    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        pass

    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)  # 使用 ast 解析列表
        except (SyntaxError, ValueError) as e:
            print(e)

    return value


def get_df(names, cols):
    csv_path = os.path.join(os.path.dirname(__file__), "result.csv")
    df = pd.read_csv(csv_path)
    df = df.apply(lambda x: x.apply(parse_value))
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
    names = [str(n).strip() for n in names]
    order_df = pd.DataFrame({"name": names, "order": range(len(names))})
    merged = order_df.merge(df, on="name", how="left")
    df = merged.sort_values(by="order").reset_index(drop=True)
    df = df.drop(columns=["order"])
    concept_cols = [col for col in df.columns if "Concept" in col]
    backbone_mask = df["name"].str.contains("backbone", case=False, na=False)
    df.loc[backbone_mask, concept_cols] = np.nan
    df.replace(0, np.nan, inplace=True)

    for atk in ["LPGD", "CPGD", "JPGD", "AA"]:
        df[f"{atk} ASR"] = df.apply(
            lambda row: 1 - row[f"{atk} Acc"] / row["Std Acc"], axis=1
        )
        df[f"{atk} Concept ASR"] = df.apply(
            lambda row: (
                None
                if "backbone" in row["name"]
                else 1 - row[f"{atk} Concept Acc"] / row["Std Concept Acc"]
            ),
            axis=1,
        )
    return df[cols]