import os
import random
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    """
    Set seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_top_k_label(output,k=1):
    indices = np.argsort(output,axis=-1)[:,-k:]
    return indices

def cal_top_k(output,label,k=1):
    indices = get_top_k_label(output, k=k)
    y = np.reshape(label,[-1,1])
    correct = (y==indices).sum()
    return correct

def gen_mask(loader, num_attributes, n_features=64):
    attrs = []
    labels = []

    for _, label, attr in loader:
        attrs.append(attr.numpy())
        labels.append(label.numpy())

    attrs = np.concatenate(attrs, axis=0)
    labels = np.concatenate(labels, axis=0)

    mi = mutual_info_classif(attrs, labels)
    top_features = np.argsort(mi)[-n_features:]

    mask = np.zeros(num_attributes, dtype=np.float32)
    mask[top_features] = 1.0

    return mask

def cal_class_imbalance_weights(loader, num_attributes):
    n_ones = torch.zeros(num_attributes, dtype=torch.float)
    total_samples = 0
    for img, label, attr in loader:
        n_ones += torch.sum(attr, dim=0).float()
        total_samples += attr.size(0)

    imbalance_ratio = total_samples / n_ones - 1
    return imbalance_ratio.tolist()