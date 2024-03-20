import os
import random
import numpy as np
import torch


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


def test_acc(model, data_loader):
    model.eval()
    label_acc_meter = AverageMeter()
    attr_acc_meter = AverageMeter()
    with torch.no_grad():
        for img, label, attr in data_loader:
            img, label, attr = img.cuda(), label.cuda(), attr.cuda()
            attr_pred, label_pred = model(img)
            label_pred = torch.argmax(label_pred, dim=1)
            correct = torch.sum(label_pred == label).int().sum().item()
            num = len(label)
            label_acc_meter.update(correct / num, num)
            attr_pred = torch.sigmoid(attr_pred).ge(0.5)
            attr_correct = torch.sum(attr_pred == attr).int().sum().item()
            attr_num = attr.shape[0] * attr.shape[1]
            attr_acc_meter.update(attr_correct / attr_num, attr_num)
    return label_acc_meter.avg, attr_acc_meter.avg

def get_top_k_label(output,k=1):
    indices = np.argsort(output,axis=-1)[:,-k:]
    return indices

def cal_top_k(output,label,k=1):
    indices = get_top_k_label(output, k=k)
    y = np.reshape(label,[-1,1])
    correct = (y==indices).sum()
    return correct