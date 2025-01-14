import torch
from torch import nn
from model import CBM
from utils import modify_fc


class backbone(CBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        modify_fc(self.base, kwargs["base"], self.num_classes)

    def forward(self, x):
        return self.base(x), torch.zeros(x.shape[0], self.num_concepts).to(x.device)
