import torch
from torch import nn
from model import CBM
from utils import modify_fc


class backbone(CBM):
    def __init__(self, **kwargs):
        kwargs["concept_weight"] = 0
        super().__init__(**kwargs)
        modify_fc(self.base, kwargs["base"], self.num_classes)
        del self.classifier

    def forward(self, x):
        x = self.base(x)
        return x, torch.zeros(x.shape[0], self.num_concepts).detach().to(x.device)
