import torch
from torch import nn
import torch.nn.functional as F
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

    def calc_loss(self, img, label, concepts):
        label_pred, concept_pred = self(img)
        label_loss = F.cross_entropy(label_pred, label, label_smoothing=0.1)
        return {"Label Loss": label_loss, "Loss": label_loss}, (
            label_pred,
            concept_pred,
        )
