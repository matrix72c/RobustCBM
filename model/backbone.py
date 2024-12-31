import torch
from torch import nn
from model import CBM
from utils import initialize_weights


class backbone(CBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_classes).apply(
            initialize_weights
        )

    def forward(self, x):
        return self.base(x), torch.zeros(x.shape[0], self.num_concepts).to(x.device)
