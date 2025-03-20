import torch
from torch import nn
from model import CBM
from utils import initialize_weights


class backbone(CBM):
    def __init__(self, **kwargs):
        kwargs["concept_weight"] = 0
        super().__init__(**kwargs)
        in_features = self.base.fc.in_features
        self.base = nn.Sequential(*list(self.base.children())[:-1])
        self.classifier = nn.Linear(in_features, self.num_classes).apply(
            initialize_weights
        )

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, torch.zeros(x.shape[0], self.num_concepts).to(x.device)
