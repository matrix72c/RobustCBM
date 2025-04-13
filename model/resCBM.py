import torch
from torch import nn
from model import CBM
from utils import initialize_weights


class resCBM(CBM):
    def __init__(self, res_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.base.fc = nn.Linear(
            self.base.fc.in_features, self.num_concepts + res_dim
        ).apply(initialize_weights)
        self.classifier = nn.Linear(
            self.num_concepts + res_dim, self.num_classes
        ).apply(initialize_weights)

    def forward(self, x, concept_pred=None):
        logits = self.base(x)
        if concept_pred is None:
            concept_pred = logits[:, : self.num_concepts]
        label_res = logits[:, self.num_concepts :]
        label_pred = self.classifier(torch.cat([concept_pred, label_res], dim=1))
        return label_pred, concept_pred
