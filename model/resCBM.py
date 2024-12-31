import torch
from torch import nn
from model import CBM
from utils import initialize_weights


class resCBM(CBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base.fc = nn.Linear(
            self.base.fc.in_features, self.num_concepts + self.num_classes
        ).apply(initialize_weights)

    def forward(self, x):
        logits = self.base(x)
        concept_pred = logits[:, : self.num_concepts]
        label_res = logits[:, self.num_concepts :]
        label_pred = self.classifier(concept_pred) + label_res
        return label_pred, concept_pred
