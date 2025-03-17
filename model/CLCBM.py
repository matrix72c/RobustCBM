import torch
from torch import nn
from model import CBM
from utils import initialize_weights, contrastive_loss, modify_fc
import torch.nn.functional as F


class CLCBM(CBM):
    def __init__(self, embed_dim: int = 32, cl_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        modify_fc(self.base, kwargs["base"], embed_dim * self.num_concepts)

        self.concept_prob = nn.Linear(
            embed_dim * self.num_concepts, self.num_concepts
        ).apply(initialize_weights)

    def forward(self, x, concept_pred=None):
        z = self.base(x)
        if concept_pred is None:
            concept_pred = self.concept_prob(z.view(z.size(0), -1))
        concept_features = z.view(z.size(0), self.num_concepts, -1)
        label_pred = self.classifier(concept_pred)
        return label_pred, concept_pred, concept_features

    def calc_loss(self, img, label, concepts):
        label_pred, concept_pred, z = self(img)
        label_loss = F.cross_entropy(label_pred, label)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        cl_loss = contrastive_loss(z, z, concepts)
        loss = (
            label_loss
            + concept_loss * self.hparams.concept_weight
            + cl_loss * self.hparams.cl_weight
        )
        self.log("cl_loss", cl_loss, prog_bar=True)
        return loss, (label_pred, concept_pred)
