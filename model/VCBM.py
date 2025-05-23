import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from mtl import mtl
from utils import calc_info_loss


class VCBM(CBM):
    def __init__(
        self,
        res_dim: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fc = nn.Linear(self.base.fc.in_features, 2 * (self.num_concepts + res_dim))
        self.base.fc = nn.Identity()
        self.classifier = nn.Linear(self.num_concepts + res_dim, self.num_classes)

    def forward(self, x, concept_pred=None):
        features = self.base(x)
        statistics = self.fc(features)
        std, mu = torch.chunk(statistics, 2, dim=1)
        logits = mu + std * torch.randn_like(std)
        if concept_pred is None:
            concept_pred = logits
        else:
            logits[:, : self.num_concepts] = concept_pred
            concept_pred = logits

        label_pred = self.classifier(concept_pred)
        return label_pred, concept_pred[:, : self.num_concepts], mu, std**2

    def calc_loss(self, img, label, concepts):
        label_pred, concept_pred, mu, var = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred,
            concepts,
            weight=(
                self.dm.imbalance_weights.to(self.device)
                if self.hparams.weighted_bce
                else None
            ),
        )
        info_loss = calc_info_loss(mu, var)
        label_loss = F.cross_entropy(label_pred, label)
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.vib * info_loss
        )
        losses = {
            "label_loss": label_loss,
            "concept_loss": concept_loss,
            "info_loss": info_loss,
            "loss": loss,
        }

        if self.hparams.mtl_mode != "normal":
            g = mtl([label_loss, concept_loss], self, self.hparams.mtl_mode)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return losses, (label_pred, concept_pred, mu, var)
