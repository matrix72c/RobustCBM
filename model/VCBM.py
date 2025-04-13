import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from mtl import mtl
from utils import calc_info_loss, calc_spectral_norm


class VCBM(CBM):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fc = nn.Linear(self.base.fc.in_features, 2 * self.num_concepts)
        self.base.fc = nn.Identity()

    def forward(self, x, concept_pred=None):
        features = self.base(x)
        statistics = self.fc(features)
        std, mu = torch.chunk(statistics, 2, dim=1)
        if concept_pred is None:
            if self.training:
                concept_pred = mu + std * torch.randn_like(std)
            else:
                concept_pred = mu

        label_pred = self.classifier(concept_pred)
        return label_pred, concept_pred, mu, std**2

    def calc_loss(self, img, label, concepts):
        label_pred, concept_pred, mu, var = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        info_loss = calc_info_loss(mu, var)
        label_loss = F.cross_entropy(label_pred, label)
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.vib * info_loss
        )
        if self.adv_mode == "adv" and self.hparams.trades > 0:
            clean_concept_pred, adv_concept_pred = torch.chunk(mu, 2, dim=0)
            clean_probs = torch.sigmoid(clean_concept_pred.detach()).clamp(
                1e-5, 1 - 1e-5
            )
            adv_probs = torch.sigmoid(adv_concept_pred).clamp(1e-5, 1 - 1e-5)

            trades_loss = F.binary_cross_entropy(
                adv_probs, clean_probs, reduction="mean"
            )
            loss += trades_loss * self.hparams.trades * min(1, self.current_epoch / 300)
            self.log("trades_loss", trades_loss)

        if self.hparams.spectral_weight > 0:
            loss += calc_spectral_norm(self.fc) * self.hparams.spectral_weight

        if self.hparams.mtl_mode != "normal":
            g = mtl([label_loss, concept_loss], self, self.hparams.mtl_mode)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return loss, (label_pred, concept_pred, mu, var)
