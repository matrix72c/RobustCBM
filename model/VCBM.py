import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from mtl import mtl
from utils import calc_info_loss, calc_spectral_norm, initialize_weights


class VIB(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2 * output_dim).apply(initialize_weights)

    def forward(self, x):
        statistics = self.fc(x)
        std, mu = torch.chunk(statistics, 2, dim=1)
        x = mu + std * torch.randn_like(std)
        loss = calc_info_loss(mu, std**2)
        return x, loss


class VCBM(CBM):
    def __init__(
        self,
        vib_lambda: float = 0.1,
        use_gate: str = "nogate",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base.fc = nn.Linear(
            self.base.fc.in_features, 2 * self.num_concepts
        )  # encoder

    def forward(self, x, concept_pred=None):
        statistics = self.base(x)
        std, mu = torch.chunk(statistics, 2, dim=1)
        if concept_pred is None:
            concept_pred = mu + std * torch.randn_like(std)

        c = torch.sigmoid(std) * 2.0 if self.hparams.use_gate == "gate" else 1.0
        label_pred = self.classifier(concept_pred * c)
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
            clean_mu, adv_mu = torch.chunk(mu, 2, dim=1)
            trades_loss = (
                F.binary_cross_entropy(
                    torch.sigmoid(adv_mu),
                    torch.sigmoid(clean_mu).detach(),
                    reduction="none",
                )
                .mean(dim=1)
                .sum()
                * self.hparams.trades
            )
            loss += trades_loss
            self.log("trades_loss", trades_loss)

        if self.hparams.spectral_weight > 0:
            loss += calc_spectral_norm(self.classifier) * self.hparams.spectral_weight

        if self.hparams.mtl_mode != "normal":
            g = mtl([label_loss, concept_loss], self, self.hparams.mtl_mode)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return loss, (label_pred, concept_pred, mu, var)
