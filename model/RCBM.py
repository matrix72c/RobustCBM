import math
import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from model.scaler import Scaler
from utils import initialize_weights


class RCBM(CBM):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        use_pretrained: bool,
        concept_weight: float,
        lr: float,
        optimizer: str,
        vib_lambda: float,
        scheduler_patience: int,
        classifier: str = "FC",
        adv_mode: bool = False,
    ):
        super().__init__(
            base,
            num_classes,
            num_concepts,
            use_pretrained,
            concept_weight,
            lr,
            optimizer,
            scheduler_patience,
            classifier,
            adv_mode,
        )
        self.base.fc = nn.Linear(self.base.fc.in_features, 2 * num_concepts).apply(initialize_weights)  # encoder
        self.mu_bn = nn.BatchNorm1d(num_concepts, affine=False).apply(initialize_weights)
        self.std_bn = nn.BatchNorm1d(num_concepts, affine=False).apply(initialize_weights)
        self.scaler = Scaler()

    def forward(self, x):
        statistics = self.base(x)
        std, mu = torch.chunk(statistics, 2, dim=1)
        mu = self.mu_bn(mu)
        mu = self.scaler(mu, "positive")
        std = self.std_bn(std)
        std = self.scaler(std, "negative")
        concept_pred = mu + std * torch.randn_like(std)
        label_pred = self.classifier(concept_pred)
        if self.get_adv_img:
            return label_pred
        return label_pred, concept_pred, mu, std**2

    def shared_step(self, img, label, concepts):
        label_pred, concept_pred, mu, var = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(concept_pred, concepts)
        var = torch.clamp(var, min=1e-8)
        info_loss = 0.5 * torch.mean(mu**2 + var - var.log() - 1) / math.log(2)
        self.log(
            "info_loss",
            info_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        class_loss = F.cross_entropy(label_pred, label)
        loss = (
            class_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.vib_lambda * info_loss
        )
        return loss, label_pred, concept_pred
