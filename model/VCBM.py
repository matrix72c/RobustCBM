import math
from numbers import Number
import torch
from torch import nn
import torch.nn.functional as F
from model import CBM


class VCBM(CBM):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        use_pretrained: bool = True,
        concept_weight: float = 0.5,
        lr: float = 1e-3,
        step_size: list = [10, 30, 45],
        gamma: float = 0.1,
        vib_lambda: Number = 0.01,
        adv_training: bool = False,
    ):
        super().__init__(
            base,
            num_classes,
            num_concepts,
            use_pretrained,
            concept_weight,
            lr,
            step_size,
            gamma,
            adv_training,
        )
        self.base.fc = nn.Linear(self.base.fc.in_features, 2 * num_concepts)  # encoder

    def forward(self, x):
        statistics = self.base(x)
        std, mu = torch.chunk(statistics, 2, dim=1)
        concept_pred = mu + std * torch.randn_like(std)
        class_pred = self.classifier(concept_pred)
        if self.get_adv_img:
            return class_pred
        return class_pred, concept_pred, mu, std**2

    def shared_step(self, batch, stage):
        img, label, concepts = batch
        if self.adv_training:
            adv_img = self.generate_adv_img(img, label, stage)
            img = torch.cat([img, adv_img], dim=0)
            label = torch.cat([label, label], dim=0)
            concepts = torch.cat([concepts, concepts], dim=0)

        class_pred, concept_pred, mu, var = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(concept_pred, concepts)

        var = torch.clamp(var, min=1e-8)  # avoid var -> 0
        info_loss = -0.5 * torch.mean(1 + var.log() - mu.pow(2) - var) / math.log(2)
        self.log(
            "info_loss",
            info_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        class_loss = F.cross_entropy(class_pred, label)
        loss = (
            class_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.vib_lambda * info_loss
        )
        self.concept_acc(concept_pred, concepts)
        self.acc(class_pred, label)
        return loss
