import math
from numbers import Number
import lightning as L
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from model import CBM


class VCEM(CBM):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        embed_size: int = 16,
        use_pretrained: bool = True,
        concept_weight: float = 0.5,
        lr: float = 1e-4,
        step_size: list = [20, 40],
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
        self.base.fc = nn.Linear(
            self.base.fc.in_features, 4 * embed_size * num_concepts
        )
        self.concept_prob_gen = nn.Linear(2 * embed_size * num_concepts, num_concepts)
        self.classifier = nn.Linear(embed_size * num_concepts, num_classes)

    def forward(self, x):
        statistics = self.base(x)
        std, mu = torch.chunk(statistics, 2, dim=1)
        concept_context = mu + std * torch.randn_like(std)

        concept_pred = self.concept_prob_gen(concept_context)

        pos_embed, neg_embed = torch.chunk(concept_context, 2, dim=1)
        pos_embed, neg_embed = pos_embed.view(
            pos_embed.size(0), -1, self.hparams.embed_size
        ), neg_embed.view(neg_embed.size(0), -1, self.hparams.embed_size)
        concept_pred.unsqueeze_(-1)
        combined_embed = pos_embed * concept_pred + neg_embed * (1 - concept_pred)
        concept_embed = combined_embed.view(combined_embed.size(0), -1)
        concept_pred = concept_pred.squeeze(-1)

        class_pred = self.classifier(concept_embed)
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
        self.log("var", var.mean(), prog_bar=True, on_step=True, on_epoch=False)
        self.log("mu", mu.mean(), prog_bar=True, on_step=True, on_epoch=False)
        class_loss = F.cross_entropy(class_pred, label)
        loss = (
            class_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.vib_lambda * info_loss
        )
        self.concept_acc(concept_pred, concepts)
        self.acc(class_pred, label)
        return loss
