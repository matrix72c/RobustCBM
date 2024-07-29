import math
from numbers import Number
import lightning as L
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from model import CBM
from model.scaler import Scaler


class RCBM(CBM):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        use_pretrained: bool = True,
        concept_weight: float = 0.01,
        lr: float = 0.001,
        step_size: int = 15,
        gamma: float = 0.1,
        vib_lambda: Number = 0.01,
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
        )
        self.base.fc = nn.Linear(self.base.fc.in_features, 2 * num_concepts)  # encoder
        self.mu_bn = nn.BatchNorm1d(num_concepts, affine=False)
        self.logvar_bn = nn.BatchNorm1d(num_concepts, affine=False)
        self.scaler = Scaler(num_concepts)

    def forward(self, x):
        statistics = self.base(x)
        logvar, mu = torch.chunk(statistics, 2, dim=1)
        mu = self.mu_bn(mu)
        mu = self.scaler(mu, "positive")
        logvar = self.logvar_bn(logvar)
        logvar = self.scaler(logvar, "negative")
        std = F.softplus(logvar - 5, beta=1)
        concept_pred = mu + std * torch.randn_like(std)
        class_pred = self.classifier(concept_pred)
        if self.get_adv_img:
            return class_pred
        return class_pred, concept_pred, mu, std

    def shared_step(self, batch):
        img, label, concepts = batch
        if self.adv_training:
            with torch.enable_grad():
                self.get_adv_img = True
                self.eval()
                if self.trainer.training:
                    self.train_atk.set_device(self.device)
                    img = self.train_atk(img, label)
                else:
                    self.val_atk.set_device(self.device)
                    img = self.val_atk(img, label)
                self.train()
                self.get_adv_img = False

        class_pred, concept_pred, mu, std = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred, concepts, weight=self.data_weight
        )
        info_loss = (
            -0.5 * torch.mean(1 + 2 * std.log() - mu.pow(2) - std.pow(2)) / math.log(2)
        )
        self.log("info_loss", info_loss)
        class_loss = F.cross_entropy(class_pred, label)
        loss = (
            concept_loss
            + self.hparams.concept_weight * class_loss
            + info_loss * self.hparams.vib_lambda
        )
        self.concept_acc(concept_pred, concepts)
        self.acc(class_pred, label)
        return loss
