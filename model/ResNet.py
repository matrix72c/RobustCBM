import os
import lightning as L
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from attacks import PGD, AutoAttack
from utils import batchnorm_no_update_context


class ResNet(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        use_pretrained: bool = True,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = torchvision.models.resnet50(
            weights=(
                torchvision.models.ResNet50_Weights.DEFAULT if use_pretrained else None
            ),
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.adv_training = False
        self.at_loss = nn.CrossEntropyLoss(reduction="sum")
        self.train_atk = PGD(
            self,
            self.at_loss,
            eps=8 / 255,
            nb_iters=7,
            rand_init=True,
            loss_scale=1,
            params_switch_grad_req=list(self.parameters()),
        )
        self.pgd_atk = PGD(
            self,
            self.at_loss,
            eps=8 / 255,
            nb_iters=10,
            rand_init=True,
            loss_scale=1,
            params_switch_grad_req=list(self.parameters()),
        )
        self.auto_atk = AutoAttack(self, norm="Linf", eps=8 / 255, version="standard", verbose=False)
        self.eval_atk = "PGD"

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        x = self.model(x)
        return x

    @torch.enable_grad()
    @torch.inference_mode(False)
    def generate_adv_img(self, img, label, stage):
        with batchnorm_no_update_context(self):
            if stage == "train":
                img = self.train_atk.perturb(img, label)
            elif stage == "val":
                img = self.pgd_atk.perturb(img, label)
            elif stage == "test":
                if self.eval_atk == "PGD":
                    img = self.pgd_atk.perturb(img, label)
                elif self.eval_atk == "AA":
                    img = self.auto_atk.run_standard_evaluation(img.clone().detach(), label.clone().detach(), bs=len(img))
        return img.clone().detach().to(self.device)

    def shared_step(self, batch, stage):
        img, label, _ = batch
        if self.adv_training:
            adv_img = self.generate_adv_img(img, label, stage)
            img = torch.cat([img, adv_img], dim=0)
            label = torch.cat([label, label], dim=0)
        logits = self(img)
        loss = F.cross_entropy(logits, label)
        self.acc(logits, label)
        return loss

    def training_step(self, batch):
        loss = self.shared_step(batch, "train")
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.acc, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch):
        _ = self.shared_step(batch, "val")
        self.log("val_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch):
        _ = self.shared_step(batch, "test")
        self.log("test_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_concept_acc", 0.0, on_epoch=True, on_step=False)
