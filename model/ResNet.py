import os
import lightning as L
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchattacks import PGD


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
        self.train_atk = PGD(
            self, eps=5 / 255, alpha=2 / 225, steps=2, random_start=True
        )
        self.train_atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.val_atk = PGD(self)
        self.val_atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        x = self.model(x)
        return x

    def generate_adv_img(self, img, label, stage):
        with torch.inference_mode(False):
            if stage == "train":
                self.train_atk.set_device(self.device)
                img = self.train_atk(img, label)
            elif stage == "val":
                self.val_atk.set_device(self.device)
                img = self.val_atk(img, label)
            elif stage == "test":
                self.test_atk.set_device(self.device)
                img = self.test_atk(img, label)
        return img.clone().detach().to(self.device)

    def shared_step(self, batch, stage):
        img, label, _ = batch
        if self.adv_training:
            img = self.generate_adv_img(img, label, stage)
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
        self.log("val_acc", self.acc, prog_bar=True, on_epoch=True, on_step=True)

    def test_step(self, batch):
        _ = self.shared_step(batch, "test")
        self.log("test_acc", self.acc, prog_bar=True, on_epoch=True, on_step=True)
        self.log("test_concept_acc_epoch", 0.0, on_epoch=True)
