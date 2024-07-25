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

    def shared_step(self, batch):
        img, label, _ = batch
        if self.adv_training:
            with torch.enable_grad():
                if self.trainer.training:
                    self.eval()
                    self.train_atk.set_device(self.device)
                    img = self.train_atk(img, label)
                    self.train()
                else:
                    self.val_atk.set_device(self.device)
                    img = self.val_atk(img, label)
        logits = self(img)
        loss = F.cross_entropy(logits, label)
        self.acc(logits, label)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.acc, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self.shared_step(batch)
        self.log("val_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch)
        self.log("test_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)
