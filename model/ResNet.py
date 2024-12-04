import lightning as L
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from attacks import AutoAttack
from utils import batchnorm_no_update_context, initialize_weights


class ResNet(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        use_pretrained: bool,
        lr: float,
        optimizer: str,
        scheduler_arg: int,
        adv_mode: bool,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.base = torchvision.models.resnet50(
            weights=(
                torchvision.models.ResNet50_Weights.DEFAULT if use_pretrained else None
            ),
        )
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes).apply(
            initialize_weights
        )
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.acc5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.acc10 = Accuracy(task="multiclass", num_classes=num_classes, top_k=10)

        self.adv_mode = adv_mode
        self.train_atk = AutoAttack(self, eps=8 / 255, n_classes=num_classes)
        self.eval_atk = AutoAttack(self, eps=8 / 255, n_classes=num_classes)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), lr=self.hparams.lr, weight_decay=5e-4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.1,
                    patience=self.hparams.scheduler_arg,
                    min_lr=1e-4,
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    def forward(self, x):
        x = self.base(x)
        return x

    @torch.enable_grad()
    @torch.inference_mode(False)
    def generate_adv_img(self, img, label, stage):
        with batchnorm_no_update_context(self):
            if stage == "train":
                self.train_atk.set_device(self.device)
                img = self.train_atk(img, label)
            else:
                self.eval_atk.set_device(self.device)
                img = self.eval_atk(img, label)
        return img

    def shared_step(self, img, label):
        logits = self(img)
        loss = F.cross_entropy(logits, label)
        return loss, logits

    def training_step(self, batch):
        img, label, _ = batch
        if self.adv_mode:
            adv_img = self.generate_adv_img(img, label, "train")
            img = torch.cat([img, adv_img], dim=0)
            label = torch.cat([label, label], dim=0)
        loss, label_pred = self.shared_step(img, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        img, label, concepts = batch
        if self.adv_mode:
            adv_img = self.generate_adv_img(img, label, "eval")
            img = torch.cat([img, adv_img], dim=0)
            label = torch.cat([label, label], dim=0)
        loss, label_pred = self.shared_step(img, label)
        self.acc(label_pred, label)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_concept_acc",
            0,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log("val_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.adv_mode:
            img = self.generate_adv_img(img, label, "eval")
        loss, label_pred = self.shared_step(img, label)
        self.acc(label_pred, label)
        self.acc5(label_pred, label)
        self.acc10(label_pred, label)
        self.log("concept_acc", 0, on_epoch=True, on_step=False)
        self.log("acc", self.acc, on_epoch=True, on_step=False)
        self.log("acc5", self.acc5, on_epoch=True, on_step=False)
        self.log("acc10", self.acc10, on_epoch=True, on_step=False)
