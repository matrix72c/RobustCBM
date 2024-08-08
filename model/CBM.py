import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchattacks import PGD


class CBM(L.LightningModule):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        use_pretrained: bool = True,
        concept_weight: float = 0.5,
        lr: float = 1e-3,
        step_size: list = [15, 30, 100],
        gamma: float = 0.1,
        adv_training: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        if base == "resnet50":
            self.base = torchvision.models.resnet50(
                weights=(
                    torchvision.models.ResNet50_Weights.DEFAULT
                    if use_pretrained
                    else None
                ),
            )
            self.base.fc = nn.Linear(self.base.fc.in_features, num_concepts)
        elif base == "inceptionv3":
            self.base = torchvision.models.inception_v3(
                weights=(
                    torchvision.models.Inception3_Weights.DEFAULT
                    if use_pretrained
                    else None
                ),
            )
            self.base.fc = nn.Linear(2048, num_concepts)
        else:
            raise ValueError("Unknown base model")
        self.classifier = nn.Linear(num_concepts, num_classes)
        self.data_weight = None

        self.concept_acc = Accuracy(task="multilabel", num_labels=num_concepts)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_atk = PGD(
            self, eps=8 / 255, alpha=2 / 225, steps=4, random_start=True
        )
        self.train_atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.val_atk = PGD(self)
        self.val_atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.test_atk = PGD(self)
        self.get_adv_img = False
        self.adv_training = adv_training

    def setup(self, stage=None):
        data_module = self.trainer.datamodule
        self.data_weight = None
        if hasattr(data_module, "imbalance_ratio"):
            self.data_weight = torch.Tensor(data_module.imbalance_ratio).to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hparams.step_size, gamma=self.hparams.gamma
        )

        return [optimizer], [scheduler]

    def forward(self, x):
        concept_pred = self.base(x)
        class_pred = self.classifier(concept_pred)
        if self.get_adv_img:
            return class_pred
        return class_pred, concept_pred

    def generate_adv_img(self, img, label, stage):
        with torch.inference_mode(False):
            self.get_adv_img = True
            self.eval()
            if stage == "train":
                self.train_atk.set_device(self.device)
                img = self.train_atk(img, label)
            elif stage == "val":
                self.val_atk.set_device(self.device)
                img = self.val_atk(img, label)
            elif stage == "test":
                self.test_atk.set_device(self.device)
                img = self.test_atk(img, label)
            self.train()
            self.get_adv_img = False
        return img

    def shared_step(self, batch, stage):
        img, label, concepts = batch
        if self.adv_training:
            img = self.generate_adv_img(img, label, stage)

        class_pred, concept_pred = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred,
            concepts,
            weight=self.data_weight if stage == "train" else None,
        )
        class_loss = F.cross_entropy(class_pred, label)
        loss = class_loss + self.hparams.concept_weight * concept_loss
        self.concept_acc(concept_pred, concepts)
        self.acc(class_pred, label)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_concept_acc",
            self.concept_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log("train_acc", self.acc, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self.shared_step(batch, "val")
        self.log(
            "val_concept_acc",
            self.concept_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log("val_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, "test")
        self.log(
            "test_concept_acc",
            self.concept_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log("test_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)
