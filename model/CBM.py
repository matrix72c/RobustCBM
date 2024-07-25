import lightning as L
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
        concept_weight: float = 0.01,
        lr: float = 1e-3,
        step_size: int = 15,
        gamma: float = 0.1,
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
            self, eps=5 / 255, alpha=2 / 225, steps=2, random_start=True
        )
        self.train_atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.val_atk = PGD(
            self, eps=5 / 255, alpha=2 / 225, steps=2, random_start=True
        )
        self.val_atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.get_adv_img = False
        self.adv_training = False

    def setup(self, stage=None):
        data_module = self.trainer.datamodule
        if hasattr(data_module, "imbalance_ratio"):
            self.data_weight = torch.Tensor(data_module.imbalance_ratio).to(self.device)
        else:
            self.data_weight = None
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        concept_pred = self.base(x)
        class_pred = self.classifier(concept_pred)
        if self.get_adv_img:
            return class_pred
        return class_pred, concept_pred

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

        class_pred, concept_pred = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred, concepts, weight=self.data_weight
        )
        class_loss = F.cross_entropy(class_pred, label)
        loss = concept_loss + self.hparams.concept_weight * class_loss
        self.concept_acc(concept_pred, concepts)
        self.acc(class_pred, label)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_concept_acc",
            self.concept_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train_acc", self.acc, prog_bar=True, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self.shared_step(batch)
        self.log("val_concept_acc", self.concept_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch)
        self.log("test_concept_acc", self.concept_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)