import lightning as L
from utils import batchnorm_no_update_context, initialize_weights
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from attacks import PGD


class CBM(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        num_concepts: int,
        real_concepts: int,
        base: str = "resnet50",
        use_pretrained: bool = True,
        concept_weight: float = 1,
        lr: float = 0.1,
        scheduler_arg: int = 30,
        adv_mode: bool = False,
        hidden_dim: int = 0,
        cbm_mode: str = "hybrid", # "bool", "fuzzy", "hybrid"
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
            self.base.fc = nn.Linear(self.base.fc.in_features, num_concepts).apply(
                initialize_weights
            )
        elif base == "inceptionv3":
            self.base = torchvision.models.inception_v3(
                weights=(
                    torchvision.models.Inception3_Weights.DEFAULT
                    if use_pretrained
                    else None
                ),
                aux_logits=False,
            )
            self.base.fc = nn.Linear(self.base.fc.in_features, num_concepts).apply(
                initialize_weights
            )
        else:
            raise ValueError("Unknown base model")
        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(num_concepts, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, num_classes),
            ).apply(initialize_weights)
        else:
            self.classifier = nn.Linear(num_concepts, num_classes).apply(
                initialize_weights
            )
        self.real_concepts = real_concepts
        self.num_classes = num_classes
        self.num_concepts = num_concepts

        self.concept_acc = Accuracy(task="multilabel", num_labels=min(num_concepts, real_concepts))
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.acc5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.acc10 = Accuracy(task="multiclass", num_classes=num_classes, top_k=10)

        self.adv_mode = adv_mode
        self.train_atk = PGD(self, eps=4 / 255, alpha=4 /2550.0, steps=10)
        self.eval_atk = PGD(self, eps=4 / 255, alpha=4 /2550.0, steps=10)


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr
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
        concept_pred = self.base(x)
        if self.hparams.cbm_mode == "fuzzy":
            concept_pred = torch.sigmoid(concept_pred)
        elif self.hparams.cbm_mode == "bool":
            concept_pred = torch.sigmoid(concept_pred)
            concept_pred = torch.where(concept_pred > 0.5, 1, 0).float()
        elif self.hparams.cbm_mode == "hybrid":
            label_pred = self.classifier(concept_pred)
        return label_pred, concept_pred

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
        return img.clone().detach()

    def shared_step(self, img, label, concepts):
        label_pred, concept_pred = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(concept_pred, concepts)
        label_loss = F.cross_entropy(label_pred, label)
        loss = label_loss + self.hparams.concept_weight * concept_loss
        return loss, label_pred, concept_pred

    def training_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.adv_mode:
            bs = img.shape[0] // 2
            adv_img = self.generate_adv_img(img[:bs], label[:bs], "train")
            img = torch.cat([img[:bs], adv_img], dim=0)
            label = torch.cat([label[:bs], label[:bs]], dim=0)
            concepts = torch.cat([concepts[:bs], concepts[:bs]], dim=0)
        loss, label_pred, concept_pred = self.shared_step(img, label, concepts)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.adv_mode:
            img = self.generate_adv_img(img, label, "eval")
        loss, label_pred, concept_pred = self.shared_step(img, label, concepts)
        if concept_pred.shape[1] > self.real_concepts:
            concept_pred = concept_pred[:, : self.real_concepts]
            concepts = concepts[:, : self.real_concepts]
        self.concept_acc(concept_pred, concepts)
        self.acc(label_pred, label)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_concept_acc",
            self.concept_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log("val_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.adv_mode:
            img = self.generate_adv_img(img, label, "eval")
        loss, label_pred, concept_pred = self.shared_step(img, label, concepts)
        if concept_pred.shape[1] > self.real_concepts:
            concept_pred = concept_pred[:, : self.real_concepts]
            concepts = concepts[:, : self.real_concepts]
        self.concept_acc(concept_pred, concepts)
        self.acc(label_pred, label)
        self.acc5(label_pred, label)
        self.acc10(label_pred, label)
        self.log("concept_acc", self.concept_acc, on_epoch=True, on_step=False)
        self.log("acc", self.acc, on_epoch=True, on_step=False)
        self.log("acc5", self.acc5, on_epoch=True, on_step=False)
        self.log("acc10", self.acc10, on_epoch=True, on_step=False)
