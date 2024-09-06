import lightning as L
from utils import batchnorm_no_update_context, initialize_weights
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from attacks import PGD, AutoAttack


class CBM(L.LightningModule):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        use_pretrained: bool,
        concept_weight: float,
        lr: float,
        optimizer: str,
        scheduler_patience: int,
        use_concept_logits: bool = True,
        adv_mode: bool = False,
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
            self.base.fc = nn.Linear(2048, num_concepts).apply(initialize_weights)
        else:
            raise ValueError("Unknown base model")

        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, 3 * num_concepts),
            nn.ReLU(),
            nn.Linear(3 * num_concepts, num_classes),
        ).apply(initialize_weights)

        self.concept_acc = Accuracy(task="multilabel", num_labels=num_concepts)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.adv_concept_acc = Accuracy(task="multilabel", num_labels=num_concepts)
        self.adv_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.adv_mode = adv_mode

    def setup(self, stage):
        self.train_atk = PGD(self, steps=7)
        self.pgd_atk = PGD(self)
        self.auto_atk = AutoAttack(self, n_classes=self.hparams.num_classes)
        self.eval_atk = self.pgd_atk
        self.get_adv_img = False

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
                    patience=self.hparams.scheduler_patience,
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
        label_pred = self.classifier(
            concept_pred
            if self.hparams.use_concept_logits
            else torch.sigmoid(concept_pred)
        )
        if self.get_adv_img:
            return label_pred
        return label_pred, concept_pred

    @torch.enable_grad()
    @torch.inference_mode(False)
    def generate_adv_img(self, img, label, stage):
        with batchnorm_no_update_context(self):
            self.get_adv_img = True
            if stage == "train":
                self.train_atk.set_device(self.device)
                img = self.train_atk(img, label)
            else:
                self.eval_atk.set_device(self.device)
                img = self.eval_atk(img, label)
            self.get_adv_img = False
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
            adv_img = self.generate_adv_img(img, label, "train")
            img = torch.cat([img, adv_img], dim=0)
            label = torch.cat([label, label], dim=0)
            concepts = torch.cat([concepts, concepts], dim=0)
        loss, label_pred, concept_pred = self.shared_step(img, label, concepts)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.adv_mode:
            adv_img = self.generate_adv_img(img, label, "val")
            img = torch.cat([adv_img, img], dim=0)
            loss, label_pred, concept_pred = self.shared_step(
                img,
                torch.cat([label, label], dim=0),
                torch.cat([concepts, concepts], dim=0),
            )
            adv_label_pred, label_pred = torch.chunk(label_pred, 2)
            adv_concept_pred, concept_pred = torch.chunk(concept_pred, 2)
            self.adv_concept_acc(adv_concept_pred, concepts)
            self.adv_acc(adv_label_pred, label)
            self.log(
                "adv_val_concept_acc",
                self.adv_concept_acc,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
            self.log(
                "adv_val_acc", self.adv_acc, prog_bar=True, on_epoch=True, on_step=False
            )
        else:
            loss, label_pred, concept_pred = self.shared_step(img, label, concepts)
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
            adv_img = self.generate_adv_img(img, label, "test")
            img = torch.cat([adv_img, img], dim=0)
            loss, label_pred, concept_pred = self.shared_step(
                img,
                torch.cat([label, label], dim=0),
                torch.cat([concepts, concepts], dim=0),
            )
            adv_label_pred, label_pred = torch.chunk(label_pred, 2)
            adv_concept_pred, concept_pred = torch.chunk(concept_pred, 2)
            self.adv_concept_acc(adv_concept_pred, concepts)
            self.adv_acc(adv_label_pred, label)
            self.log(
                "adv_test_concept_acc",
                self.adv_concept_acc,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
            self.log(
                "adv_test_acc",
                self.adv_acc,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
        else:
            loss, label_pred, concept_pred = self.shared_step(img, label, concepts)
        self.concept_acc(concept_pred, concepts)
        self.acc(label_pred, label)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test_concept_acc",
            self.concept_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log("test_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)
