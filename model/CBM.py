import lightning as L
from utils import batchnorm_no_update_context
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
        use_pretrained: bool = True,
        concept_weight: float = 0.5,
        lr: float = 1e-3,
        step_size: list = [10, 30, 45],
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

        self.concept_acc = Accuracy(task="multilabel", num_labels=num_concepts)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)

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
        self.get_adv_img = False
        self.adv_training = adv_training

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

    @torch.enable_grad()
    @torch.inference_mode(False)
    def generate_adv_img(self, img, label, stage):
        with batchnorm_no_update_context(self):
            self.get_adv_img = True
            if stage == "train":
                img = self.train_atk.perturb(img, label)
            elif stage == "val":
                img = self.pgd_atk.perturb(img, label)
            elif stage == "test":
                if self.eval_atk == "PGD":
                    img = self.pgd_atk.perturb(img, label)
                elif self.eval_atk == "AA":
                    img = self.auto_atk.run_standard_evaluation(img.clone().detach(), label.clone().detach(), bs=len(img))
            self.get_adv_img = False
        return img.clone().detach().to(self.device)

    def shared_step(self, batch, stage):
        img, label, concepts = batch
        if self.adv_training:
            adv_img = self.generate_adv_img(img, label, stage)
            img = torch.cat([img, adv_img], dim=0)
            label = torch.cat([label, label], dim=0)
            concepts = torch.cat([concepts, concepts], dim=0)

        class_pred, concept_pred = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(concept_pred, concepts)
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
