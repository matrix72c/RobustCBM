import math
import random
import numpy as np
import lightning as L
import wandb
from utils import initialize_weights, build_base, cls_wrapper, calc_spectral_norm
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MeanMetric
import attacks
from mtl import mtl


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CBM(L.LightningModule):
    def __init__(
        self,
        dm: L.LightningDataModule,
        base: str = "resnet50",
        use_pretrained: bool = True,
        concept_weight: float = 1,
        optimizer: str = "SGD",
        optimizer_args: dict = {"lr": 0.1, "momentum": 0.9, "weight_decay": 4e-5},
        scheduler: str = "ReduceLROnPlateau",
        scheduler_args: dict = {
            "mode": "max",
            "patience": 30,
            "factor": 0.1,
            "min_lr": 1e-5,
        },
        plateau_args: dict = {
            "monitor": "acc",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        },
        hidden_dim: int = 0,
        cbm_mode: str = "hybrid",
        mtl_mode: str = "normal",
        adv_mode: str = "std",
        **kwargs,
    ):
        super().__init__()
        if mtl_mode != "normal":
            self.automatic_optimization = False
        self.save_hyperparameters(ignore="dm")
        num_classes = dm.num_classes
        num_concepts = dm.num_concepts
        self.dm = dm
        self.max_intervene_budget = dm.max_intervene_budget
        self.concept_group_map = dm.concept_group_map
        self.group_concept_map = dm.group_concept_map
        self.base = build_base(base, num_concepts, use_pretrained)

        if hidden_dim > 0:
            self.classifier = MLP(num_concepts, hidden_dim, num_classes)
        else:
            self.classifier = nn.Linear(num_concepts, num_classes).apply(
                initialize_weights
            )
        self.num_classes = num_classes
        self.num_concepts = num_concepts

        self.concept_acc = Accuracy(task="multilabel", num_labels=num_concepts)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.losses = MeanMetric()

        self.adv_mode = adv_mode

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), **self.hparams.optimizer_args
        )
        scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler)(
            optimizer, **self.hparams.scheduler_args
        )
        if self.hparams.scheduler == "ReduceLROnPlateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.hparams.plateau_args,
                },
            }
        else:
            return [optimizer], [scheduler]

    def intervene(self, concepts, concept_pred, intervene_budget):
        c = torch.sigmoid(concept_pred).ge(0.5).long()
        for b in range(concepts.shape[0]):
            s = set()
            for idx in range(self.num_concepts):
                if c[b, idx].item() != concepts[b, idx].item():
                    s.add(self.group_concept_map[idx])
            intervene_groups = list(s)
            if len(intervene_groups) > intervene_budget:
                intervene_groups = random.sample(intervene_groups, intervene_budget)
            for group in intervene_groups:
                for idx in self.concept_group_map[group]:
                    if c[b, idx].item() != concepts[b, idx].item():
                        concept_pred[b, idx] = (
                            concepts[b, idx] * self.pos_logits[idx]
                            + (1 - concepts[b, idx]) * self.neg_logits[idx]
                        )
        return concept_pred

    def forward(self, x, concept_pred=None):
        if concept_pred is None:
            concept_pred = self.base(x)

        if self.hparams.cbm_mode == "fuzzy":
            concept = torch.sigmoid(concept_pred)
        elif self.hparams.cbm_mode == "bool":
            concept = torch.sigmoid(concept_pred).ge(0.5).float()
        elif self.hparams.cbm_mode == "hybrid":
            concept = concept_pred
        label_pred = self.classifier(concept)
        return label_pred, concept_pred

    def calc_loss(self, img, label, concepts):
        label_pred, concept_pred = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred,
            concepts,
            weight=(
                self.dm.imbalance_weights.to(self.device)
                if self.hparams.dataset == "CUB"
                else None
            ),
        )
        label_loss = F.cross_entropy(label_pred, label)
        loss = label_loss + self.hparams.concept_weight * concept_loss

        if self.hparams.mtl_mode != "normal":
            g = mtl([label_loss, concept_loss], self, self.hparams.mtl_mode)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return loss, (label_pred, concept_pred)

    def generate_adv(self, img, label, concepts):
        if self.atk_target == "combined":
            adv_img = self.combined_atk(self, img, (label, concepts))
        elif self.atk_target == "label":
            adv_img = self.label_atk(cls_wrapper(self, 0), img, label)
        elif self.atk_target == "concept":
            adv_img = self.concept_atk(cls_wrapper(self, 1), img, concepts)
        return adv_img

    def training_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.adv_mode == "adv":
            bs = img.shape[0] // 2
            adv_img = self.generate_adv(img[:bs], label[:bs], concepts[:bs])
            img = torch.cat([img[:bs], adv_img], dim=0)
            label = torch.cat([label[:bs], label[:bs]], dim=0)
            concepts = torch.cat([concepts[:bs], concepts[:bs]], dim=0)
        loss, _ = self.calc_loss(img, label, concepts)
        if self.hparams.mtl_mode != "normal":
            self.optimizers().step()
            self.optimizers().zero_grad()
        return loss

    def on_train_epoch_end(self):
        if self.hparams.mtl_mode != "normal" and self.global_rank == 0:
            sch = self.lr_schedulers()
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(self.trainer.callback_metrics["acc"])

    def validation_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.adv_mode == "adv":
            img = self.generate_adv(img, label, concepts)

        loss, o = self.calc_loss(img, label, concepts)
        self.losses(loss)
        label_pred, concept_pred = o[0], o[1]
        self.concept_acc(concept_pred, concepts)
        self.acc(label_pred, label)

    def on_validation_epoch_end(self):
        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
        )
        self.log("acc", self.acc.compute(), prog_bar=True)
        self.log("concept_acc", self.concept_acc.compute(), prog_bar=True)
        self.log("val_loss", self.losses.compute(), prog_bar=True)
        self.acc.reset()
        self.concept_acc.reset()
        self.losses.reset()

    def on_test_start(self):
        concept_logits = []
        for batch in self.dm.train_dataloader():
            outputs = self(batch[0].to(self.device))
            concept_logits.append(outputs[1].detach().cpu())
        c = torch.cat(concept_logits, dim=0).numpy()

        percentiles = np.percentile(c, [5, 95], axis=0)
        self.pos_logits = torch.from_numpy(percentiles[1]).to(self.device)
        self.neg_logits = torch.from_numpy(percentiles[0]).to(self.device)

    def test_step(self, batch, batch_idx):
        img, label, concepts = batch

        outputs = self(img)
        label_pred, concept_pred = outputs[0], outputs[1]
        self.acc(label_pred, label)
        self.concept_acc(concept_pred, concepts)
        self.log("test acc", self.acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test concept acc",
            self.concept_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
