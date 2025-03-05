import random
import numpy as np
import lightning as L
from utils import initialize_weights, build_base, cls_wrapper
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import attacks
from mtl import mtl


class CBM(L.LightningModule):
    def __init__(
        self,
        dm: L.LightningDataModule,
        base: str = "resnet50",
        use_pretrained: bool = True,
        concept_weight: float = 1,
        optimizer: str = "SGD",
        lr: float = 0.1,
        optimizer_args: dict = {},
        scheduler: str = "ReduceLROnPlateau",
        scheduler_args: dict = {},
        adv_mode: str = "std",
        hidden_dim: int = 0,
        cbm_mode: str = "hybrid",
        attacker: str = "pgd",
        train_atk_args: dict = {},
        eval_atk_args: dict = {},
        mtl_mode: str = "normal",
        intervene_budget: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore="dm")
        num_classes = dm.num_classes
        num_concepts = dm.num_concepts
        real_concepts = dm.real_concepts
        self.dm = dm
        self.base = build_base(base, num_concepts, use_pretrained)

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

        self.concept_acc = Accuracy(
            task="multilabel", num_labels=min(num_concepts, real_concepts)
        )
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.acc5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.acc10 = Accuracy(task="multiclass", num_classes=num_classes, top_k=10)

        self.adv_mode = adv_mode
        self.train_atk = getattr(attacks, attacker)(
            num_classes=num_classes,
            **train_atk_args,
        )
        self.eval_atk = getattr(attacks, attacker)(
            num_classes=num_classes,
            **eval_atk_args,
        )

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), lr=self.hparams.lr, **self.hparams.optimizer_args
        )
        scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler)(
            optimizer, **self.hparams.scheduler_args
        )
        if self.hparams.scheduler == "ReduceLROnPlateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "acc",
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                },
            }
        else:
            return [optimizer], [scheduler]

    def intervene(
        self, concepts, concept_pred, intervene_budget, concept_group_map=None
    ):
        group_concept_map = {}
        for name, idxs in concept_group_map.items():
            for idx in idxs:
                group_concept_map[idx] = name
        c = torch.sigmoid(concept_pred).ge(0.5).long()
        for b in range(concepts.shape[0]):
            s = set()
            for idx in range(self.num_concepts):
                if c[b, idx].item() != concepts[b, idx].item():
                    s.add(group_concept_map[idx])
            intervene_groups = list(s)
            if len(intervene_groups) > intervene_budget:
                intervene_groups = random.sample(intervene_groups, intervene_budget)
            for group in intervene_groups:
                for idx in concept_group_map[group]:
                    if c[b, idx].item() != concepts[b, idx].item():
                        concept_pred[b, idx] = (
                            concepts[b, idx] * self.pos_logits[idx]
                            + (1 - concepts[b, idx]) * self.neg_logits[idx]
                        )
        return concept_pred

    def forward(self, x, concepts=None):
        concept_pred = self.base(x)
        # human intervene
        if concepts is not None:
            concept_pred = self.intervene(
                concepts,
                concept_pred,
                self.hparams.intervene_budget,
                self.concept_group_map,
            )

        if self.hparams.cbm_mode == "fuzzy":
            concept = torch.sigmoid(concept_pred)
        elif self.hparams.cbm_mode == "bool":
            concept = torch.sigmoid(concept_pred).ge(0.5).float()
        elif self.hparams.cbm_mode == "hybrid":
            concept = concept_pred
        label_pred = self.classifier(concept)
        return label_pred, concept_pred

    def train_step(self, img, label, concepts):
        label_pred, concept_pred = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        label_loss = F.cross_entropy(label_pred, label)
        loss = label_loss + self.hparams.concept_weight * concept_loss
        if self.hparams.mtl_mode == "normal":
            self.manual_backward(loss)
        else:
            g = mtl([label_loss, concept_loss], self, self.hparams.mtl_mode)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return loss

    def training_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.adv_mode == "adv":
            bs = img.shape[0] // 2
            adv_img = self.train_atk(cls_wrapper(self), img[:bs], label[:bs])
            img = torch.cat([img[:bs], adv_img], dim=0)
            label = torch.cat([label[:bs], label[:bs]], dim=0)
            concepts = torch.cat([concepts[:bs], concepts[:bs]], dim=0)
        loss = self.train_step(img, label, concepts)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizers().step()
        self.optimizers().zero_grad()
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["acc"])

    def eval_step(self, batch):
        img, label, concepts = batch
        if self.adv_mode == "adv":
            img = self.eval_atk(cls_wrapper(self), img, label)
        if self.hparams.intervene_budget > 0:
            outputs = self(img, concepts)
        else:
            outputs = self(img)
        label_pred, concept_pred = outputs[0], outputs[1]
        if concept_pred.shape[1] > self.real_concepts:
            concept_pred = concept_pred[:, : self.real_concepts]
            concepts = concepts[:, : self.real_concepts]
        self.concept_acc(concept_pred, concepts)
        self.acc(label_pred, label)
        self.acc5(label_pred, label)
        self.acc10(label_pred, label)
        self.log(
            "concept_acc", self.concept_acc, on_epoch=True, on_step=False, prog_bar=True
        )
        self.log("acc", self.acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("acc5", self.acc5, on_epoch=True, on_step=False)
        self.log("acc10", self.acc10, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch)
        self.log(
            "lr", self.optimizers().param_groups[0]["lr"], on_step=False, on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        self.eval_step(batch)

    def on_test_start(self):
        concept_logits = []
        for batch in self.dm.train_dataloader():
            outputs = self(batch[0].to(self.device))
            concept_logits.append(outputs[1])
        c = torch.cat(concept_logits, dim=0).detach().cpu().numpy()

        pos_logits = torch.ones(self.num_concepts).to(self.device)
        neg_logits = torch.zeros(self.num_concepts).to(self.device)
        for idx in range(self.num_concepts):
            pos_logits[idx] = torch.tensor(np.percentile(c[:, idx], 95))
            neg_logits[idx] = torch.tensor(np.percentile(c[:, idx], 5))
        self.pos_logits = pos_logits
        self.neg_logits = neg_logits
        if hasattr(self.dm, "concept_group_map"):
            self.concept_group_map = self.dm.concept_group_map
        else:
            self.concept_group_map = {}
            for idx in range(self.num_concepts):
                self.concept_group_map[idx] = idx
