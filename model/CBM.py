import math
import random
import numpy as np
import lightning as L
import wandb
from utils import initialize_weights, build_base, cls_wrapper, calc_spectral_norm
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
        attacker: str = "PGD",
        train_atk_args: dict = {},
        eval_atk_args: dict = {},
        mtl_mode: str = "normal",
        intervene_budget: int = 0,
        spectral_weight: float = 0,
        max_intervene_budget: int = 29,
        **kwargs,
    ):
        super().__init__()
        if mtl_mode != "normal":
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
        self.intervene_budget = intervene_budget

        self.concept_acc = Accuracy(
            task="multilabel", num_labels=min(num_concepts, real_concepts)
        )
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.adv_mode = adv_mode
        self.train_atk = getattr(attacks, attacker)(
            num_classes=num_classes,
            **train_atk_args,
        )
        self.eval_atk = getattr(attacks, attacker)(
            num_classes=num_classes,
            **eval_atk_args,
        )

        if adv_mode == "std":
            self.epses = [0, 0.001, 0.01, 0.1, 1.0]
        else:
            self.epses = list(range(5))

        self.robust_accs = nn.ModuleList(
            [
                Accuracy(task="multiclass", num_classes=num_classes)
                for _ in range(len(self.epses))
            ]
        )
        self.robust_concept_accs = nn.ModuleList(
            [
                Accuracy(task="multilabel", num_labels=min(num_concepts, real_concepts))
                for _ in range(len(self.epses))
            ]
        )
        self.intervene_clean_accs = nn.ModuleList(
            [
                Accuracy(task="multiclass", num_classes=num_classes)
                for _ in range(max_intervene_budget + 1)
            ]
        )
        self.intervene_clean_concept_accs = nn.ModuleList(
            [
                Accuracy(task="multilabel", num_labels=min(num_concepts, real_concepts))
                for _ in range(max_intervene_budget + 1)
            ]
        )
        self.intervene_robust_accs = nn.ModuleList(
            [
                Accuracy(task="multiclass", num_classes=num_classes)
                for _ in range(max_intervene_budget + 1)
            ]
        )
        self.intervene_robust_concept_accs = nn.ModuleList(
            [
                Accuracy(task="multilabel", num_labels=min(num_concepts, real_concepts))
                for _ in range(max_intervene_budget + 1)
            ]
        )

    def configure_optimizers(self):
        if self.hparams.spectral_weight == 0:
            optimizer = getattr(torch.optim, self.hparams.optimizer)(
                self.parameters(), lr=self.hparams.lr, **self.hparams.optimizer_args
            )
        else:
            optimizer = getattr(torch.optim, self.hparams.optimizer)(
                [
                    {
                        "params": self.base.parameters(),
                        **self.hparams.optimizer_args,
                    },
                    {
                        "params": self.classifier.parameters(),
                        "momentum": self.hparams.optimizer_args.get("momentum", 0.9),
                        "weight_decay": 0,
                    },
                ],
                lr=self.hparams.lr,
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

    def train_step(self, img, label, concepts):
        label_pred, concept_pred = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        label_loss = F.cross_entropy(label_pred, label)
        loss = label_loss + self.hparams.concept_weight * concept_loss

        if self.hparams.spectral_weight > 0:
            loss += calc_spectral_norm(self.classifier) * self.hparams.spectral_weight

        if self.hparams.mtl_mode != "normal":
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
        if self.hparams.mtl_mode != "normal":
            self.optimizers().step()
            self.optimizers().zero_grad()
        self.log(
            "loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def on_train_epoch_end(self):
        if self.hparams.mtl_mode != "normal" and self.global_rank == 0:
            sch = self.lr_schedulers()
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(self.trainer.callback_metrics["acc"])

    def validation_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.adv_mode == "adv":
            img = self.eval_atk(cls_wrapper(self), img, label)

        outputs = self(img)
        label_pred, concept_pred = outputs[0], outputs[1]
        if concept_pred.shape[1] > self.real_concepts:
            concept_pred = concept_pred[:, : self.real_concepts]
            concepts = concepts[:, : self.real_concepts]
        self.concept_acc(concept_pred, concepts)
        self.acc(label_pred, label)

    def on_validation_epoch_end(self):
        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
            sync_dist=True,
        )
        self.log("acc", self.acc.compute(), prog_bar=True)
        self.log("concept_acc", self.concept_acc.compute(), prog_bar=True)
        self.acc.reset()
        self.concept_acc.reset()

    def on_test_start(self):
        concept_logits = []
        for batch in self.dm.train_dataloader():
            outputs = self(batch[0].to(self.device))
            concept_logits.append(outputs[1].detach().cpu())
        c = torch.cat(concept_logits, dim=0).numpy()

        percentiles = np.percentile(c, [5, 95], axis=0)
        self.pos_logits = torch.from_numpy(percentiles[1]).to(self.device)
        self.neg_logits = torch.from_numpy(percentiles[0]).to(self.device)

        if hasattr(self.dm, "concept_group_map"):
            self.concept_group_map = self.dm.concept_group_map
        else:
            self.concept_group_map = {}
            for idx in range(self.num_concepts):
                self.concept_group_map[idx] = idx

        self.group_concept_map = {}
        for name, idxs in self.concept_group_map.items():
            for idx in idxs:
                self.group_concept_map[idx] = name

    def test_step(self, batch, batch_idx):
        img, label, concepts = batch

        for i, eps in enumerate(self.epses):
            if eps == 0:
                x = img
            else:
                atk_args = self.hparams.eval_atk_args
                atk_args["eps"] = eps / 255
                atk = getattr(attacks, self.hparams.attacker)(**atk_args)
                x = atk(cls_wrapper(self), img, label)
            outputs = self(x)
            label_pred, concept_pred = outputs[0], outputs[1]
            self.robust_accs[i](label_pred, label)
            self.robust_concept_accs[i](concept_pred, concepts)

        if self.hparams.model == "backbone":
            return

        outputs = self(img)
        label_pred, concept_pred = outputs[0], outputs[1]
        for i, intervene_budget in enumerate(
            range(self.hparams.max_intervene_budget + 1)
        ):
            cur_concept_pred = self.intervene(
                concepts, concept_pred.clone(), intervene_budget
            )
            cur_label_pred = self(img, cur_concept_pred)[0]
            self.intervene_clean_accs[i](cur_label_pred, label)
            self.intervene_clean_concept_accs[i](cur_concept_pred, concepts)

        x = self.eval_atk(cls_wrapper(self), img, label)
        outputs = self(x)
        label_pred, concept_pred = outputs[0], outputs[1]
        for i, intervene_budget in enumerate(
            range(self.hparams.max_intervene_budget + 1)
        ):
            cur_concept_pred = self.intervene(
                concepts, concept_pred.clone(), intervene_budget
            )
            cur_label_pred = self(x, cur_concept_pred)[0]
            self.intervene_robust_accs[i](cur_label_pred, label)
            self.intervene_robust_concept_accs[i](cur_concept_pred, concepts)

    def on_test_epoch_end(self):
        for i, eps in enumerate(self.epses):
            acc = self.robust_accs[i].compute()
            concept_acc = self.robust_concept_accs[i].compute()
            self.robust_accs[i].reset()
            self.robust_concept_accs[i].reset()
            if eps == 0:
                clean_acc = acc
                clean_concept_acc = concept_acc
            wandb.log(
                {
                    "Test Acc": acc,
                    "Test Concept Acc": concept_acc,
                    "Test ASR": (clean_acc - acc) / clean_acc,
                    "Test Concept ASR": (clean_concept_acc - concept_acc)
                    / clean_concept_acc,
                    "eps": i,
                }
            )

        if self.hparams.model == "backbone":
            return

        for i, intervene_budget in enumerate(
            range(self.hparams.max_intervene_budget + 1)
        ):
            acc = self.intervene_clean_accs[i].compute()
            concept_acc = self.intervene_clean_concept_accs[i].compute()
            self.intervene_clean_accs[i].reset()
            self.intervene_clean_concept_accs[i].reset()
            wandb.log(
                {
                    "Clean Acc under Intervene": acc,
                    "Clean Concept Acc under Intervene": concept_acc,
                    "Intervene Budget": intervene_budget,
                }
            )

        for i, intervene_budget in enumerate(
            range(self.hparams.max_intervene_budget + 1)
        ):
            acc = self.intervene_robust_accs[i].compute()
            concept_acc = self.intervene_robust_concept_accs[i].compute()
            self.intervene_robust_accs[i].reset()
            self.intervene_robust_concept_accs[i].reset()
            wandb.log(
                {
                    "Robust Acc under Intervene": acc,
                    "Robust Concept Acc under Intervene": concept_acc,
                    "Adv Intervene Budget": intervene_budget,
                }
            )
