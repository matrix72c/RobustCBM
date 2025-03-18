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
        atk_target: str = "label",
        label_atk_args: dict = {},
        concept_atk_args: dict = {},
        combined_atk_args: dict = {},
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
        self.losses = MeanMetric()

        self.adv_mode = adv_mode
        self.atk_target = atk_target
        self.label_atk = getattr(attacks, attacker)(
            num_classes=num_classes,
            loss_fn=F.cross_entropy,
            **label_atk_args,
        )
        self.concept_atk = getattr(attacks, attacker)(
            num_classes=num_concepts,
            loss_fn=F.binary_cross_entropy_with_logits,
            **concept_atk_args,
        )
        self.combined_atk = getattr(attacks, attacker)(
            num_classes=num_classes,
            loss_fn=lambda o, y: F.cross_entropy(o[0], y[0])
            + F.binary_cross_entropy_with_logits(o[1], y[1]),
            **combined_atk_args,
        )

        if adv_mode == "std":
            self.epses = [0, 0.001, 0.01, 0.1, 1.0]
        else:
            self.epses = list(range(5))

        for s in ["label", "concept", "combined"]:
            setattr(
                self,
                f"{s}_atk_accs",
                nn.ModuleList(
                    [
                        Accuracy(task="multiclass", num_classes=num_classes)
                        for _ in range(len(self.epses))
                    ]
                ),
            )
            setattr(
                self,
                f"{s}_atk_concept_accs",
                nn.ModuleList(
                    [
                        Accuracy(
                            task="multilabel",
                            num_labels=min(num_concepts, real_concepts),
                        )
                        for _ in range(len(self.epses))
                    ]
                ),
            )
            setattr(
                self,
                f"{s}_atk_intervene_accs",
                nn.ModuleList(
                    [
                        Accuracy(task="multiclass", num_classes=num_classes)
                        for _ in range(max_intervene_budget + 1)
                    ]
                ),
            )
            setattr(
                self,
                f"{s}_atk_intervene_concept_accs",
                nn.ModuleList(
                    [
                        Accuracy(
                            task="multilabel",
                            num_labels=min(num_concepts, real_concepts),
                        )
                        for _ in range(max_intervene_budget + 1)
                    ]
                ),
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
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        label_loss = F.cross_entropy(label_pred, label)
        loss = label_loss + self.hparams.concept_weight * concept_loss

        if self.adv_mode == "adv" and self.hparams.trades > 0:
            clean_concept, adv_concept = torch.chunk(concept_pred, 2, dim=1)
            trades_loss = (
                F.kl_div(
                    F.log_softmax(clean_concept, dim=1),
                    F.softmax(adv_concept, dim=1),
                    reduction="batchmean",
                )
                * self.hparams.trades
            )
            loss += trades_loss

        if self.hparams.spectral_weight > 0:
            spectral_loss = calc_spectral_norm(self.classifier) * self.hparams.spectral_weight
            loss += spectral_loss
            self.log("spectral_loss", spectral_loss)

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
        if concept_pred.shape[1] > self.real_concepts:
            concept_pred = concept_pred[:, : self.real_concepts]
            concepts = concepts[:, : self.real_concepts]
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

        for atk_name in ["label", "concept", "combined"]:
            if self.hparams.model == "backbone" and atk_name != "label":
                continue
            atk = getattr(self, f"{atk_name}_atk")
            original_eps = getattr(atk, "eps", 0)
            self.atk_target = atk_name
            for i, eps in enumerate(self.epses):
                if eps == 0:
                    x = img
                else:
                    atk.eps = eps / 255
                    x = self.generate_adv(img, label, concepts)
                outputs = self(x)
                label_pred, concept_pred = outputs[0], outputs[1]
                getattr(self, f"{atk_name}_atk_accs")[i](label_pred, label)
                getattr(self, f"{atk_name}_atk_concept_accs")[i](concept_pred, concepts)
            atk.eps = original_eps

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

        for atk_name in ["label", "concept", "combined"]:
            atk = getattr(self, f"{atk_name}_atk")
            original_eps = getattr(atk, "eps", 0)
            self.atk_target = atk_name
            x = self.generate_adv(img, label, concepts)
            outputs = self(x)
            label_pred, concept_pred = outputs[0], outputs[1]
            for i, intervene_budget in enumerate(
                range(self.hparams.max_intervene_budget + 1)
            ):
                cur_concept_pred = self.intervene(
                    concepts, concept_pred.clone(), intervene_budget
                )
                cur_label_pred = self(x, cur_concept_pred)[0]
                getattr(self, f"{atk_name}_atk_intervene_accs")[i](
                    cur_label_pred, label
                )
                getattr(self, f"{atk_name}_atk_intervene_concept_accs")[i](
                    cur_concept_pred, concepts
                )

    def on_test_epoch_end(self):
        for atk_name in ["label", "concept", "combined"]:
            for i, eps in enumerate(self.epses):
                acc = getattr(self, f"{atk_name}_atk_accs")[i].compute()
                concept_acc = getattr(self, f"{atk_name}_atk_concept_accs")[i].compute()
                getattr(self, f"{atk_name}_atk_accs")[i].reset()
                getattr(self, f"{atk_name}_atk_concept_accs")[i].reset()
                if eps == 0:
                    clean_acc = acc
                    clean_concept_acc = concept_acc
                wandb.log(
                    {
                        f"{atk_name} Attack Acc": acc,
                        f"{atk_name} Attack Concept Acc": concept_acc,
                        f"{atk_name} Attack ASR": (clean_acc - acc) / clean_acc,
                        f"{atk_name} Attack Concept ASR": (
                            clean_concept_acc - concept_acc
                        )
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

        for atk_name in ["label", "concept", "combined"]:
            for i, intervene_budget in enumerate(
                range(self.hparams.max_intervene_budget + 1)
            ):
                acc = getattr(self, f"{atk_name}_atk_intervene_accs")[i].compute()
                concept_acc = getattr(self, f"{atk_name}_atk_intervene_concept_accs")[
                    i
                ].compute()
                getattr(self, f"{atk_name}_atk_intervene_accs")[i].reset()
                getattr(self, f"{atk_name}_atk_intervene_concept_accs")[i].reset()
                wandb.log(
                    {
                        f"{atk_name} Attack Acc under Intervene": acc,
                        f"{atk_name} Attack Concept Acc under Intervene": concept_acc,
                        f"Adv Intervene Budget": intervene_budget,
                    }
                )
