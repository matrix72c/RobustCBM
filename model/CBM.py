import os
import random
import numpy as np
import lightning as L
import pandas as pd
from utils import initialize_weights, build_base, cls_wrapper, suppress_stdout
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from attacks import PGD
from autoattack import AutoAttack
from mtl import mtl
from collections import defaultdict


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
        base: str,
        use_pretrained: bool,
        concept_weight: float,
        optimizer: str,
        optimizer_args: dict,
        scheduler: str,
        scheduler_args: dict,
        plateau_args: dict,
        hidden_dim: int,
        res_dim: int,
        cbm_mode: str,
        mtl_mode: str,
        weighted_bce: bool,
        add_residual: bool,
        res_alpha: float,
        ignore_intervenes: bool,
        train_mode: str,
        lpgd_args: dict,
        cpgd_args: dict,
        jpgd_args: dict,
        aa_args: dict,
        cw_args: dict,
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
        self.base = build_base(base, num_concepts + res_dim, use_pretrained)

        if hidden_dim > 0:
            self.classifier = MLP(num_concepts + res_dim, hidden_dim, num_classes)
        else:
            self.classifier = nn.Linear(num_concepts + res_dim, num_classes).apply(
                initialize_weights
            )
        self.num_classes = num_classes
        self.num_concepts = num_concepts

        if add_residual:
            self.res_alpha = nn.Parameter(torch.logit(torch.tensor(res_alpha)))

        self.concept_acc = Accuracy(task="multilabel", num_labels=num_concepts)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_mode = train_mode
        self.lpgd = PGD(
            loss_fn=F.cross_entropy,
            **lpgd_args,
        )
        self.cpgd = PGD(
            loss_fn=F.binary_cross_entropy_with_logits,
            **cpgd_args,
        )
        self.jpgd = PGD(
            loss_fn=lambda o, y: F.cross_entropy(o[0], y[0])
            + F.binary_cross_entropy_with_logits(o[1], y[1]),
            **jpgd_args,
        )

        for s in ["Std", "LPGD", "CPGD", "JPGD", "AA"]:
            setattr(
                self,
                f"{s}_acc",
                Accuracy(task="multiclass", num_classes=num_classes),
            )
            setattr(
                self,
                f"{s}_concept_acc",
                Accuracy(task="multilabel", num_labels=num_concepts),
            )
            setattr(
                self,
                f"intervene_{s}_accs",
                nn.ModuleList(
                    [
                        Accuracy(task="multiclass", num_classes=num_classes)
                        for _ in range(11)
                    ]
                ),
            )
            setattr(
                self,
                f"intervene_{s}_concept_accs",
                nn.ModuleList(
                    [
                        Accuracy(
                            task="multilabel",
                            num_labels=num_concepts,
                        )
                        for _ in range(11)
                    ]
                ),
            )
        self.losses = defaultdict(list)

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
                "lr_scheduler": {"scheduler": scheduler, **self.hparams.plateau_args},
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
        elif self.hparams.res_dim > 0:
            l = self.base(x)
            l[:, : self.num_concepts] = concept_pred
            concept_pred = l

        if self.hparams.cbm_mode == "fuzzy":
            concept = torch.sigmoid(concept_pred)
        elif self.hparams.cbm_mode == "relu":
            concept = torch.relu(concept_pred)
        elif self.hparams.cbm_mode == "bool":
            concept_prob = torch.sigmoid(concept_pred)
            concept_binary = concept_prob.ge(0.5).float()
            concept = concept_prob + (concept_binary - concept_prob).detach()
        elif self.hparams.cbm_mode == "hybrid":
            concept = concept_pred
        label_pred = self.classifier(concept)
        if self.hparams.res_dim == self.num_classes and self.hparams.add_residual:
            label_pred += concept_pred[:, self.num_concepts :] * torch.sigmoid(self.res_alpha)
        return label_pred, concept_pred[:, : self.num_concepts]

    def calc_loss(self, img, label, concepts):
        label_pred, concept_pred = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred,
            concepts,
            weight=(
                self.dm.imbalance_weights.to(self.device)
                if self.hparams.weighted_bce
                else None
            ),
        )
        label_loss = F.cross_entropy(label_pred, label)
        loss = label_loss + self.hparams.concept_weight * concept_loss
        losses = {
            "Label Loss": label_loss,
            "Concept Loss": concept_loss,
            "Loss": loss,
        }

        if self.hparams.mtl_mode != "normal":
            g = mtl([label_loss, concept_loss], self, self.hparams.mtl_mode)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return losses, (label_pred, concept_pred)

    @suppress_stdout
    @torch.enable_grad()
    def generate_adv(self, img, label, concepts, atk):
        if atk == "JPGD":
            adv_img = self.jpgd(self, img, (label, concepts))
        elif atk == "LPGD":
            adv_img = self.lpgd(cls_wrapper(self, 0), img, label)
        elif atk == "CPGD":
            adv_img = self.cpgd(cls_wrapper(self, 1), img, concepts)
        elif atk == "AA":
            adv_img = self.aa.run_standard_evaluation(img, label, bs=img.shape[0])
        elif atk == "Std":
            adv_img = img
        else:
            raise NotImplementedError
        return adv_img

    def training_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.train_mode != "Std":
            bs = img.shape[0] // 2
            adv_img = self.generate_adv(
                img[:bs], label[:bs], concepts[:bs], self.train_mode
            )
            img = torch.cat([img[:bs], adv_img], dim=0)
            label = torch.cat([label[:bs], label[:bs]], dim=0)
            concepts = torch.cat([concepts[:bs], concepts[:bs]], dim=0)
        losses, _ = self.calc_loss(img, label, concepts)
        loss = losses["Loss"]
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
        if self.train_mode != "Std":
            img = self.generate_adv(img, label, concepts, self.train_mode)

        losses, o = self.calc_loss(img, label, concepts)
        label_pred, concept_pred = o[0], o[1]
        self.concept_acc(concept_pred, concepts)
        self.acc(label_pred, label)
        for name, val in losses.items():
            self.log(f"{name}", val, on_step=False, on_epoch=True)

        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
        )
        self.log("acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("concept_acc", self.concept_acc, prog_bar=True, on_epoch=True, on_step=False)

    def on_test_start(self):
        self.aa = AutoAttack(
            cls_wrapper(self, 0),
            verbose=False,
            **self.hparams.aa_args,
        )
        if self.hparams.ignore_intervenes:
            return
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

        for mode in ["Std", "LPGD", "CPGD", "JPGD", "AA"]:
            if self.hparams.model == "backbone" and (mode == "CPGD" or mode == "JPGD"):
                continue
            if mode == "Std":
                x = img
            else:
                x = self.generate_adv(img, label, concepts, mode)
            losses, outputs = self.calc_loss(x, label, concepts)
            label_pred, concept_pred = outputs[0], outputs[1]
            getattr(self, f"{mode}_acc")(label_pred, label)
            getattr(self, f"{mode}_concept_acc")(concept_pred, concepts)
            for name, val in losses.items():
                self.losses[f"{mode} {name}"].append(val)

            if self.hparams.model == "backbone" or self.hparams.ignore_intervenes:
                continue

            for i in range(11):
                intervene_budget = self.max_intervene_budget * i // 10
                cur_concept_pred = self.intervene(
                    concepts, concept_pred.clone(), intervene_budget
                )
                cur_label_pred = self(x, cur_concept_pred)[0]
                getattr(self, f"intervene_{mode}_accs")[i](cur_label_pred, label)
                getattr(self, f"intervene_{mode}_concept_accs")[i](
                    cur_concept_pred, concepts
                )

    def on_test_epoch_end(self):
        res = {}
        res["name"] = self.hparams.run_name
        for mode in ["Std", "LPGD", "CPGD", "JPGD", "AA"]:
            res[f"{mode} Acc"] = getattr(self, f"{mode}_acc").compute().item()
            res[f"{mode} Concept Acc"] = getattr(self, f"{mode}_concept_acc").compute().item()
            for name, val in self.losses.items():
                res[f"{name}"] = torch.tensor(val).mean().item()

            if self.hparams.model == "backbone" or self.hparams.ignore_intervenes:
                continue

            int_acc = []
            int_concept_acc = []
            for i in range(11):
                acc = getattr(self, f"intervene_{mode}_accs")[i].compute()
                concept_acc = getattr(self, f"intervene_{mode}_concept_accs")[
                    i
                ].compute()
                getattr(self, f"intervene_{mode}_accs")[i].reset()
                getattr(self, f"intervene_{mode}_concept_accs")[i].reset()
                self.log(f"{mode} Acc with {i}0% Intervene", acc)
                self.log(f"{mode} Concept Acc with {i}0% Intervene", concept_acc)
                int_acc.append(acc.item())
                int_concept_acc.append(concept_acc.item())
            res[f"{mode} Acc with Intervene"] = int_acc
            res[f"{mode} Concept Acc with Intervene"] = int_concept_acc

        for k, v in res.items():
            if isinstance(v, list) or k ==  "name":
                continue
            self.log(k, v)

        row = pd.DataFrame([res], columns=res.keys())
        if os.path.exists("result.csv"):
            df = pd.read_csv("result.csv")
            df = pd.concat([df, row], ignore_index=True)
        else:
            df = row
        df.to_csv("result.csv", index=False)
