import random
import numpy as np
import lightning as L
import wandb
from utils import initialize_weights, build_base, cls_wrapper, suppress_stdout
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from attacks import PGD
from torchattacks import CW
from autoattack import AutoAttack
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
        lr: float = 0.1,
        optimizer_args: dict = {},
        scheduler: str = "ReduceLROnPlateau",
        scheduler_args: dict = {},
        hidden_dim: int = 0,
        res_dim: int = 0,
        cbm_mode: str = "hybrid",
        train_mode: str = "std",
        label_atk_args: dict = {},
        concept_atk_args: dict = {},
        combined_atk_args: dict = {},
        auto_atk_args: dict = {"eps": 4 / 255},
        cw_atk_args: dict = {},
        mtl_mode: str = "normal",
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
        self.real_concepts = real_concepts
        self.num_classes = num_classes
        self.num_concepts = num_concepts

        self.concept_acc = Accuracy(
            task="multilabel", num_labels=min(num_concepts, real_concepts)
        )
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_mode = train_mode
        self.label_atk = PGD(
            loss_fn=F.cross_entropy,
            **label_atk_args,
        )
        self.concept_atk = PGD(
            loss_fn=F.binary_cross_entropy_with_logits,
            **concept_atk_args,
        )
        self.combined_atk = PGD(
            loss_fn=lambda o, y: F.cross_entropy(o[0], y[0])
            + F.binary_cross_entropy_with_logits(o[1], y[1]),
            **combined_atk_args,
        )

        for s in ["label", "concept", "combined", "auto", "cw"]:
            setattr(
                self,
                f"{s}_atk_acc",
                Accuracy(task="multiclass", num_classes=num_classes),
            )
            setattr(
                self,
                f"{s}_atk_concept_acc",
                Accuracy(
                    task="multilabel",
                    num_labels=min(num_concepts, real_concepts),
                ),
            )
            setattr(
                self,
                f"intervene_{s}_atk_accs",
                nn.ModuleList(
                    [
                        Accuracy(task="multiclass", num_classes=num_classes)
                        for _ in range(11)
                    ]
                ),
            )
            setattr(
                self,
                f"intervene_{s}_atk_concept_accs",
                nn.ModuleList(
                    [
                        Accuracy(
                            task="multilabel",
                            num_labels=min(num_concepts, real_concepts),
                        )
                        for _ in range(11)
                    ]
                ),
            )

        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_concept_acc = Accuracy(
            task="multilabel", num_labels=min(num_concepts, real_concepts)
        )
        self.intervene_clean_accs = nn.ModuleList(
            [Accuracy(task="multiclass", num_classes=num_classes) for _ in range(11)]
        )
        self.intervene_clean_concept_accs = nn.ModuleList(
            [
                Accuracy(task="multilabel", num_labels=min(num_concepts, real_concepts))
                for _ in range(11)
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
                    "monitor": "fit/acc",
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
        logits = self.base(x)
        if concept_pred is None:
            concept_pred = logits
        else:
            logits[:, : self.num_concepts] = concept_pred
            concept_pred = logits
        if self.hparams.cbm_mode == "fuzzy":
            concept = torch.sigmoid(concept_pred)
        elif self.hparams.cbm_mode == "bool":
            concept_prob = torch.sigmoid(concept_pred)
            concept_binary = concept_prob.ge(0.5).float()
            concept = concept_prob + (concept_binary - concept_prob).detach()
        elif self.hparams.cbm_mode == "hybrid":
            concept = concept_pred
        label_pred = self.classifier(concept)
        return label_pred, concept_pred[:, : self.num_concepts]

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

        losses = {
            "label loss": label_loss,
            "concept loss": concept_loss,
            "loss": loss,
        }

        if self.hparams.mtl_mode != "normal":
            g = mtl([label_loss, concept_loss], self, self.hparams.mtl_mode)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return losses, (label_pred, concept_pred)

    @torch.enable_grad()
    @suppress_stdout
    def generate_adv(self, img, label, concepts, atk):
        if atk == "combined":
            adv_img = self.combined_atk(self, img, (label, concepts))
        elif atk == "label":
            adv_img = self.label_atk(cls_wrapper(self, 0), img, label)
        elif atk == "concept":
            adv_img = self.concept_atk(cls_wrapper(self, 1), img, concepts)
        elif atk == "auto":
            adv_img = self.auto_atk.run_standard_evaluation(img, label, bs=img.shape[0])
        elif atk == "cw":
            adv_img = self.cw_atk(img, label)
        else:
            raise NotImplementedError
        return adv_img

    def training_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.train_mode != "std":
            bs = img.shape[0] // 2
            adv_img = self.generate_adv(
                img[:bs], label[:bs], concepts[:bs], self.train_mode
            )
            img = torch.cat([img[:bs], adv_img], dim=0)
            label = torch.cat([label[:bs], label[:bs]], dim=0)
            concepts = torch.cat([concepts[:bs], concepts[:bs]], dim=0)
        losses, _ = self.calc_loss(img, label, concepts)
        loss = losses["loss"]
        if self.hparams.mtl_mode != "normal":
            self.optimizers().step()
            self.optimizers().zero_grad()
        return loss

    def on_train_epoch_end(self):
        if self.hparams.mtl_mode != "normal" and self.global_rank == 0:
            sch = self.lr_schedulers()
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(self.trainer.callback_metrics["fit/acc"])

    def validation_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.train_mode != "std":
            img = self.generate_adv(img, label, concepts, "label")

        losses, o = self.calc_loss(img, label, concepts)
        label_pred, concept_pred = o[0], o[1]
        if concept_pred.shape[1] > self.real_concepts:
            concept_pred = concept_pred[:, : self.real_concepts]
            concepts = concepts[:, : self.real_concepts]
        self.concept_acc(concept_pred, concepts)
        self.acc(label_pred, label)
        for name, val in losses.items():
            self.log(f"fit/{name}", val, on_step=False, on_epoch=True)
        self.log(
            "fit/lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
        )
        self.log("fit/acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log(
            "fit/concept_acc",
            self.concept_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    def on_test_start(self):
        concept_logits = []
        for batch in self.dm.train_dataloader():
            outputs = self(batch[0].to(self.device))
            concept_logits.append(outputs[1].detach().cpu())
        c = torch.cat(concept_logits, dim=0).numpy()

        percentiles = np.percentile(c, [5, 95], axis=0)
        self.pos_logits = torch.from_numpy(percentiles[1]).to(self.device)
        self.neg_logits = torch.from_numpy(percentiles[0]).to(self.device)
        self.cw_atk = CW(cls_wrapper(self, 0), **self.hparams.cw_atk_args)
        self.auto_atk = AutoAttack(
            cls_wrapper(self, 0),
            verbose=False,
            **self.hparams.auto_atk_args,
        )

    def test_step(self, batch, batch_idx):
        img, label, concepts = batch

        o = self(img)
        label_pred, concept_pred = o[0], o[1]
        self.test_acc(label_pred, label)
        self.test_concept_acc(concept_pred, concepts)

        label_loss = F.cross_entropy(label_pred, label)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred,
            concepts,
            weight=(
                self.dm.imbalance_weights.to(self.device)
                if self.hparams.dataset == "CUB"
                else None
            ),
        )
        loss = label_loss + self.hparams.concept_weight * concept_loss
        self.log("Clean Loss", loss, on_step=False, on_epoch=True)
        self.log("Clean Label Loss", label_loss, on_step=False, on_epoch=True)
        self.log("Clean Concept Loss", concept_loss, on_step=False, on_epoch=True)

        if self.hparams.model != "backbone":
            for i in range(11):
                intervene_budget = self.max_intervene_budget * i // 10
                cur_concept_pred = self.intervene(
                    concepts, concept_pred.clone(), intervene_budget
                )
                cur_label_pred = self(img, cur_concept_pred)[0]
                self.intervene_clean_accs[i](cur_label_pred, label)
                self.intervene_clean_concept_accs[i](cur_concept_pred, concepts)

        for atk_name in ["label", "concept", "combined", "cw", "auto"]:
            if self.hparams.model == "backbone" and (
                atk_name == "concept" or atk_name == "combined"
            ):
                continue
            x = self.generate_adv(img, label, concepts, atk_name)
            outputs = self(x)
            label_pred, concept_pred = outputs[0], outputs[1]
            getattr(self, f"{atk_name}_atk_acc")(label_pred, label)
            getattr(self, f"{atk_name}_atk_concept_acc")(concept_pred, concepts)

            label_loss = F.cross_entropy(label_pred, label)
            concept_loss = F.binary_cross_entropy_with_logits(
                concept_pred,
                concepts,
                weight=(
                    self.dm.imbalance_weights.to(self.device)
                    if self.hparams.dataset == "CUB"
                    else None
                ),
            )
            loss = label_loss + self.hparams.concept_weight * concept_loss
            self.log(f"Loss under {atk_name} atk", loss, on_step=False, on_epoch=True)
            self.log(
                f"Label Loss under {atk_name} atk",
                label_loss,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"Concept Loss under {atk_name} atk",
                concept_loss,
                on_step=False,
                on_epoch=True,
            )

            if self.hparams.model == "backbone":
                continue

            for i in range(11):
                intervene_budget = self.max_intervene_budget * i // 10
                cur_concept_pred = self.intervene(
                    concepts, concept_pred.clone(), intervene_budget
                )
                cur_label_pred = self(x, cur_concept_pred)[0]
                getattr(self, f"intervene_{atk_name}_atk_accs")[i](
                    cur_label_pred, label
                )
                getattr(self, f"intervene_{atk_name}_atk_concept_accs")[i](
                    cur_concept_pred, concepts
                )
            del x
            torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        clean_acc = self.test_acc.compute()
        clean_concept_acc = self.test_concept_acc.compute()
        self.test_acc.reset()
        self.test_concept_acc.reset()
        self.log("Clean Acc", clean_acc)
        self.log("Clean Concept Acc", clean_concept_acc)
        for i in range(11):
            acc = self.intervene_clean_accs[i].compute()
            concept_acc = self.intervene_clean_concept_accs[i].compute()
            self.intervene_clean_accs[i].reset()
            self.intervene_clean_concept_accs[i].reset()
            wandb.log(
                {
                    "Clean Acc with Intervene": acc,
                    "Clean Concept Acc with Intervene": concept_acc,
                    "Intervene Budget": i,
                }
            )

        for atk_name in ["label", "concept", "combined", "cw", "auto"]:
            acc = getattr(self, f"{atk_name}_atk_acc").compute()
            concept_acc = getattr(self, f"{atk_name}_atk_concept_acc").compute()
            getattr(self, f"{atk_name}_atk_acc").reset()
            getattr(self, f"{atk_name}_atk_concept_acc").reset()
            self.log(f"{atk_name} Attack Acc", acc)
            self.log(f"{atk_name} Attack Concept Acc", concept_acc)
            self.log(f"{atk_name} Attack ASR", (clean_acc - acc) / clean_acc)
            self.log(
                f"{atk_name} Attack Concept ASR",
                (clean_concept_acc - concept_acc) / clean_concept_acc,
            )

            if self.hparams.model == "backbone":
                continue

            for i in range(11):
                acc = getattr(self, f"intervene_{atk_name}_atk_accs")[i].compute()
                concept_acc = getattr(self, f"intervene_{atk_name}_atk_concept_accs")[
                    i
                ].compute()
                getattr(self, f"intervene_{atk_name}_atk_accs")[i].reset()
                getattr(self, f"intervene_{atk_name}_atk_concept_accs")[i].reset()
                wandb.log(
                    {
                        f"{atk_name} Attack Acc with Intervene": acc,
                        f"{atk_name} Attack Concept Acc with Intervene": concept_acc,
                        "Intervene Budget": i,
                    }
                )
