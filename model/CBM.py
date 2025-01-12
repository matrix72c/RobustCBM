import lightning as L
from utils import batchnorm_no_update_context, initialize_weights
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import attacks
from mtl import get_grad, gradient_normalize, gradient_ordered


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
        adv_mode: bool = False,
        hidden_dim: int = 0,
        cbm_mode: str = "hybrid",
        attacker: str = "PGD",
        attacker_args: dict = {},
        mtl_mode: str = "normal",
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore="dm")
        num_classes = dm.num_classes
        num_concepts = dm.num_concepts
        real_concepts = dm.real_concepts
        self.dm = dm
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
                    torchvision.models.Inception_V3_Weights.DEFAULT
                    if use_pretrained
                    else None
                ),
            )
            self.base.aux_logits = False
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

        self.concept_acc = Accuracy(
            task="multilabel", num_labels=min(num_concepts, real_concepts)
        )
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.acc5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.acc10 = Accuracy(task="multiclass", num_classes=num_classes, top_k=10)

        self.adv_mode = adv_mode
        self.train_atk = getattr(attacks, attacker)(self, **attacker_args)
        self.eval_atk = getattr(attacks, attacker)(self, **attacker_args)

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

    def forward(self, x):
        concept_pred = self.base(x)
        if self.hparams.cbm_mode == "fuzzy":
            concept = torch.sigmoid(concept_pred)
        elif self.hparams.cbm_mode == "bool":
            concept = torch.sigmoid(concept_pred)
            concept = torch.where(concept_pred > 0.5, 1, 0).float()
        elif self.hparams.cbm_mode == "hybrid":
            concept = concept_pred
        label_pred = self.classifier(concept)
        return label_pred, concept_pred

    def get_loss(self, logits, labels, adv_loss):
        label, concept = labels
        label_pred, concept_pred = logits[0], logits[1]
        if adv_loss == "bce":
            loss = F.binary_cross_entropy_with_logits(concept_pred, concept)
        elif adv_loss == "ce":
            loss = F.cross_entropy(label_pred, label)
        elif adv_loss == "combo":
            loss = F.cross_entropy(
                label_pred, label
            ) + self.hparams.concept_weight * F.binary_cross_entropy_with_logits(
                concept_pred, concept
            )
        return loss

    def shared_params(self):
        shared_params = {}
        for name, param in self.named_parameters():
            if "classifier" not in name:
                shared_params[name] = param
        return shared_params

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
            g0 = get_grad(label_loss, self)
            g1 = get_grad(concept_loss, self)
            if self.hparams.mtl_mode == "equal":
                g = gradient_normalize(g0, g1)
            else:
                g = gradient_ordered(g0, g1)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return loss

    def training_step(self, batch, batch_idx):
        img, label, concepts = batch
        if self.adv_mode:
            bs = img.shape[0] // 2
            adv_img = self.generate_adv_img(
                img[:bs], (label[:bs], concepts[:bs]), "train"
            )
            img = torch.cat([img[:bs], adv_img], dim=0)
            label = torch.cat([label[:bs], label[:bs]], dim=0)
            concepts = torch.cat([concepts[:bs], concepts[:bs]], dim=0)
        loss = self.train_step(img, label, concepts)
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
        if self.adv_mode:
            img = self.generate_adv_img(img, (label, concepts), "eval")
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

    def test_step(self, batch, batch_idx):
        self.eval_step(batch)
