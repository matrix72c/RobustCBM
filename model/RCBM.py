from torch import nn
import torch
import torch.nn.functional as F
from model import CBM, VIB
from VQ import VectorQuantizeEMA
from mtl import get_grad, gradient_ordered
from utils import initialize_weights


class RCBM(CBM):
    def __init__(
        self,
        embedding_dim: int = 32,
        code_weight: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base.fc = nn.Linear(
            self.base.fc.in_features, embedding_dim * self.num_concepts
        ).apply(initialize_weights)

        self.embed = nn.Embedding(self.num_concepts, embedding_dim).apply(
            initialize_weights
        )

        self.classifier = nn.Linear(
            embedding_dim * self.num_concepts, self.num_classes
        ).apply(initialize_weights)

    def forward(self, x):
        z = self.base(x)
        z = z.view(
            z.size(0), self.hparams.num_concepts, -1
        )  # B, num_concepts, embedding_dim
        z_q = self.embed.weight.unsqueeze(0).expand(z.size(0), -1, -1)
        concept_pred = F.cosine_similarity(z, z_q, dim=2)
        concept_pred = (concept_pred + 1) / 2
        diff = (z.detach() - z_q).pow(2).mean() + 0.25 * (z - z_q.detach()).pow(2).mean()
        # z_q = z + (z_q - z).detach()
        weight_z = z_q * concept_pred.unsqueeze(-1)
        label_pred = self.classifier(weight_z.view(z.size(0), -1))
        return label_pred, concept_pred, diff

    def train_step(self, img, label, concepts):
        label_pred, concept_pred, code_loss = self(img)
        label_loss = F.cross_entropy(label_pred, label)
        concept_loss = F.binary_cross_entropy(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.code_weight * code_loss
        )
        self.log("label_loss", label_loss, prog_bar=True)
        self.log("concept_loss", concept_loss, prog_bar=True)
        self.log("code_loss", code_loss, prog_bar=True)
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        return loss
