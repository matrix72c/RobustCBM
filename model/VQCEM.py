import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from VQ import VectorQuantizeEMA
from mtl import get_grad, gradient_ordered
from utils import initialize_weights


class VQCEM(CBM):
    def __init__(
        self,
        embed_dim: int = 32,
        codebook_size: int = 5120,
        codebook_weight: float = 0.1,
        quantizer: str = "EMA",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base.fc = nn.Linear(
            self.base.fc.in_features, embed_dim * self.num_concepts
        ).apply(initialize_weights)

        self.concept_prob = nn.ModuleList(
            [
                nn.Linear(embed_dim, 1).apply(initialize_weights)
                for _ in range(self.num_concepts)
            ]
        )

        self.quantizer = VectorQuantizeEMA(embed_dim, codebook_size)
        self.classifier = nn.Linear(
            self.num_concepts * embed_dim, self.num_classes
        ).apply(initialize_weights)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1, self.hparams.embed_dim)
        vq, codebook_loss, embed_ind = self.quantizer(x)
        concept_pred = torch.cat(
            [
                self.concept_prob[i](vq[:, i])
                for i in range(self.num_concepts)
            ],
            dim=1,
        )
        label_pred = self.classifier(vq.view(vq.size(0), -1))
        return label_pred, torch.sigmoid(concept_pred), codebook_loss

    def train_step(self, img, label, concepts):
        label_pred, concept_pred, codebook_loss = self(img)
        label_loss = F.cross_entropy(label_pred, label)
        concept_loss = F.binary_cross_entropy(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.codebook_weight * codebook_loss
        )
        self.manual_backward(loss)
        return loss
