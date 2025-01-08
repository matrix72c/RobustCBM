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
        embedding_dim: int = 32,
        codebook_size: int = 5120,
        codebook_weight: float = 0.25,
        quantizer: str = "EMA",
        **kwargs,
    ):
        super().__init__(**kwargs)
        base = list(self.base.children())[:-2]
        in_channels = self.base.fc.in_features
        base.append(nn.Conv2d(in_channels, embedding_dim, 1))
        self.base = nn.Sequential(*base)
        self.quantizer = VectorQuantizeEMA(embedding_dim, codebook_size)

        self.pool = nn.AdaptiveAvgPool1d(self.num_concepts)

        self.concept_prob_gen = nn.Linear(
            self.hparams.embedding_dim * self.num_concepts, self.num_concepts
        )

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1, self.hparams.embedding_dim)
        vq, codebook_loss, _ = self.quantizer(x)
        vq.transpose_(1, 2)
        embed = self.pool(vq)
        embed = embed.reshape(embed.size(0), -1)
        concept_pred = self.concept_prob_gen(embed)
        label_pred = self.classifier(concept_pred)
        return label_pred, concept_pred, codebook_loss

    def train_step(self, img, label, concepts):
        label_pred, concept_pred, codebook_loss = self(img)
        label_loss = F.cross_entropy(label_pred, label)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.codebook_weight * codebook_loss
        )
        if self.hparams.mtl_mode == "normal":
            self.manual_backward(loss)
        else:
            g0 = get_grad(label_loss, self)
            g1 = get_grad(concept_loss, self)
            g2 = get_grad(codebook_loss, self)
            g = gradient_ordered(g1, g2)
            g = gradient_ordered(g0, g)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return loss
