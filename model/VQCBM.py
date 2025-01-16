from torch import nn
import torch.nn.functional as F
from model import CBM
from VQ import VectorQuantizeEMA
from mtl import get_grad, gradient_ordered
from utils import initialize_weights


class VQCBM(CBM):
    def __init__(
        self,
        # VQ params
        embed_dim: int = 32,
        codebook_size: int = 5120,
        codebook_weight: float = 0.25,
        quantizer: str = "EMA",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base.fc = nn.Linear(
            self.base.fc.in_features, embed_dim * self.num_concepts
        ).apply(
            initialize_weights
        )  # output num_concepts embeddings for VQ

        self.concept_prob_gen = nn.Linear(
            embed_dim * self.num_concepts, self.num_concepts
        ).apply(initialize_weights)

        self.classifier = nn.Linear(
            self.num_concepts, self.num_classes
        ).apply(initialize_weights)

        if quantizer == "EMA":
            self.quantizer = VectorQuantizeEMA(embed_dim, codebook_size)

    def forward(self, x):
        logits = self.base(x)
        logits = logits.view(
            logits.size(0), self.num_concepts, -1
        )  # B, num_concepts, embed_dim
        quantized_concept, codebook_loss, _ = self.quantizer(logits)
        concept_pred = self.concept_prob_gen(
            quantized_concept.view(quantized_concept.size(0), -1)
        )
        label_pred = self.classifier(concept_pred)
        return label_pred, concept_pred, codebook_loss

    def train_step(self, img, label, concepts):
        label_pred, concept_pred, codebook_loss = self(img)
        label_loss = F.cross_entropy(label_pred, label)
        concept_loss=F.binary_cross_entropy_with_logits(concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device))
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.codebook_weight * codebook_loss
        )
        self.manual_backward(loss)
        return loss
