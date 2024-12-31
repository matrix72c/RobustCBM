import torch
from torch import nn
import torch.nn.functional as F
from model import VQCBM
from VQ import VectorQuantizeEMA
from utils import initialize_weights


class VQCEM(VQCBM):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        embedding_dim = self.hparams.embedding_dim
        self.base.fc = nn.Linear(
            self.base.fc.in_features, 2 * embedding_dim * self.num_concepts
        ).apply(
            initialize_weights
        )  # output num_concepts embeddings for VQ

        self.concept_prob_gen = nn.Linear(
            2 * embedding_dim * self.num_concepts, self.num_concepts
        ).apply(initialize_weights)

    def forward(self, x):
        logits = self.base(x)
        logits = logits.view(
            logits.size(0), 2 * self.hparams.num_concepts, -1
        )  # B, num_concepts, embedding_dim
        quantized_concept, codebook_loss, _ = self.quantizer(logits)
        concept_pred = self.concept_prob_gen(
            quantized_concept.view(quantized_concept.size(0), -1)
        )
        pos_embed, neg_embed = torch.chunk(quantized_concept, 2, dim=1)
        pos_embed, neg_embed = pos_embed.view(
            pos_embed.size(0), -1, self.hparams.embedding_dim
        ), neg_embed.view(neg_embed.size(0), -1, self.hparams.embedding_dim)
        concept_pred.unsqueeze_(-1)
        combined_embed = pos_embed * concept_pred + neg_embed * (1 - concept_pred)
        concept_embed = combined_embed.view(combined_embed.size(0), -1)
        concept_pred = concept_pred.squeeze(-1)
        label_pred = self.classifier(concept_embed)
        return label_pred, concept_pred, codebook_loss
