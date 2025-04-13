import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from utils import initialize_weights, modify_fc


class CEM(CBM):
    def __init__(self, embed_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        modify_fc(self.base, kwargs["base"], 2 * embed_dim * self.num_concepts)

        self.concept_prob_gen = nn.Linear(2 * embed_dim, 1).apply(initialize_weights)

        self.classifier = nn.Linear(
            embed_dim * self.num_concepts, self.num_classes
        ).apply(initialize_weights)

    def forward(self, x, concept_pred=None):
        concept_context = self.base(x).reshape(x.size(0), -1, 2 * self.hparams.embed_dim)
        if concept_pred is None:
            concept_pred = (
                self.concept_prob_gen(
                    concept_context
                )
                .squeeze(-1)
            )
        concept_probs = torch.sigmoid(concept_pred)
        pos_embed, neg_embed = concept_context.chunk(2, dim=2)
        concept_probs = concept_probs.unsqueeze(-1)
        combined_embed = pos_embed * concept_probs + neg_embed * (1 - concept_probs)
        concept_embed = combined_embed.view(combined_embed.size(0), -1)

        label_pred = self.classifier(concept_embed)
        return label_pred, concept_pred
