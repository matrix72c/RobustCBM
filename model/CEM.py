import torch
from torch import nn
from model import CBM
from utils import initialize_weights, modify_fc


class CEM(CBM):
    def __init__(self, embed_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        modify_fc(self.base, kwargs["base"], 2 * embed_dim * self.num_concepts)

        self.concept_prob_gen = nn.Linear(
            2 * embed_dim * self.num_concepts, self.num_concepts
        ).apply(initialize_weights)

        self.classifier = nn.Linear(
            embed_dim * self.num_concepts, self.num_classes
        ).apply(initialize_weights)

    def forward(self, x, concepts=None):
        concept_context = self.base(x)
        concept_pred = self.concept_prob_gen(concept_context)

        if concepts is not None:
            concept_pred = self.intervene(
                concepts,
                concept_pred,
                self.intervene_budget,
                self.concept_group_map,
            )

        pos_embed, neg_embed = torch.chunk(concept_context, 2, dim=1)
        pos_embed, neg_embed = pos_embed.view(
            pos_embed.size(0), -1, self.hparams.embed_dim
        ), neg_embed.view(neg_embed.size(0), -1, self.hparams.embed_dim)
        concept_pred.unsqueeze_(-1)
        combined_embed = pos_embed * concept_pred + neg_embed * (1 - concept_pred)
        concept_embed = combined_embed.view(combined_embed.size(0), -1)
        concept_pred = concept_pred.squeeze(-1)

        label_pred = self.classifier(concept_embed)
        return label_pred, torch.sigmoid(concept_pred)
