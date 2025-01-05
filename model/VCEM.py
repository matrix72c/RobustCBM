import torch
from torch import nn
import torch.nn.functional as F
from model import VCBM
from utils import initialize_weights


class VCEM(VCBM):
    def __init__(
        self,
        embed_size: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base.fc = nn.Linear(
            self.base.fc.in_features, 4 * embed_size * self.num_concepts
        ).apply(initialize_weights)

        self.concept_prob_gen = nn.Linear(
            2 * embed_size * self.num_concepts, self.num_concepts
        ).apply(initialize_weights)

        self.classifier = nn.Linear(embed_size * self.num_concepts, self.num_classes).apply(
            initialize_weights
        )

    def forward(self, x):
        statistics = self.base(x)
        std, mu = torch.chunk(statistics, 2, dim=1)
        concept_context = mu + std * torch.randn_like(std)

        concept_pred = self.concept_prob_gen(concept_context)

        pos_embed, neg_embed = torch.chunk(concept_context, 2, dim=1)
        pos_embed, neg_embed = pos_embed.view(
            pos_embed.size(0), -1, self.hparams.embed_size
        ), neg_embed.view(neg_embed.size(0), -1, self.hparams.embed_size)
        concept_pred.unsqueeze_(-1)
        combined_embed = pos_embed * concept_pred + neg_embed * (1 - concept_pred)
        concept_embed = combined_embed.view(combined_embed.size(0), -1)
        concept_pred = concept_pred.squeeze(-1)

        label_pred = self.classifier(concept_embed)
        return label_pred, concept_pred, mu, std**2
