import torch
from torch import nn
from model import CBM
from utils import initialize_weights


class CEM(CBM):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        use_pretrained: bool,
        concept_weight: float,
        lr: float,
        optimizer: str,
        step_size: int,
        adv_mode: bool,
        adv_strategy: str,
        embed_size: int,
    ):
        super().__init__(
            base,
            num_classes,
            num_concepts,
            use_pretrained,
            concept_weight,
            lr,
            optimizer,
            step_size,
            adv_mode,
            adv_strategy,
        )
        self.base.fc = nn.Linear(
            self.base.fc.in_features, 2 * embed_size * num_concepts
        ).apply(initialize_weights)

        self.concept_prob_gen = nn.Linear(
            2 * embed_size * num_concepts, num_concepts
        ).apply(initialize_weights)

        self.classifier = nn.Linear(embed_size * num_concepts, num_classes).apply(
            initialize_weights
        )

    def forward(self, x):
        concept_context = self.base(x)
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
        if self.get_adv_img:
            return label_pred
        return label_pred, concept_pred
