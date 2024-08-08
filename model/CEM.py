import lightning as L
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from model import CBM


class CEM(CBM):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        embed_size: int = 16,
        use_pretrained: bool = True,
        concept_weight: float = 0.5,
        lr: float = 1e-4,
        step_size: list = [15, 30],
        gamma: float = 0.1,
        adv_training: bool = False,
    ):
        super().__init__(
            base,
            num_classes,
            num_concepts,
            use_pretrained,
            concept_weight,
            lr,
            step_size,
            gamma,
            adv_training,
        )
        self.base.fc = nn.Linear(
            self.base.fc.in_features, 2 * embed_size * num_concepts
        )
        self.concept_prob_gen = nn.Linear(2 * embed_size * num_concepts, num_concepts)
        self.classifier = nn.Linear(embed_size * num_concepts, num_classes)

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

        class_pred = self.classifier(concept_embed)
        if self.get_adv_img:
            return class_pred
        return class_pred, concept_pred
