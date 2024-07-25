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
        concept_weight: float = 0.01,
        lr: float = 1e-3,
        step_size: int = 15,
        gamma: float = 0.1,
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
        )
        self.concept_context_gen = nn.ModuleList(
            [
                nn.Linear(self.base.fc.in_features, 2 * embed_size)
                for _ in range(num_concepts)
            ]
        )
        self.concept_prob_gen = nn.ModuleList(
            [nn.Linear(2 * embed_size, 1) for _ in range(num_concepts)]
        )
        self.base = nn.Sequential(*list(self.base.children())[:-1])
        self.classifier = nn.Linear(embed_size * num_concepts, num_classes)

    def forward(self, x):
        features = self.base(x)
        features = features.view(features.size(0), -1)
        concept_context = torch.cat(
            [context(features).unsqueeze(1) for context in self.concept_context_gen],
            dim=1,
        )
        concept_pred = torch.stack(
            [
                F.sigmoid(prob(concept_context[:, i, :]))
                for i, prob in enumerate(self.concept_prob_gen)
            ],
            dim=1,
        )
        pos_embed = concept_context[:, :, : self.hparams.embed_size]
        neg_embed = concept_context[:, :, self.hparams.embed_size :]
        combined_embed = pos_embed * concept_pred + neg_embed * (1 - concept_pred)
        concept_embed = combined_embed.view(combined_embed.size(0), -1)
        concept_pred = concept_pred.squeeze(-1)
        class_pred = self.classifier(concept_embed)
        if self.get_adv_img:
            return class_pred
        return class_pred, concept_pred
