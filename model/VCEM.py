import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from utils import calc_info_loss, initialize_weights


class VCEM(CBM):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        use_pretrained: bool,
        concept_weight: float,
        lr: float,
        optimizer: str,
        scheduler_patience: int,
        adv_mode: bool,
        adv_strategy: str,
        embed_size: int,
        vib_lambda: float,
    ):
        super().__init__(
            base,
            num_classes,
            num_concepts,
            use_pretrained,
            concept_weight,
            lr,
            optimizer,
            scheduler_patience,
            adv_mode,
            adv_strategy,
        )
        self.base.fc = nn.Linear(
            self.base.fc.in_features, 4 * embed_size * num_concepts
        ).apply(initialize_weights)

        self.concept_prob_gen = nn.Linear(
            2 * embed_size * num_concepts, num_concepts
        ).apply(initialize_weights)

        self.classifier = nn.Linear(embed_size * num_concepts, num_classes).apply(
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
        if self.get_adv_img:
            return label_pred
        return label_pred, concept_pred, mu, std**2

    def shared_step(self, img, label, concepts):
        label_pred, concept_pred, mu, var = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(concept_pred, concepts)
        info_loss = calc_info_loss(mu, var)
        self.log(
            "info_loss",
            info_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        class_loss = F.cross_entropy(label_pred, label)
        loss = (
            class_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.vib_lambda * info_loss
        )
        return loss, label_pred, concept_pred
