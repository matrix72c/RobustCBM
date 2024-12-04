import torch
from torch import nn
import torch.nn.functional as F
from model import CBM, VectorQuantizeEMA
from utils import initialize_weights


class VQCEM(CBM):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        use_pretrained: bool,
        concept_weight: float,
        lr: float,
        optimizer: str,
        embedding_dim: int,
        codebook_size: int,
        codebook_weight: float,
        quantizer: str,
        scheduler_arg: int,
        classifier: str = "FC",
        adv_mode: bool = False,
    ):
        super().__init__(
            base,
            num_classes,
            num_concepts,
            use_pretrained,
            concept_weight,
            lr,
            optimizer,
            scheduler_arg,
            adv_mode,
        )
        self.base.fc = nn.Linear(
            self.base.fc.in_features, 2 * embedding_dim * num_concepts
        ).apply(
            initialize_weights
        )  # output num_concepts embeddings for VQ

        self.concept_prob_gen = nn.Linear(
            2 * embedding_dim * num_concepts, num_concepts
        ).apply(initialize_weights)
        if classifier == "FC":
            self.classifier = nn.Linear(
                embedding_dim * num_concepts, num_classes
            ).apply(initialize_weights)
        elif classifier == "MLP":
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim * num_concepts, 3 * num_concepts),
                nn.ReLU(),
                nn.Linear(3 * num_concepts, num_classes),
            ).apply(initialize_weights)
        if quantizer == "EMA":
            self.quantizer = VectorQuantizeEMA(embedding_dim, codebook_size)

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
        if self.get_adv_img:
            return label_pred
        return label_pred, concept_pred, codebook_loss

    def shared_step(self, img, label, concepts):
        label_pred, concept_pred, codebook_loss = self(img)
        loss = (
            F.cross_entropy(label_pred, label)
            + self.hparams.concept_weight
            * F.binary_cross_entropy_with_logits(concept_pred, concepts)
            + self.hparams.codebook_weight * codebook_loss
        )
        return loss, label_pred, concept_pred
