from torch import nn
import torch.nn.functional as F
from model import CBM, VectorQuantizeEMA
from utils import initialize_weights


class VQCBM(CBM):
    def __init__(
        self,
        base: str,
        num_classes: int,
        num_concepts: int,
        use_pretrained: bool,
        concept_weight: float,
        lr: float,
        optimizer: str,
        scheduler_arg: int,
        adv_mode: bool,
        adv_strategy: str,
        # VQ params
        embedding_dim: int,
        codebook_size: int,
        codebook_weight: float,
        quantizer: str,
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
            adv_strategy,
        )
        self.base.fc = nn.Linear(
            self.base.fc.in_features, embedding_dim * num_concepts
        ).apply(
            initialize_weights
        )  # output num_concepts embeddings for VQ

        self.concept_prob_gen = nn.Linear(
            embedding_dim * num_concepts, num_concepts
        ).apply(initialize_weights)

        self.classifier = nn.Linear(
            embedding_dim * num_concepts, num_classes
        ).apply(initialize_weights)

        if quantizer == "EMA":
            self.quantizer = VectorQuantizeEMA(embedding_dim, codebook_size)

    def forward(self, x):
        logits = self.base(x)
        logits = logits.view(
            logits.size(0), self.hparams.num_concepts, -1
        )  # B, num_concepts, embedding_dim
        quantized_concept, codebook_loss, _ = self.quantizer(logits)
        concept_pred = self.concept_prob_gen(
            quantized_concept.view(quantized_concept.size(0), -1)
        )
        label_pred = self.classifier(
            quantized_concept.view(quantized_concept.size(0), -1)
        )
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
