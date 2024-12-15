from torch import nn
import torch.nn.functional as F
from model import CBM, VectorQuantizeEMA
from utils import initialize_weights


class VQCBM(CBM):
    def __init__(
        self,
        num_classes: int,
        num_concepts: int,
        real_concepts: int,
        base: str = "resnet50",
        use_pretrained: bool = True,
        concept_weight: float = 1,
        lr: float = 0.1,
        scheduler_arg: int = 30,
        adv_mode: bool = False,
        # VQ params
        embedding_dim: int = 64,
        codebook_size: int = 5120,
        codebook_weight: float = 0.25,
        quantizer: str = "EMA",
    ):
        super().__init__(
            num_classes=num_classes,
            num_concepts=num_concepts,
            real_concepts=real_concepts,
            base=base,
            use_pretrained=use_pretrained,
            concept_weight=concept_weight,
            lr=lr,
            scheduler_arg=scheduler_arg,
            adv_mode=adv_mode,
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
