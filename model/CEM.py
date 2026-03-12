from typing import Dict, Optional

import torch
from torch import nn
from torch import Tensor

from model.CBM import CBM
from utils import initialize_weights, modify_fc


class CEM(CBM):
    """Concept Embedding Model with probabilistic concept representations.

    Extends CBM with learned concept embeddings for more expressive
    concept-based predictions.
    """

    def __init__(self, embed_dim: int = 16, **kwargs):
        """Initialize the CEM model.

        Args:
            embed_dim: Dimension of concept embeddings.
            **kwargs: Arguments passed to parent CBM class.

        Returns:
            None.
        """
        super().__init__(**kwargs)
        modify_fc(self.base, kwargs["base"], 2 * embed_dim * self.num_concepts)

        self.concept_prob_gen = nn.Linear(2 * embed_dim, 1).apply(initialize_weights)

        self.classifier = nn.Linear(
            embed_dim * self.num_concepts, self.num_classes
        ).apply(initialize_weights)

    def forward(self, x: Tensor, concept_pred: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Forward pass through the CEM model.

        Args:
            x: Input image tensor of shape (batch_size, channels, height, width).
            concept_pred: Optional pre-computed concept predictions.

        Returns:
            Dictionary with 'label' (class predictions) and 'concept' (concept logits).
        """
        concept_context = self.base(x).reshape(
            x.size(0), -1, 2 * self.hparams.embed_dim
        )
        concept_context = nn.functional.leaky_relu(concept_context)
        if concept_pred is None:
            concept_pred = self.concept_prob_gen(concept_context).squeeze(-1)
        concept_probs = torch.sigmoid(concept_pred)
        pos_embed, neg_embed = concept_context.chunk(2, dim=2)
        concept_probs = concept_probs.unsqueeze(-1)
        combined_embed = pos_embed * concept_probs + neg_embed * (1 - concept_probs)
        concept_embed = combined_embed.view(combined_embed.size(0), -1)

        label_pred = self.classifier(concept_embed)
        return {"label": label_pred, "concept": concept_pred}
