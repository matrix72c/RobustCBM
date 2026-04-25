import logging
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

    def __init__(
        self,
        embed_dim: int = 16,
        training_intervention_prob: float = 0.25,
        **kwargs,
    ):
        """Initialize the CEM model.

        Args:
            embed_dim: Dimension of concept embeddings.
            training_intervention_prob: Probability of applying RandInt to each
                concept during training.
            **kwargs: Arguments passed to parent CBM class.

        Returns:
            None.
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.training_intervention_prob = training_intervention_prob
        modify_fc(self.base, kwargs["base"], 2 * embed_dim * self.num_concepts)

        self.concept_prob_gen = nn.Linear(2 * embed_dim, 1).apply(initialize_weights)

        self.classifier = nn.Linear(
            embed_dim * self.num_concepts, self.num_classes
        ).apply(initialize_weights)


    def forward(
        self,
        x: Tensor,
        concept_pred: Optional[Tensor] = None,
        intervention_idxs: Optional[Tensor] = None,
        concepts: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass through the CEM model.

        Args:
            x: Input image tensor of shape (batch_size, channels, height, width).
            concept_pred: Optional pre-computed concept predictions.
            intervention_idxs: Optional binary mask indicating concepts whose
                probabilities should be replaced by ground-truth concepts.
            concepts: Ground-truth concept values used for interventions.

        Returns:
            Dictionary with 'label' (class predictions) and 'concept' (concept logits).
        """
        concept_context = self.base(x).reshape(
            x.size(0), -1, 2 * self.embed_dim
        )
        concept_context = nn.functional.leaky_relu(concept_context)
        if concept_pred is None:
            concept_pred = self.concept_prob_gen(concept_context).squeeze(-1)
        concept_probs = torch.sigmoid(concept_pred)
        if intervention_idxs is not None:
            if concepts is None:
                raise ValueError("concepts must be provided when intervention_idxs is set")
            intervention_idxs = intervention_idxs.to(
                device=concept_probs.device,
                dtype=concept_probs.dtype,
            )
            concepts = concepts.to(device=concept_probs.device, dtype=concept_probs.dtype)
            concept_probs = (
                concept_probs * (1 - intervention_idxs)
                + concepts * intervention_idxs
            )
        pos_embed, neg_embed = concept_context.chunk(2, dim=2)
        concept_probs = concept_probs.unsqueeze(-1)
        combined_embed = pos_embed * concept_probs + neg_embed * (1 - concept_probs)
        concept_embed = combined_embed.view(combined_embed.size(0), -1)

        label_pred = self.classifier(concept_embed)
        return {"label": label_pred, "concept": concept_pred}

    def _sample_training_intervention_idxs(self, concepts: Tensor) -> Optional[Tensor]:
        """Sample the RandInt mask used by the reference CEM implementation."""
        if self.training_intervention_prob == 0:
            return None
        mask = torch.bernoulli(
            torch.full(
                (self.num_concepts,),
                self.training_intervention_prob,
                device=concepts.device,
                dtype=concepts.dtype,
            )
        )
        return mask.unsqueeze(0).expand(concepts.size(0), -1)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], _batch_idx: int) -> Tensor:
        """Training step for CEM with RandInt training interventions."""
        img, label, concepts = batch
        if self.train_mode != "Std":
            bs = img.shape[0] // 2
            adv_img = self.generate_adv(
                img[:bs], label[:bs], concepts[:bs], self.train_mode
            )
            img = torch.cat([img[:bs], adv_img], dim=0)
            label = torch.cat([label[:bs], label[:bs]], dim=0)
            concepts = torch.cat([concepts[:bs], concepts[:bs]], dim=0)

        intervention_idxs = self._sample_training_intervention_idxs(concepts)
        pred = self(
            img,
            intervention_idxs=intervention_idxs,
            concepts=concepts,
        )
        losses = self.calc_loss({"label": label, "concept": concepts}, pred)
        loss = losses["Loss"]
        return loss
