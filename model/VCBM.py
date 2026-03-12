import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from hsic import nhsic, standardize
from model.CBM import CBM
from mtl import mtl
from utils import calc_info_loss

logger = logging.getLogger(__name__)


class VCBM(CBM):
    """Variational Concept Bottleneck Model with variational inference.

    Extends CBM with a variational autoencoder for learning stochastic
    concept representations.
    """

    def __init__(
        self,
        vib: float = 0.1,
        res_dim: int = 0,
        hsic_weight: float = 1e-4,
        hsic_kernel: str = "rbf",
        **kwargs,
    ):
        """Initialize the VCBM model.

        Args:
            vib: Weight for variational information bottleneck loss.
            res_dim: Dimension of residual/virtual concepts.
            hsic_weight: Weight for HSIC independence penalty.
            hsic_kernel: Kernel type for HSIC ('rbf', 'linear').
            **kwargs: Arguments passed to parent CBM class.

        Returns:
            None.
        """
        super().__init__(res_dim=res_dim, hsic_weight=hsic_weight, hsic_kernel=hsic_kernel, **kwargs)
        self.fc = nn.Linear(self.base.fc.in_features, 2 * (self.num_concepts + res_dim))
        self.base.fc = nn.Identity()
        self.classifier = nn.Linear(self.num_concepts + res_dim, self.num_classes)
        self.vib = vib
        self.res_dim = res_dim
        self.hsic_weight = hsic_weight
        self.hsic_kernel = hsic_kernel

        logger.info(
            f"VCBM initialized: vib=%.4f, res_dim=%d, hsic_weight=%.4f",
            vib,
            res_dim,
            hsic_weight,
        )

    def forward(self, x: Tensor, concept_pred: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Forward pass through the VCBM model.

        Args:
            x: Input image tensor of shape (batch_size, channels, height, width).
            concept_pred: Optional pre-computed concept predictions.

        Returns:
            Dictionary with 'label', 'concept', 'mu', and 'var' tensors.
        """
        features = self.base(x)
        statistics = self.fc(features)
        std, mu = torch.chunk(statistics, 2, dim=1)
        logits = mu + std * torch.randn_like(std)
        if concept_pred is None:
            concept_pred = logits
        else:
            logits[:, : self.num_concepts] = concept_pred[:, : self.num_concepts]
            concept_pred = logits

        label_pred = self.classifier(concept_pred)

        return {"label": label_pred, "concept": concept_pred, "mu": mu, "var": std**2}

    def calc_loss(self, gt: Dict[str, Tensor], pred: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Calculate loss for VCBM training including variational terms.

        Args:
            gt: Ground truth dictionary with 'label' and 'concept' tensors.
            pred: Prediction dictionary with 'label', 'concept', 'mu', and 'var' tensors.

        Returns:
            Dictionary of losses including 'Label Loss', 'Concept Loss', 'Info Loss', and 'Loss'.
        """
        label_pred, concept_pred, mu, var = pred["label"], pred["concept"], pred["mu"], pred["var"]
        label, concepts = gt["label"], gt["concept"]
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred[:, : self.num_concepts],
            concepts,
            weight=(
                self.dm.imbalance_weights.to(self.device)
                if self.hparams.weighted_bce
                else None
            ),
        )
        info_loss = calc_info_loss(mu, var)
        label_loss = F.cross_entropy(label_pred, label)
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.vib * info_loss
        )
        losses = {
            "Label Loss": label_loss,
            "Concept Loss": concept_loss,
            "Info Loss": info_loss,
            "Loss": loss,
        }

        # add HSIC constraint
        if self.hsic_weight > 0 and self.res_dim > 0:
            # separate semantic and virtual concepts
            semantic_concepts = concept_pred[:, : self.num_concepts]
            virtual_concepts = concept_pred[:, self.num_concepts :]
            
            # standardize semantic and virtual concepts
            semantic_std = standardize(semantic_concepts)
            virtual_std = standardize(virtual_concepts)
            
            # compute normalized HSIC
            hsic_loss = nhsic(semantic_std, virtual_std, 
                            kernel_c=self.hsic_kernel, 
                            kernel_v=self.hsic_kernel)
            
            loss = loss + self.hsic_weight * hsic_loss
            losses["HSIC Loss"] = hsic_loss
            losses["Loss"] = loss

        if self.hparams.mtl_mode != "normal":
            g = mtl([label_loss, concept_loss], self, self.hparams.mtl_mode)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return losses
