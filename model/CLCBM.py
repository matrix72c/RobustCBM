import torch
from torch import nn
from model import CBM
from utils import initialize_weights, modify_fc
import torch.nn.functional as F


class CLCBM(CBM):
    def __init__(self, embed_dim: int = 32, cl_weight: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        modify_fc(self.base, kwargs["base"], embed_dim * self.num_concepts)

        self.concept_prob = nn.Linear(
            embed_dim * self.num_concepts, self.num_concepts
        ).apply(initialize_weights)

    def forward(self, x):
        z = self.base(x)
        concept_pred = self.concept_prob(z.view(z.size(0), -1))
        concept_features = z.view(z.size(0), self.num_concepts, -1)
        label_pred = self.classifier(concept_pred)
        return label_pred, torch.sigmoid(concept_pred), concept_features

    def contrastive_loss(self, z, concepts, margin=1.0, lambda_neg=1.0):
        B, N, E = z.size()
        z_flat = z.view(B * N, E)
        c_flat = concepts.view(-1)
        dist_matrix = torch.cdist(z_flat, z_flat, p=2)
        c_i = c_flat.unsqueeze(1)
        c_j = c_flat.unsqueeze(0)
        pos_mask = (c_i == 1) & (c_j == 1)
        eye_mask = torch.eye(B * N, device=z.device).bool()
        pos_mask = pos_mask & (~eye_mask)
        neg_mask = ((c_i == 1) & (c_j == 0)) | ((c_i == 0) & (c_j == 1))

        pos_loss = dist_matrix[pos_mask].pow(2).mean()
        neg_loss = F.relu(margin - dist_matrix[neg_mask]).pow(2).mean()
        return pos_loss + lambda_neg * neg_loss

    def train_step(self, img, label, concepts):
        label_pred, concept_pred, z = self(img)
        label_loss = F.cross_entropy(label_pred, label)
        concept_loss = F.binary_cross_entropy(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        contrastive_loss = self.contrastive_loss(z, concepts)
        loss = (
            label_loss
            + concept_loss * self.hparams.concept_weight
            + contrastive_loss * self.hparams.cl_weight
        )
        self.manual_backward(loss)
        self.log(
            "cl_loss", contrastive_loss, prog_bar=True, on_step=True, on_epoch=False
        )
        return loss
