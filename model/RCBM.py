from torch import nn
import torch
import torch.nn.functional as F
from model import CBM
from utils import initialize_weights, modify_fc


class RCBM(CBM):
    def __init__(
        self,
        embedding_dim: int = 32,
        codebook_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        base = list(self.base.children())[:-2]
        self.base = nn.Sequential(*base)

        self.embed = nn.Embedding(self.num_concepts, embedding_dim).apply(
            initialize_weights
        )

        self.attn = nn.MultiheadAttention(
            embedding_dim, 4, dropout=0.1, batch_first=True
        )
        self.proj_z = nn.Linear(2048, embedding_dim).apply(initialize_weights)
        self.attn_layernorm = nn.LayerNorm(embedding_dim)

        self.concept_prob = nn.ModuleList(
            [nn.Linear(embedding_dim, 1) for _ in range(self.num_concepts)]
        )

        self.classifier = nn.Linear(
            embedding_dim * self.num_concepts, self.num_classes
        ).apply(initialize_weights)

    def forward(self, x):
        z = self.base(x)
        B, C, H, W = z.size()
        z = z.view(B, C, -1).permute(0, 2, 1)
        z = self.proj_z(z)
        z_q = self.embed.weight.unsqueeze(0).expand(B, -1, -1)
        attn_z, _ = self.attn(z_q, z, z)
        attn_z = attn_z + z_q
        attn_z = self.attn_layernorm(attn_z)
        concept_pred = torch.cat(
            [
                torch.sigmoid(self.concept_prob[i](attn_z[:, i]))
                for i in range(self.num_concepts)
            ],
            dim=1,
        )
        weight_z = attn_z * concept_pred.unsqueeze(-1)
        label_pred = self.classifier(weight_z.reshape(B, -1))
        return label_pred, concept_pred, attn_z

    def prototype_loss(self, z, z_q, concepts, margin=1.0, lambda_neg=1.0):
        """
        z: Tensor of shape [B, N, E]
        z_q: Tensor of shape [N, E]
        concepts: Tensor of shape [B, N], binary indicators
        margin: float, margin for negative samples
        lambda_neg: float, weight for negative loss
        """
        B, N, E = z.shape

        z_q_expanded = z_q.unsqueeze(0).expand(B, -1, -1) # [B, N, E]
        concepts = concepts.unsqueeze(-1)  # [B, N, 1]
        positive_loss = concepts * F.mse_loss(z, z_q_expanded, reduction='none')
        positive_loss = positive_loss.sum() / (B * N)

        z_flat = z.reshape(B * N, E)  # [B*N, E]
        z_q_flat = z_q  # [N, E]
        distances = torch.cdist(z_flat, z_q_flat, p=2)  # [B*N, N]
        mask_neg = (~concepts.view(B * N).bool()).unsqueeze(1).float()  # [B*N, 1]
        hinge = F.relu(margin - distances)  # [B*N, N]
        negative_loss = mask_neg * hinge
        negative_loss = negative_loss.sum() / (B * N)
        total_loss = positive_loss + lambda_neg * negative_loss
        return total_loss

    def train_step(self, img, label, concepts):
        label_pred, concept_pred, z = self(img)
        label_loss = F.cross_entropy(label_pred, label)
        concept_loss = F.binary_cross_entropy(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        code_loss = self.prototype_loss(z, self.embed.weight, concepts)
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.codebook_weight * code_loss
        )
        self.log("label_loss", label_loss, prog_bar=True)
        self.log("concept_loss", concept_loss, prog_bar=True)
        self.log("code_loss", code_loss, prog_bar=True)
        self.manual_backward(loss)
        return loss
