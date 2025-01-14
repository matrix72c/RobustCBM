import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from VQ import VectorQuantizeEMA
from mtl import get_grad, gradient_ordered
from utils import initialize_weights


class VQCEM(CBM):
    def __init__(
        self,
        embedding_dim: int = 16,
        codebook_size: int = 512,
        codebook_weight: float = 0.25,
        quantizer: str = "EMA",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base.fc = nn.Linear(
            self.base.fc.in_features, embedding_dim * self.num_concepts
        ).apply(initialize_weights)

        self.concept_prob = [nn.Linear(embedding_dim, 1).apply(initialize_weights) for _ in range(self.num_concepts)]

        self.quantizer = VectorQuantizeEMA(embedding_dim, codebook_size)
        self.classifier = nn.Linear(
            self.num_concepts * embedding_dim, self.num_classes
        ).apply(initialize_weights)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1, self.hparams.embedding_dim)
        vq, codebook_loss, embed_ind = self.quantizer(x)
        concept_pred = torch.cat(
            [
                self.concept_prob[i].to(self.device)(vq[:, i])
                for i in range(self.num_concepts)
            ],
            dim=1,
        )
        label_pred = self.classifier(vq.view(vq.size(0), -1))
        return label_pred, concept_pred, codebook_loss, vq, embed_ind

    def train_step(self, img, label, concepts):
        label_pred, concept_pred, codebook_loss, vq, embed_ind = self(img)
        label_loss = F.cross_entropy(label_pred, label)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.codebook_weight * codebook_loss
        )
        if self.hparams.mtl_mode == "normal":
            self.manual_backward(loss)
        else:
            g0 = get_grad(label_loss, self)
            g1 = get_grad(concept_loss, self)
            g2 = get_grad(codebook_loss, self)
            g = gradient_ordered(g1, g2)
            g = gradient_ordered(g0, g)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return loss

    def compute_concept_means(self, vq, embed_ind, concepts):
        """
        计算每个概念的具备和不具备样本的均值向量
        """
        device = vq.device
        num_concepts = self.hparams.num_concepts
        embedding_dim = self.hparams.embedding_dim

        # 初始化均值向量
        concept_positive_means = torch.zeros(num_concepts, embedding_dim, device=device)
        concept_negative_means = torch.zeros(num_concepts, embedding_dim, device=device)
        concept_positive_counts = torch.zeros(num_concepts, device=device)
        concept_negative_counts = torch.zeros(num_concepts, device=device)

        for c in range(num_concepts):
            # 找到具备该概念的样本索引
            pos_indices = concepts[:, c].bool()
            neg_indices = ~pos_indices

            if pos_indices.sum() > 0:
                # 聚合具备该概念的样本的特征向量
                pos_vq = vq[pos_indices].view(-1, embedding_dim)
                concept_positive_means[c] = pos_vq.mean(dim=0)
                concept_positive_counts[c] = pos_vq.size(0)
            if neg_indices.sum() > 0:
                # 聚合不具备该概念的样本的特征向量
                neg_vq = vq[neg_indices].view(-1, embedding_dim)
                concept_negative_means[c] = neg_vq.mean(dim=0)
                concept_negative_counts[c] = neg_vq.size(0)

        # 防止除以零
        concept_positive_counts = concept_positive_counts.clamp(min=1).unsqueeze(1)
        concept_negative_counts = concept_negative_counts.clamp(min=1).unsqueeze(1)

        # 计算最终均值
        concept_positive_means /= concept_positive_counts
        concept_negative_means /= concept_negative_counts

        return concept_positive_means, concept_negative_means

    def compute_contrastive_loss(self, vq, embed_ind, concepts):
        """
        计算概念对比损失
        """
        concept_positive_means, concept_negative_means = self.compute_concept_means(vq, embed_ind, concepts)

        device = vq.device
        num_concepts = self.hparams.num_concepts

        # 扩展均值向量以匹配批次大小
        pos_means = concept_positive_means.unsqueeze(0).expand(vq.size(0), -1, -1)
        neg_means = concept_negative_means.unsqueeze(0).expand(vq.size(0), -1, -1)

        # 计算距离
        # 计算每个样本中每个嵌入向量到正均值和负均值的距离
        # vq: (batch_size, embedding_num, embedding_dim)
        # pos_means, neg_means: (batch_size, num_concepts, embedding_dim)
        # 使用广播机制计算欧氏距离
        distance_pos = torch.norm(vq.unsqueeze(1) - pos_means.unsqueeze(2), dim=-1)  # (batch_size, num_concepts, embedding_num)
        distance_neg = torch.norm(vq.unsqueeze(1) - neg_means.unsqueeze(2), dim=-1)  # (batch_size, num_concepts, embedding_num)

        # 对每个概念，最小化到正均值的距离，最大化到负均值的距离
        # 使用对比损失函数
        margin = 1.0  # 可以作为超参数调整
        loss_pos = distance_pos.min(dim=2)[0].mean(dim=1)  # (batch_size, num_concepts)
        loss_neg = F.relu(margin - distance_neg.min(dim=2)[0]).mean(dim=1)  # (batch_size, num_concepts)

        contrastive_loss = (loss_pos + loss_neg).mean()

        return contrastive_loss