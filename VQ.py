import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizeEMA(nn.Module):
    def __init__(self, embedding_dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.embed = nn.Embedding(n_embed, embedding_dim)
        self.embed.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", self.embed.weight.data.clone())

        self.decay = decay
        self.eps = eps

    def forward(self, z_e):
        B, N, E = z_e.shape  # Batch, Num_concepts, Embedding_dim
        flatten = z_e.reshape(-1, self.embedding_dim)  # (B*N, E)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)  # (B*N, 1)
            - 2 * flatten @ self.embed.weight.t()  # (B*N, E)
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()  # (1, E)
        )  # (B*N, E)

        _, embed_ind = (-dist).max(1)  # choose the nearest neighboor
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(
            flatten.dtype
        )  # (B*N, E)
        embed_ind = embed_ind.view(B, N)  # (B, N)

        z_q = self.embed_code(embed_ind)  # B, N, E

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = (flatten.transpose(0, 1) @ embed_onehot).transpose(0, 1)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.weight.data.copy_(embed_normalized)

        diff = (z_q.detach() - z_e).pow(2).mean()

        z_q = z_e + (z_q - z_e).detach()
        return z_q, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


if __name__ == "__main__":
    pass
