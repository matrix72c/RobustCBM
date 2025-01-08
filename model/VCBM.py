import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from mtl import get_grad, gradient_ordered
from utils import calc_info_loss, initialize_weights


class VIB(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2 * output_dim).apply(initialize_weights)

    def forward(self, x):
        statistics = self.fc(x)
        std, mu = torch.chunk(statistics, 2, dim=1)
        x = mu + std * torch.randn_like(std)
        loss = calc_info_loss(mu, std**2)
        return x, loss


class VCBM(CBM):
    def __init__(
        self,
        vib_lambda: float = 0.1,
        use_gate: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base.fc = nn.Linear(
            self.base.fc.in_features, 2 * self.num_concepts
        )  # encoder

    def forward(self, x):
        statistics = self.base(x)
        std, mu = torch.chunk(statistics, 2, dim=1)
        concept_pred = mu + std * torch.randn_like(std)
        c = torch.sigmoid(std) * 2.0 if self.hparams.use_gate else 1.0
        label_pred = self.classifier(concept_pred * c)
        return label_pred, concept_pred, mu, std**2

    def train_step(self, img, label, concepts):
        label_pred, concept_pred, mu, var = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_pred, concepts, weight=self.dm.imbalance_weights.to(self.device)
        )
        info_loss = calc_info_loss(mu, var)
        label_loss = F.cross_entropy(label_pred, label)
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.vib_lambda * info_loss
        )
        if self.hparams.mtl_mode == "normal":
            self.manual_backward(loss)
        else:
            g0 = get_grad(label_loss, self)
            g1 = get_grad(concept_loss, self)
            g2 = get_grad(info_loss, self)
            g = gradient_ordered(g1, g2)
            g = gradient_ordered(g0, g)
            for name, param in self.named_parameters():
                param.grad = g[name]
        return loss
