import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from mtl import mtl
from utils import calc_info_loss


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
        concept_loss = F.binary_cross_entropy_with_logits(concept_pred, concepts)
        info_loss = calc_info_loss(mu, var)
        label_loss = F.cross_entropy(label_pred, label)
        loss = (
            label_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.vib_lambda * info_loss
        )
        if self.hparams.mtl_mode == "normal":
            self.manual_backward(loss)
            self.optimizers().step()
        else:
            mtl(
                [label_loss, concept_loss, info_loss],
                self,
                self.hparams.mtl_mode,
            )
        return loss
