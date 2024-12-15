import torch
from torch import nn
import torch.nn.functional as F
from model import CBM
from utils import calc_info_loss


class VCBM(CBM):
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
        vib_lambda: float = 0.1,
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
        self.base.fc = nn.Linear(self.base.fc.in_features, 2 * num_concepts)  # encoder

    def forward(self, x):
        statistics = self.base(x)
        std, mu = torch.chunk(statistics, 2, dim=1)
        concept_pred = mu + std * torch.randn_like(std)
        label_pred = self.classifier(concept_pred)
        return label_pred, concept_pred, mu, std**2

    def shared_step(self, img, label, concepts):
        label_pred, concept_pred, mu, var = self(img)
        concept_loss = F.binary_cross_entropy_with_logits(concept_pred, concepts)
        info_loss = calc_info_loss(mu, var)
        class_loss = F.cross_entropy(label_pred, label)
        loss = (
            class_loss
            + self.hparams.concept_weight * concept_loss
            + self.hparams.vib_lambda * info_loss
        )
        return loss, label_pred, concept_pred
