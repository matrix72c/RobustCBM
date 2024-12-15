import torch
from torch import nn
from model import CBM
from utils import initialize_weights


class resCBM(CBM):
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
        self.base.fc = nn.Linear(
            self.base.fc.in_features, num_concepts + num_classes
        ).apply(initialize_weights)

    def forward(self, x):
        logits = self.base(x)
        concept_pred = logits[:, : self.num_concepts]
        label_res = logits[:, self.num_concepts :]
        label_pred = self.classifier(concept_pred) + label_res
        return label_pred, concept_pred
