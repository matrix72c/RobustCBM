import torch
import torch.nn as nn
import torchvision


class OldCBM(nn.Module):
    def __init__(
        self,
        base,
        num_attributes,
        num_classes,
        use_pretrained=False,
        use_adv="",
    ):
        super(OldCBM, self).__init__()

        if base == "resnet50":
            self.base = torchvision.models.resnet50(
                num_classes=num_attributes, pretrained=False
            )
        elif base == "vgg":
            self.base = None

        self.fc = nn.Sequential(
            nn.Linear(num_attributes, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )
        # self.fc = nn.Sequential(nn.Linear(num_attributes,num_classes))

    def forward(self, x):
        concept_pred = self.base(x)
        if self.use_adv == "image2label":
            return self.fc(concept_pred)
        return concept_pred, self.fc(concept_pred)
