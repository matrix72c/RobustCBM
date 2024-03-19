import torch
import torch.nn as nn
import torchvision


class FC(nn.Module):
    """
    FC: Fully Connected Layer
    """

    def __init__(self, num_attributes, num_classes):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_attributes, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


class CBM(nn.Module):
    """
    CBM: Concept Based Model

    mode:
    Independent: X->C and C->Y are trained independently.
    Sequential: X->C and C->Y are trained sequentially.
    Joint: X->C and C->Y are trained jointly.
    Standard: Directly train X->Y.
    In independent and sequential mode, the CBM only return the concept model.

    Args:
    mode: Independent, Joint, Sequential, Standard.
    base: base model: resnet50, inceptionv3.
    num_attributes: number of attributes.
    num_classes: number of classes.
    attr_loss_weight: weight ratio of attribute loss.
    is_training: training or testing/attack.
    use_pretrained: use pretrained model.
    attack: attack backbone or fc layer.
    """

    def __init__(
        self,
        base,
        num_attributes,
        num_classes,
        use_pretrained=False,
        use_adv=False,
    ):
        super(CBM, self).__init__()

        if base == "resnet50":
            if use_pretrained is True:
                self.model = torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.DEFAULT
                )
                self.model.fc = nn.Linear(2048, num_attributes)
            elif use_pretrained is False:
                self.model = torchvision.models.resnet50(
                    num_classes=num_attributes, weights=None
                )
            else:
                self.model = torchvision.models.resnet50(weights=None)
                model_weights_path = use_pretrained
                self.model.load_state_dict(torch.load(model_weights_path, map_location='cuda'))
                self.model.fc = nn.Linear(2048, num_attributes)
        elif base == "inceptionv3":
            if use_pretrained is True:
                self.model = torchvision.models.inception_v3(
                    weights=torchvision.models.Inception_V3_Weights.DEFAULT
                )
                self.model.fc = nn.Linear(2048, num_attributes)
            elif use_pretrained is False:
                self.model = torchvision.models.inception_v3(
                    num_classes=num_attributes, aux_logits=True, weights=None
                )
            else:
                self.model = torchvision.models.inception_v3(aux_logits=True, weights=None)
                model_weights_path = use_pretrained
                self.model.load_state_dict(torch.load(model_weights_path, map_location='cuda'))
                self.model.fc = nn.Linear(2048, num_attributes)
        else:
            raise ValueError("Unknown base model")
        self.use_adv = use_adv
        self.fc = FC(num_attributes, num_classes)

    def forward(self, x):
        attr_pred = self.model(x)
        label_pred = self.fc(
            attr_pred if not isinstance(attr_pred, tuple) else attr_pred[0]
        )
        if self.use_adv == "label":
            return label_pred
        elif self.use_adv == "concept":
            return attr_pred
        else:
            return attr_pred, label_pred
