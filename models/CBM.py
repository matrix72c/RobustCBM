import os
import torch
import torch.nn as nn
from torchattacks import PGD, PGD_V2V
import torchvision

from utils import AverageMeter


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
    base: base model: resnet50, inceptionv3.
    num_attributes: number of attributes.
    num_classes: number of classes.
    use_pretrained: use pretrained model.
    """

    def __init__(self, conf):
        super(CBM, self).__init__()
        self.conf = conf
        base = conf["base"]
        num_attributes = conf["num_attributes"]
        num_classes = conf["num_classes"]
        use_pretrained = conf["use_pretrained"]

        # create model
        if base == "resnet50":
            if use_pretrained is True:
                self.backbone = torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.DEFAULT
                )
                self.backbone.fc = nn.Linear(2048, num_attributes)
            elif use_pretrained is False:
                self.backbone = torchvision.models.resnet50(
                    num_classes=num_attributes, weights=None
                )
            else:
                self.backbone = torchvision.models.resnet50(weights=None)
                model_weights_path = use_pretrained
                self.backbone.load_state_dict(
                    torch.load(model_weights_path, map_location="cuda")
                )
                self.backbone.fc = nn.Linear(2048, num_attributes)
        elif base == "inceptionv3":
            if use_pretrained is True:
                self.backbone = torchvision.models.inception_v3(
                    weights=torchvision.models.Inception_V3_Weights.DEFAULT,
                    aux_logits=False,
                )
                self.backbone.fc = nn.Linear(2048, num_attributes)
            elif use_pretrained is False:
                self.backbone = torchvision.models.inception_v3(
                    num_classes=num_attributes, aux_logits=False, weights=None
                )
            else:
                self.backbone = torchvision.models.inception_v3(
                    aux_logits=False, weights=None
                )
                model_weights_path = use_pretrained
                self.backbone.load_state_dict(
                    torch.load(model_weights_path, map_location="cuda")
                )
                self.backbone.fc = nn.Linear(2048, num_attributes)
        else:
            raise ValueError("Unknown base model")
        self.fc = FC(num_attributes, num_classes)

        # load checkpoint
        if ".pth" in conf["checkpoint"] and os.path.exists(conf["checkpoint"]):
            self.load_state_dict(torch.load(conf["checkpoint"]))
            print("Load model from", conf["checkpoint"])

        # set training mode
        self.use_adv = conf["use_adv"]
        self.use_noise = conf["use_noise"]
        self.atk_mode = False
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.attr_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), **conf["optimizer_args"])
        self.backbone_optimizer = torch.optim.SGD(self.backbone.parameters(), **conf["optimizer_args"])
        self.fc_optimizer = torch.optim.SGD(self.fc.parameters(), **conf["fc_optimizer_args"])

    def forward(self, x):
        attr_pred = self.backbone(x)
        label_pred = self.fc(attr_pred)
        if self.atk_mode:
            return label_pred
        else:
            return attr_pred, label_pred

    def Joint(self, loader):
        label_loss_meter = AverageMeter()
        label_acc_meter = AverageMeter()
        attr_loss_meter = AverageMeter()
        attr_acc_meter = AverageMeter()
        for img, label, attr in loader:
            if "image2label" in self.use_adv:
                self.atk_mode = True
                atk = PGD(self, eps=5 / 255, alpha=2 / 225, steps=2, random_start=True)
                atk.set_normalization_used(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
                adv_img = atk(img, label).cpu()
                adv_label = label.clone().detach().cpu()
                adv_attr = attr.clone().detach().cpu()
                img = torch.cat([img, adv_img], dim=0)
                label = torch.cat([label, adv_label], dim=0)
                attr = torch.cat([attr, adv_attr], dim=0)
                self.atk_mode = False
            if self.use_noise == "image":
                noise = torch.empty_like(img).uniform_(-5 / 255, 5 / 255)
                noise_img = img.clone().detach() + noise
                noise_attr = attr.clone().detach()
                noise_label = label.clone().detach()
                img = torch.cat([img, noise_img], dim=0)
                label = torch.cat([label, noise_label], dim=0)
                attr = torch.cat([attr, noise_attr], dim=0)
            img, label, attr = img.cuda(), label.cuda(), attr.cuda()
            attr_pred, label_pred = self(img)
            attr_loss = self.attr_loss_fn(attr_pred, attr)
            label_loss = self.loss_fn(label_pred, label)

            (label_loss + attr_loss * self.conf["attr_loss_weight"]).backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            label_pred = torch.argmax(label_pred, dim=1)
            attr_pred = torch.sigmoid(attr_pred).ge(0.5)
            attr_correct = torch.sum(attr_pred == attr).int().sum().item()
            correct = torch.sum(label_pred == label).int().sum().item()
            num = len(label)
            attr_num = attr.shape[0] * attr.shape[1]
            label_loss_meter.update(label_loss.item(), num)
            label_acc_meter.update(correct / num, num)
            attr_loss_meter.update(attr_loss.item(), attr_num)
            attr_acc_meter.update(attr_correct / attr_num, attr_num)
        return {
            "label_loss": label_loss_meter.avg,
            "label_acc": label_acc_meter.avg,
            "attr_loss": attr_loss_meter.avg,
            "attr_acc": attr_acc_meter.avg,
        }

    def Sequential(self, loader):
        label_loss_meter = AverageMeter()
        label_acc_meter = AverageMeter()
        attr_loss_meter = AverageMeter()
        attr_acc_meter = AverageMeter()
        for img, label, attr in loader:
            if "image2label" in self.use_adv:
                self.atk_mode = True
                atk = PGD(self, eps=5 / 255, alpha=2 / 225, steps=2, random_start=True)
                atk.set_normalization_used(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
                adv_img = atk(img, label).cpu()
                adv_label = label.clone().detach().cpu()
                adv_attr = attr.clone().detach().cpu()
                img = torch.cat([img, adv_img], dim=0)
                label = torch.cat([label, adv_label], dim=0)
                attr = torch.cat([attr, adv_attr], dim=0)
                self.atk_mode = False
            if self.use_noise == "image":
                noise = torch.empty_like(img).uniform_(-5 / 255, 5 / 255)
                noise_img = img.clone().detach() + noise
                noise_attr = attr.clone().detach()
                noise_label = label.clone().detach()
                img = torch.cat([img, noise_img], dim=0)
                label = torch.cat([label, noise_label], dim=0)
                attr = torch.cat([attr, noise_attr], dim=0)
            img, label, attr = img.cuda(), label.cuda(), attr.cuda()
            attr_pred = self.backbone(img)
            attr_loss = self.attr_loss_fn(attr_pred, attr)
            attr_loss.backward()
            self.backbone_optimizer.step()
            self.backbone_optimizer.zero_grad()

            attr_pred = self.backbone(img)
            if "concept2label" in self.use_adv:
                atk = PGD_V2V(self.fc, eps=self.conf["adv_v2v_eps"], alpha=5e-2, steps=10, random_start=True)
                adv_attr = atk(attr_pred, label).cuda()
                adv_label = label.clone().detach().cuda()
                attr_pred = torch.cat([attr_pred, adv_attr], dim=0).cuda()
                label = torch.cat([label, adv_label], dim=0).cuda()
                attr = torch.cat([attr, attr]).cuda()

            if self.use_noise == "concept":
                noise = torch.empty_like(attr).uniform_(-self.conf["noise_eps"], self.conf["noise_eps"])
                noise_attr = attr.clone().detach() + noise
                noise_label = label.clone().detach()
                attr = torch.cat([attr, noise_attr], dim=0)
                label = torch.cat([label, noise_label], dim=0)
            
            label_pred = self.fc(attr_pred)
            label_loss = self.loss_fn(label_pred, label)
            label_loss.backward()
            self.fc_optimizer.step()
            self.fc_optimizer.zero_grad()

            attr_pred = torch.sigmoid(attr_pred).ge(0.5)
            attr_correct = torch.sum(attr_pred == attr).int().sum().item()
            attr_num = attr.shape[0] * attr.shape[1]
            attr_loss_meter.update(attr_loss.item(), attr_num)
            attr_acc_meter.update(attr_correct / attr_num, attr_num)
            label_pred = torch.argmax(label_pred, dim=1)
            correct = torch.sum(label_pred == label).int().sum().item()
            num = len(label)
            label_loss_meter.update(label_loss.item(), num)
            label_acc_meter.update(correct / num, num)
        
        return {
            "label_loss": label_loss_meter.avg,
            "label_acc": label_acc_meter.avg,
            "attr_loss": attr_loss_meter.avg,
            "attr_acc": attr_acc_meter.avg,
        }
    
    def run_epoch(self, loader):
        self.train()
        if self.conf["mode"] == "Joint":
            return self.Joint(loader)
        elif self.conf["mode"] == "Sequential":
            return self.Sequential(loader)
        else:
            raise ValueError("Unknown mode")