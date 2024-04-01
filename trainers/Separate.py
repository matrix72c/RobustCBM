import torch
import torch.optim as optim
from torchattacks import PGD, PGD_V2V

def image2concept(
    img,
    label,
    attr,
    model,
    model_base,
    label_loss_meter,
    label_acc_meter,
    attr_loss_meter,
    attr_acc_meter,
    backbone_optimizer,
    fc_optimizer,
    loss_fn,
    attr_loss_fn,
    attr_loss_weight = 1,
    use_adv = "",
    use_noise = "",
    adv_v2v_eps = 0.3,
    noise_eps = 0.3,
):
    if "image2concept" in use_adv:
        model.use_adv = use_adv
        atk = PGD(model, eps=5 / 255, alpha=2 / 225, steps=2, random_start=True)
        atk.set_normalization_used(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        adv_img = atk(img, label).cpu()
        adv_label = label.clone().detach().cpu()
        adv_attr = attr.clone().detach().cpu()
        img = torch.cat([img, adv_img], dim=0)
        label = torch.cat([label, adv_label], dim=0)
        attr = torch.cat([attr, adv_attr], dim=0)
        model.use_adv = ""
    if use_noise == "image":
        noise = torch.empty_like(img).uniform_(-5 / 255, 5 / 255)
        noise_img = img.clone().detach() + noise
        noise_attr = attr.clone().detach()
        noise_label = label.clone().detach()
        img = torch.cat([img, noise_img], dim=0)
        label = torch.cat([label, noise_label], dim=0)
        attr = torch.cat([attr, noise_attr], dim=0)
    img, label, attr = img.cuda(), label.cuda(), attr.cuda()
    attr_pred = model.backbone(img)
    attr_loss = attr_loss_fn(attr_pred, attr)

    attr_loss.backward()
    backbone_optimizer.step()
    backbone_optimizer.zero_grad()

    attr_pred = torch.sigmoid(attr_pred).ge(0.5)
    attr_correct = torch.sum(attr_pred == attr).int().sum().item()
    attr_num = attr.shape[0] * attr.shape[1]
    attr_loss_meter.update(attr_loss.item(), attr_num)
    attr_acc_meter.update(attr_correct / attr_num, attr_num)

def concept2label(
    img,
    label,
    attr,
    model,
    model_base,
    label_loss_meter,
    label_acc_meter,
    attr_loss_meter,
    attr_acc_meter,
    backbone_optimizer,
    fc_optimizer,
    loss_fn,
    attr_loss_fn,
    attr_loss_weight = 0.01,
    use_adv = "",
    use_noise = "",
    adv_v2v_eps = 0.3,
    noise_eps = 0.3,
    seperate_mode = "combined",
):
    img, label, attr = img.cuda(), label.cuda(), attr.cuda()

    if seperate_mode == "combined":
        if model_base == "inceptionv3":
            attr_pred, logits_pred = model.backbone(img)
        else:
            attr_pred = model.backbone(img)
    else:
        attr_pred = attr

    if "concept2label" in use_adv:
        atk = PGD_V2V(model.fc, eps=adv_v2v_eps, alpha=5e-2, steps=10, random_start=True)
        adv_attr = atk(attr_pred, label).cuda()
        adv_label = label.clone().detach().cuda()
        attr_pred = torch.cat([attr_pred, adv_attr], dim=0).cuda()
        label = torch.cat([label, adv_label], dim=0).cuda()

    if use_noise == "concept":
        noise = torch.empty_like(attr_pred).uniform_(-noise_eps, noise_eps)
        noise_attr = attr_pred.clone().detach() + noise
        noise_label = label.clone().detach()
        attr = torch.cat([attr_pred, noise_attr], dim=0)
        label = torch.cat([label, noise_label], dim=0)
    
    label_pred = model.fc(attr_pred)
    label_loss = loss_fn(label_pred, label)

    label_loss.backward()
    fc_optimizer.step()
    fc_optimizer.zero_grad()

    label_pred = torch.argmax(label_pred, dim=1)
    correct = torch.sum(label_pred == label).int().sum().item()
    num = len(label)
    label_loss_meter.update(label_loss.item(), num)
    label_acc_meter.update(correct / num, num)