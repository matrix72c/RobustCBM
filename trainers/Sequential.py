import torch
import torch.optim as optim
from torchattacks import PGD, PGD_V2V

def Sequential(
    img,
    label,
    attr,
    model,
    model_base,
    label_loss_meter,
    label_acc_meter,
    attr_loss_meter,
    attr_acc_meter,
    optimizer_type,
    optimizer_args,
    scheduler_type,
    scheduler_args,
    loss_fn,
    attr_loss_fn,
    attr_loss_weight = 0.01,
    use_adv = "",
    use_noise = "",
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
    for name, param in model.named_parameters():
        if "fc" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    img, label, attr = img.cuda(), label.cuda(), attr.cuda()
    attr_losses = []
    if model_base == "inceptionv3":
        attr_pred, logits_pred = model.backbone(img)
        for i in range(attr_pred.shape[1]):
            attr_losses.append(
                attr_loss_fn[i](attr_pred[:, i], attr[:, i]) * 0.7
                + attr_loss_fn[i](logits_pred[:, i], attr[:, i]) * 0.3
            )
    else:
        attr_pred = model.backbone(img)
        for i in range(attr_pred.shape[1]):
            attr_losses.append(attr_loss_fn[i](attr_pred[:, i], attr[:, i]))
    attr_loss = sum(attr_losses)

    attr_optimizer = getattr(optim, optimizer_type)(
        model.backbone.parameters(), **optimizer_args
    )
    attr_scheduler = getattr(optim.lr_scheduler, scheduler_type)(
        attr_optimizer, **scheduler_args
    )
    attr_optimizer.zero_grad()
    attr_loss.backward()
    attr_optimizer.step()
    attr_scheduler.step()

    for name, param in model.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # re-calculate label pred
    if model_base == "inceptionv3":
        attr_pred, logits_pred = model.backbone(img)
    else:
        attr_pred = model.backbone(img)

    if "concept2label" in use_adv or "conceptpred2label" in use_adv:
        atk = PGD_V2V(model.fc, eps=5e-2, alpha=1e-2, steps=10, random_start=True)
        adv_attr = atk(attr, label).cuda() if "concept2label" in use_adv else atk(attr_pred, label).cuda()
        adv_label = label.clone().detach().cuda()
        attr_pred = torch.cat([attr.cuda() if "concept2label" in use_adv else attr_pred, adv_attr], dim=0).cuda()
        label = torch.cat([label, adv_label], dim=0).cuda()
        attr = torch.cat([attr, attr]).cuda()
    
    # 如果上方 if 未触发，label_pred 依赖 attr_pred 计算，而非 attr
    label_pred = model.fc(attr_pred)
    label_loss = loss_fn(label_pred, label)

    label_optimizer = getattr(optim, optimizer_type)(
        model.fc.parameters(), **optimizer_args
    )
    label_scheduler = getattr(optim.lr_scheduler, scheduler_type)(
        label_optimizer, **scheduler_args
    )
    label_optimizer.zero_grad()
    label_loss.backward()
    label_optimizer.step()
    label_scheduler.step()




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