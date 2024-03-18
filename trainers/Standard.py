import torch
import torch.optim as optim


def Standard(
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
    attr_loss_weight,
):
    img, label, attr = img.cuda(), label.cuda(), attr.cuda()
    attr_losses = []
    if model_base == "inceptionv3":
        attr_logits_pred, label_pred = model(img)
        attr_pred, logits_pred = attr_logits_pred
        for i in range(attr_pred.shape[1]):
            attr_losses.append(
                attr_loss_fn[i](attr_pred[:, i], attr[:, i]) * 0.7
                + attr_loss_fn[i](logits_pred[:, i], attr[:, i]) * 0.3
            )
    else:
        attr_pred, label_pred = model(img)
        for i in range(attr_pred.shape[1]):
            attr_losses.append(attr_loss_fn[i](attr_pred[:, i], attr[:, i]))
    attr_loss = sum(attr_losses)
    label_loss = loss_fn(label_pred, label)

    optimizer = getattr(optim, optimizer_type)(model.parameters(), **optimizer_args)
    scheduler = getattr(optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_args)
    optimizer.zero_grad()
    label_loss.backward()
    optimizer.step()
    scheduler.step()

    label_pred = torch.argmax(label_pred, dim=1)
    correct = torch.sum(label_pred == label).int().sum().item()
    num = len(label)
    label_loss_meter.update(label_loss.item(), num)
    label_acc_meter.update(correct / num, num)
    attr_pred = torch.sigmoid(attr_pred).ge(0.5)
    attr_correct = torch.sum(attr_pred == attr).int().sum().item()
    attr_num = attr.shape[0] * attr.shape[1]
    attr_loss_meter.update(attr_loss.item(), attr_num)
    attr_acc_meter.update(attr_correct / attr_num, attr_num)
