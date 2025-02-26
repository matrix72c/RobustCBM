from attacks import Attack

import torch
import torch.nn.functional as F


class Apgdt(Attack):

    def __init__(
        self,
        eps: float = 8 / 255,
        num_classes: int = 10,
        steps: int = 10,
        eot_iter: int = 1,
        threshold: float = 0.75,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        **kwargs
    ):
        self.eps = eps
        self.num_classes = num_classes
        self.steps = steps
        self.eot_iter = eot_iter
        self.threshold = threshold
        self.clip_min = clip_min
        self.clip_max = clip_max

    def check_oscillation(self, losses, idx, window):
        if idx < window + 1:
            return torch.zeros_like(losses[0], dtype=torch.bool)
        return (losses[idx - window : idx] > losses[idx - window - 1 : idx - 1]).sum(
            0
        ) <= window * self.threshold

    def dlr_loss(self, x, y):
        x_sorted, _ = x.sort(dim=1)
        return -(
            x[torch.arange(x.shape[0], device=x.device), y]
            - x[torch.arange(x.shape[0], device=x.device), self.y_target]
        ) / (x_sorted[:, -1] - 0.5 * x_sorted[:, -3] - 0.5 * x_sorted[:, -4] + 1e-12)

    def forward(self, model, x, y):
        grad = torch.zeros_like(x)
        x.requires_grad = True
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                o = model(x)
                loss_indiv = self.dlr_loss(o, y)
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x])[0].detach()

        grad /= float(self.eot_iter)
        return grad, loss_indiv.detach(), o

    def attack_single(self, model, x, y):
        window, window_min, window_delta = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )

        delta = torch.rand_like(x, device=x.device) * 2 - 1
        delta = (
            self.eps
            * delta
            / delta.reshape([x.size(0), -1])
            .abs()
            .max(dim=1, keepdim=True)[0]
            .reshape([-1, 1, 1, 1])  # normalize
        )
        x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max)

        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])

        label_pred = model(x_adv)
        self.y_target = label_pred.sort(dim=1)[1][:, -self.target_class]

        grad, loss_best, _ = self.forward(model, x_adv, y)
        grad_best = grad.clone()

        step_size = self.eps * torch.ones_like(x).detach() * 2.0
        counter = 0
        indices = torch.arange(x.shape[0]).to(x.device)
        x_adv_prev = x_adv.clone()
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best, dtype=torch.bool)

        for i in range(self.steps):
            with torch.no_grad():
                # iterate x_adv
                x_adv = x_adv.detach()
                d = x_adv - x_adv_prev
                x_adv_prev = x_adv.clone()
                alpha = 0.75 if i > 0 else 1.0
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(
                    torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                    self.clip_min,
                    self.clip_max,
                )
                x_adv = torch.clamp(
                    torch.min(
                        torch.max(
                            x_adv + (x_adv_1 - x_adv) * alpha + d * (1 - alpha),
                            x - self.eps,
                        ),
                        x + self.eps,
                    ),
                    self.clip_min,
                    self.clip_max,
                )

                # update x_best_adv
                grad, loss, label_pred = self.forward(model, x_adv, y)
                label_pred = label_pred.max(1)[1] == y
                x_best_adv[(label_pred == 0).nonzero().squeeze()] = x_adv[
                    (label_pred == 0).nonzero().squeeze()
                ]

                # update x_best
                loss_steps[i] = loss
                mask = (loss > loss_best).nonzero().squeeze()
                x_best[mask] = x_adv[mask].clone()
                loss_best[mask] = loss[mask]
                grad_best[mask] = grad[mask].clone()

                # update step_size
                counter += 1

                if counter == window:
                    fl_oscillation = self.check_oscillation(loss_steps, i, window).to(
                        x.device
                    )
                    fl_reduce_no_impr = (~reduced_last_check) * (
                        loss_best_last_check >= loss_best
                    )
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        step_size[indices[fl_oscillation]] /= 2.0

                        fl_oscillation = fl_oscillation.nonzero().squeeze()

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter = 0
                    window = max(window - window_delta, window_min)

        return x_best_adv

    def attack(self, model, x, y):
        x_adv = x.clone()
        label_pred = model(x_adv)
        acc = label_pred.max(1)[1] == y

        for target_class in range(2, self.num_classes + 1):
            self.target_class = target_class
            idx_to_fool = acc.nonzero().squeeze(1)
            if len(idx_to_fool) == 0:
                idx_to_fool = idx_to_fool.unsqueeze(0)
            if idx_to_fool.numel() != 0:
                x_to_fool = x[idx_to_fool].clone()
                y_to_fool = y[idx_to_fool].clone()
                adv_curr = self.attack_single(model, x_to_fool, y_to_fool)
                label_pred = model(adv_curr)
                acc_curr = label_pred.max(1)[1] == y_to_fool
                ind_curr = (acc_curr == 0).nonzero().squeeze()
                acc[idx_to_fool[ind_curr]] = 0
                x_adv[idx_to_fool[ind_curr]] = adv_curr[ind_curr]
        return x_adv
