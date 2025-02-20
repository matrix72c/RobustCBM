from attacks import Attack

import torch
import torch.nn.functional as F


class Apgd(Attack):

    def __init__(
        self,
        eps: float = 8 / 255,
        steps: int = 10,
        eot_iter: int = 1,
        threshold: float = 0.75,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        **kwargs
    ):
        self.eps = eps
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

    def forward(self, model, x, y):
        grad = torch.zeros_like(x)
        x.requires_grad = True
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                o = model(x)
                loss_indiv = F.cross_entropy(o, y, reduction="none")
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x])[0].detach()

        grad /= float(self.eot_iter)
        return grad, loss_indiv.detach(), o

    def attack(self, model, x, y):
        window, window_min, window_delta = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )

        delta = torch.rand_like(x, device=x.device) * 2 - 1
        delta = (
            self.eps
            * delta
            / delta.reshape(x.size(0), -1)
            .abs()
            .max(dim=1, keepdim=True)[0]
            .reshape([-1, 1, 1, 1])  # normalize
        )
        x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        grad, loss_best, _ = self.forward(model, x_adv, y)

        step_size = self.eps * torch.ones_like(x).detach() * 2.0
        counter = 0
        indices = torch.arange(x.shape[0]).to(x.device)
        x_adv_prev = x_adv.clone()
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.zeros_like(loss_best, dtype=torch.bool)

        for i in range(self.steps):
            with torch.no_grad():
                # iterate x_adv
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
                x_best_adv[(label_pred == 0).nonzero().squeeze()] = (
                    x_adv[(label_pred == 0).nonzero().squeeze()] + 0.0
                )

                # update x_best
                loss_steps[i] = loss
                mask = (loss > loss_best).nonzero().squeeze()
                x_best[mask] = x_adv[mask].clone()
                loss_best[mask] = loss[mask]

                # update step_size
                counter += 1

                if counter == window:
                    fl_oscillation = self.check_oscillation(loss_steps, i, window).to(x.device)
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

                    counter = 0
                    window = max(window - window_delta, window_min)

        return x_best_adv
