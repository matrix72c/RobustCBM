from attacks import attack

import torch
import torch.nn.functional as F


class apgd(attack):

    def __init__(
        self,
        eps: float = 8 / 255,
        steps: int = 10,
        loss_fn: callable = F.cross_entropy,
        eot_iter: int = 1,
        rho: float = 0.75,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        get_cls_fn: callable = lambda y_pred, y_gt: (y_pred, y_gt),
    ):
        self.eps = eps
        self.steps = steps
        self.loss_fn = loss_fn
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.get_cls_fn = get_cls_fn

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = (x[j - k : j] > x[j - k - 1 : j - 1]).sum(dim=0)

        return t <= k * k3

    @torch.enable_grad()
    def attack(self, model, x, y):
        steps_2, steps_min, size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )

        t = 2 * torch.rand(x.shape, device=x.device).detach() - 1
        x_adv = x.clone().detach() + self.eps * torch.ones(
            [x.shape[0], 1, 1, 1], device=x.device
        ).detach() * t / (
            t.reshape([t.shape[0], -1])
            .abs()
            .max(dim=1, keepdim=True)[0]
            .reshape([-1, 1, 1, 1])
        )
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])

        x_adv.requires_grad = True
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                o = model(x_adv)
                loss_indiv = self.loss_fn(o, y)
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        loss_best = loss_indiv.detach().clone()

        step_size = (
            self.eps
            * torch.ones([x.shape[0], 1, 1, 1], device=x.device).detach()
            * torch.Tensor([2.0]).to(x.device).detach().reshape([1, 1, 1, 1])
        )
        x_adv_old = x_adv.clone()
        k = steps_2 + 0
        u = torch.arange(x.shape[0]).to(x.device)
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.zeros_like(loss_best, dtype=torch.bool)

        for i in range(self.steps):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(
                    torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                    self.clip_min,
                    self.clip_max,
                )
                x_adv_1 = torch.clamp(
                    torch.min(
                        torch.max(
                            x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                            x - self.eps,
                        ),
                        x + self.eps,
                    ),
                    self.clip_min,
                    self.clip_max,
                )
                x_adv = x_adv_1 + 0.0

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    o = model(x_adv)
                    loss_indiv = self.loss_fn(o, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)
            label_pred, label_gt = self.get_cls_fn(o, y)
            label_pred = label_pred.detach().max(1)[1] == label_gt
            x_best_adv[(label_pred == 0).nonzero().squeeze()] = (
                x_adv[(label_pred == 0).nonzero().squeeze()] + 0.0
            )

            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach(),
                        i,
                        k,
                        loss_best.detach(),
                        k3=self.thr_decr,
                    ).to(x.device)
                    fl_reduce_no_impr = (~reduced_last_check) * (
                        loss_best_last_check >= loss_best
                    )
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        step_size[u[fl_oscillation]] /= 2.0

                        fl_oscillation = fl_oscillation.nonzero().squeeze()

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = max(k - size_decr, steps_min)

        return x_best_adv
