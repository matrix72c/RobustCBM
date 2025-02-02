from attacks import attack

import torch
import torch.nn.functional as F


class apgdt(attack):

    def __init__(
        self,
        eps: float = 8 / 255,
        steps: int = 10,
        n_classes: int = 10,
        n_target_classes: int = 9,
        loss_fn: callable = F.cross_entropy,
        get_cls_fn: callable = lambda y_pred, y_gt: (y_pred, y_gt),
        norm: str = "Linf",
        n_restarts: int = 1,
        eot_iter: int = 1,
        rho: float = 0.75,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        self.eps = eps
        self.steps = steps
        self.target_class = None
        self.n_target_classes = n_target_classes
        self.loss_fn = loss_fn
        self.norm = norm
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.get_cls_fn = get_cls_fn
        self.n_restarts = n_restarts

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = (x[j - k : j] > x[j - k - 1 : j - 1]).sum(dim=0)

        return t <= k * k3

    def dlr_loss_targeted(self, x, y, y_target):
        x_sorted, ind_sorted = x.sort(dim=1)

        return -(
            x[torch.arange(x.shape[0]), y] - x[torch.arange(x.shape[0]), y_target]
        ) / (x_sorted[:, -1] - 0.5 * x_sorted[:, -3] - 0.5 * x_sorted[:, -4] + 1e-12)

    def attack_single_run(self, model, x, y):

        self.steps_2, self.steps_min, self.size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )

        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(x.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                x.device
            ).detach() * t / (
                t.reshape([t.shape[0], -1])
                .abs()
                .max(dim=1, keepdim=True)[0]
                .reshape([-1, 1, 1, 1])
            )
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(x.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                x.device
            ).detach() * t / ((t**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(self.clip_min, self.clip_max)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        o = model(x)
        label_pred, label_gt = self.get_cls_fn(o, y)
        y_target = label_pred.sort(dim=1)[1][:, -self.target_class]

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                o = model(x_adv)
                label_pred, label_gt = self.get_cls_fn(o, y)
                loss_indiv = self.dlr_loss_targeted(label_pred, label_gt, y_target)
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = label_pred.detach().max(1)[1] == label_gt
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = (
            self.eps
            * torch.ones([x.shape[0], 1, 1, 1]).to(x.device).detach()
            * torch.Tensor([2.0]).to(x.device).detach().reshape([1, 1, 1, 1])
        )
        x_adv_old = x_adv.clone()
        k = self.steps_2 + 0
        u = torch.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.zeros(loss_best.shape) == torch.zeros(
            loss_best.shape
        )

        for i in range(self.steps):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == "Linf":
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

                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size[0] * grad / (
                        (grad**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(x.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        self.clip_min,
                        self.clip_max,
                    )
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(x.device).detach(),
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12,
                        ),
                        self.clip_min,
                        self.clip_max,
                    )

                x_adv = x_adv_1 + 0.0

            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    o = model(x_adv)
                    label_pred, label_gt = self.get_cls_fn(o, y)
                    loss_indiv = self.dlr_loss_targeted(label_pred, label_gt, y_target)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            pred = label_pred.detach().max(1)[1] == label_gt
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                x_adv[(pred == 0).nonzero().squeeze()] + 0.0
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
                    k = max(k - self.size_decr, self.steps_min)

        return x_best, acc, loss_best, x_best_adv

    @torch.enable_grad()
    def attack(self, model, x, y):
        adv = x.clone().detach()
        o = model(adv)
        label_pred, label_gt = self.get_cls_fn(o, y)
        acc = label_pred.detach().max(1)[1] == label_gt

        for target_class in range(2, self.n_target_classes + 2):
            self.target_class = target_class
            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone() if not isinstance(y, tuple) else (
                        y[0][ind_to_fool].clone(),
                        y[1][ind_to_fool].clone(),
                    )
                    (
                        best_curr,
                        acc_curr,
                        loss_curr,
                        adv_curr,
                    ) = self.attack_single_run(model, x_to_fool, y_to_fool)
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

        return adv
