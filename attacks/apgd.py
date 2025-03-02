from attacks import Attack

import torch
import torch.nn as nn


def L2_norm(x, keepdim=False):
    z = (x**2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1] * (len(x.shape) - 1))
    return z


class APGD(Attack):

    def __init__(
        self,
        eps: float = 8 / 255,
        steps: int = 10,
        norm: str = "Linf",
        eot_iter: int = 1,
        threshold: float = 0.75,
        n_restarts: int = 1,
        loss: str = "ce",
        **kwargs
    ):
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.eot_iter = eot_iter
        self.threshold = threshold
        self.n_restarts = n_restarts
        self.loss = loss

        self.n_iter_2 = max(int(0.22 * self.steps), 1)
        self.n_iter_min = max(int(0.06 * self.steps), 1)
        self.size_decr = max(int(0.03 * self.steps), 1)

    def init_hyperparam(self, x):
        self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def normalize(self, x):
        if self.norm == "Linf":
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == "L2":
            t = (x**2).view(x.shape[0], -1).sum(-1).sqrt()

        elif self.norm == "L1":
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def attack_single(self, x, y):
        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x).detach() * self.normalize(t)
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x).detach() * self.normalize(t)

        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]]).to(self.device)
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]]).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if self.loss == "ce":
            criterion_indiv = nn.CrossEntropyLoss(reduction="none")
        elif self.loss == "dlr-targeted":
            criterion_indiv = self.dlr_loss_targeted

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= self.eot_iter
        grad_best = grad.clone()
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = (
            2.0
            * self.eps
            * torch.ones([x.shape[0], *([1] * self.ndims)]).to(self.device).detach()
        )
        counter = 0
        x_adv_old = x_adv.clone()
        k = self.n_iter_2

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

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
                        0.0,
                        1.0,
                    )
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        0.0,
                        1.0,
                    )

                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(
                        x
                        + self.normalize(x_adv_1 - x)
                        * torch.min(
                            self.eps * torch.ones_like(x).detach(),
                            L2_norm(x_adv_1 - x, keepdim=True),
                        ),
                        0.0,
                        1.0,
                    )
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x
                        + self.normalize(x_adv_1 - x)
                        * torch.min(
                            self.eps * torch.ones_like(x).detach(),
                            L2_norm(x_adv_1 - x, keepdim=True),
                        ),
                        0.0,
                        1.0,
                    )
                x_adv = x_adv_1 + 0.0

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= self.eot_iter
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.0

            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter += 1
                if counter == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps, i, k, loss_best, k3=self.threshold
                    )
                    fl_reduce_no_impr = (1.0 - reduced_last_check) * (
                        loss_best_last_check >= loss_best
                    ).float()
                    fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    k = max(k - self.size_decr, self.n_iter_min)
                    counter = 0

        return (x_best, acc, loss_best, x_best_adv)

    def attack(self, model, x, y):
        self.model = model
        self.init_hyperparam(x)
        y_pred = model(x).argmax(1)
        adv = x.clone().detach()
        acc = y_pred == y
        loss = -1e10 * torch.ones_like(acc).float()

        for _ in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze(1)
            if len(ind_to_fool) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                best_curr, acc_curr, loss_curr, adv_curr = self.attack_single(
                    x_to_fool, y_to_fool
                )
                ind_curr = (acc_curr == 0).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
        return adv


class APGDT(APGD):
    def __init__(self, num_classes: int = 10, loss: str = "dlr-targeted", **kwargs):
        super().__init__(loss=loss, **kwargs)
        self.y_target = None
        self.n_target_classes = num_classes - 1

    def dlr_loss_targeted(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x[u, self.y_target]) / (
            x_sorted[:, -1] - 0.5 * (x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12
        )

    def attack(self, model, x, y):
        self.model = model
        self.init_hyperparam(x)
        y_pred = model(x).argmax(1)
        adv = x.clone().detach()
        acc = y_pred == y

        for target_class in range(2, self.n_target_classes + 2):
            for _ in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze(1)
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    output = self.model(x_to_fool)
                    self.y_target = output.sort(dim=1)[1][:, -target_class]

                    best_curr, acc_curr, loss_curr, adv_curr = self.attack_single(
                        x_to_fool, y_to_fool
                    )
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

        return adv
