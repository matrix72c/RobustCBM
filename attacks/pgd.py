import torch
from typing import List, Union, Literal
from torch.cuda.amp.autocast_mode import autocast


class PGD:
    def __init__(
        self,
        predict,
        loss_fn,
        eps=8.0 / 255,
        clip_min=0.0,
        clip_max=1.0,
        nb_iters=10,
        eps_iter=2.0 / 255,
        rand_init: bool = True,
        targeted: bool = False,
        loss_scale: float = 1.0,
        params_switch_grad_req: List[torch.Tensor] = [],
    ):
        self.predict = predict
        self.loss_fn = loss_fn
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.nb_iters = nb_iters
        self.eps_iter = eps_iter
        self.targeted = targeted
        self.rand_init = rand_init
        self.loss_scale = loss_scale
        self.params_switch_grad_req = params_switch_grad_req

    @torch.enable_grad()
    @torch.inference_mode(False)
    def perturb(self, x: torch.Tensor, y: torch.Tensor, cst_init: torch.Tensor = None):
        # 创建干净的拷贝
        x = x.detach().clone()
        y = y.detach().clone()

        # 设置初始delta
        if cst_init is not None:
            start_delta = cst_init.detach().clone()
        else:
            if not self.rand_init:
                start_delta = torch.zeros_like(x)
            else:
                start_delta = torch.rand_like(x) * 2 * self.eps - self.eps
                start_delta = (
                    torch.clamp(x + start_delta, self.clip_min, self.clip_max) - x
                )
        delta = start_delta.detach().clone().requires_grad_(True)

        # 切换模型参数的requires_grad属性
        for param in self.params_switch_grad_req:
            param.requires_grad_(False)

        # 循环迭代，求解对抗样本
        for _ in range(self.nb_iters):
            self._iter(delta, x, y, self.eps_iter)

        for param in self.params_switch_grad_req:
            param.requires_grad_(True)
        return (delta + x).data

    def _iter(self, delta, x, y, eps_iter):
        adv = x + delta
        # 求解损失
        loss = self.loss_fn(self.predict(adv), y)
        # 求解关于输入中adv（delta）的梯度
        with autocast(enabled=False):
            (loss * self.loss_scale).backward()
        grad_sign = delta.grad.data.sign()

        # 如果是目标攻击，则需要让loss越来越小，因此grad取反
        if self.targeted:
            grad_sign = -grad_sign
        # 更新delta
        delta.data += eps_iter * grad_sign

        # 投影操作，即截断超出范围的值
        delta.data = torch.clamp(delta.data, -self.eps, self.eps)
        delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x
        delta.grad.data.zero_()
